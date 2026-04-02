from __future__ import annotations

import argparse
import csv
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import torch
from monai.losses import DiceCELoss
from tqdm import tqdm
from monai.utils import set_determinism

from .dataset import PreprocessedACDCDataset, create_loader
from .metrics import evaluate_multiclass
from .models import build_model
from .settings import model_run_dir

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _validate(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    all_scores: List[float] = []
    running_val_loss = 0.0
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = model(image)
            val_loss = criterion(logits, label.unsqueeze(1))
            running_val_loss += val_loss.item()
            pred = logits.argmax(dim=1).cpu().numpy()
            target = label.cpu().numpy()
            for p, t in zip(pred, target):
                metrics = evaluate_multiclass(p, t)
                all_scores.append(float(metrics["mean_dice"]))
    return {
        "mean_dice": float(sum(all_scores) / max(len(all_scores), 1)),
        "val_loss": running_val_loss / max(len(loader), 1),
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    best_dice: float,
    best_epoch: int,
    history: List[Dict[str, float]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_mean_dice": best_dice,
            "best_epoch": best_epoch,
            "history": history,
        },
        path,
    )


def _save_history_csv(history: List[Dict[str, float]], path: Path) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mean_dice"])
        writer.writeheader()
        writer.writerows(history)


def _save_training_plot(history: List[Dict[str, float]], path: Path) -> Optional[str]:
    if plt is None or not history:
        return None

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item.get("val_loss", 0.0)) for item in history]
    val_dice = [float(item["val_mean_dice"]) for item in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    p1, = ax1.plot(epochs, train_loss, color="tab:red", label="Train Loss")
    p2, = ax1.plot(epochs, val_loss, color="tab:orange", linestyle="--", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    p3, = ax2.plot(epochs, val_dice, color="tab:blue", label="Val Mean Dice")
    ax2.set_ylabel("Val Mean Dice", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    lines = [p1, p2, p3]
    ax1.legend(lines, [l.get_label() for l in lines])

    fig.suptitle("Training Curves")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _maybe_resume(
    resume_path: Optional[Path],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
) -> tuple[int, float, int, List[Dict[str, float]]]:
    if resume_path is None:
        return 1, -1.0, 0, []

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best_dice = float(checkpoint.get("best_val_mean_dice", -1.0))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    history = list(checkpoint.get("history", []))
    return start_epoch, best_dice, best_epoch, history


def train_model(
    model_name: str,
    epochs: int = 1000,
    batch_size: int = 16,
    lr: float = 1e-4,
    num_workers: int = 8,
    preprocessed_root: Path | None = None,
    augment: bool = True,
    seed: int = 42,
    resume_checkpoint: Path | None = None,
    patience: int = 50,
) -> Dict[str, object]:

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed=seed)

    run_dir = model_run_dir(model_name)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = preprocessed_root if preprocessed_root else None
    train_ds = (
        PreprocessedACDCDataset(
            "train",
            root_dir=dataset_root,
            augment=augment,
        )
        if dataset_root
        else PreprocessedACDCDataset(
            "train",
            augment=augment,
        )
    )
    val_ds = (
        PreprocessedACDCDataset("val", root_dir=dataset_root, augment=False)
        if dataset_root
        else PreprocessedACDCDataset("val", augment=False)
    )
    train_loader = create_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_loader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # เพิ่ม Scheduler เพื่อลด LR ลงครึ่งหนึ่ง (factor=0.5) หาก Validation Dice ไม่เพิ่มขึ้นติดต่อกัน 15 Epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15
    )
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = torch.amp.GradScaler('cuda')

    start_epoch, best_dice, best_epoch, history = _maybe_resume(
        resume_checkpoint,
        model,
        optimizer,
        scheduler,
        device,
    )

    epochs_no_improve = 0
    effective_batch_size = 16
    accum_iter = max(1, effective_batch_size // batch_size)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(tqdm(train_loader, desc=f"{model_name} epoch {epoch}/{epochs}")):
            image = batch["image"].to(device)
            label = batch["label"].to(device).unsqueeze(1)

            with torch.amp.autocast('cuda'):
                logits = model(image)
                loss = criterion(logits, label) / accum_iter

            scaler.scale(loss).backward()

            if ((i + 1) % accum_iter == 0) or (i + 1 == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item() * accum_iter)

        val_metrics = _validate(model, val_loader, device, criterion)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / max(len(train_loader), 1),
            "val_loss": val_metrics["val_loss"],
            "val_mean_dice": val_metrics["mean_dice"],
        }
        history.append(epoch_record)
        print(f"Epoch {epoch} | Train Loss: {epoch_record['train_loss']:.4f} | Val Loss: {epoch_record['val_loss']:.4f} | Val Dice: {epoch_record['val_mean_dice']:.4f}")
        
        # สั่งให้ Scheduler อัปเดตและเช็คว่าควรลด Learning Rate หรือยัง
        scheduler.step(val_metrics["mean_dice"])
        
        # พิมพ์บอก LR ปัจจุบันเพื่อให้ทราบหากมีการลดค่าเกิดขึ้น
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")

        _save_checkpoint(
            checkpoint_dir / "last_checkpoint.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_dice,
            best_epoch,
            history,
        )
        torch.save(model.state_dict(), checkpoint_dir / "last.pt")
        if val_metrics["mean_dice"] > best_dice:
            best_dice = val_metrics["mean_dice"]
            best_epoch = epoch
            epochs_no_improve = 0
            print(f"New best model found at epoch {epoch} with Val Dice: {best_dice:.4f}! Saving model...")
            _save_checkpoint(
                checkpoint_dir / "best_checkpoint.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_dice,
                best_epoch,
                history,
            )
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
        else:
            epochs_no_improve += 1
            
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
            break

    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    _save_history_csv(history, run_dir / "history.csv")
    plot_path = _save_training_plot(history, run_dir / "training_curves.png")
    summary = {
        "model": model_name,
        "best_val_mean_dice": best_dice,
        "best_epoch": best_epoch,
        "run_dir": str(run_dir),
        "augment": augment,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "history_json": str(run_dir / "history.json"),
        "history_csv": str(run_dir / "history.csv"),
        "training_plot": plot_path,
        "best_model_path": str(checkpoint_dir / "best.pt"),
        "best_checkpoint_path": str(checkpoint_dir / "best_checkpoint.pt"),
        "last_model_path": str(checkpoint_dir / "last.pt"),
        "last_checkpoint_path": str(checkpoint_dir / "last_checkpoint.pt"),
    }
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn", "transformer", "segmamba"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--preprocessed-root", type=Path, default=None)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement before stopping (0 to disable)")
    args = parser.parse_args()
    print(
        json.dumps(
            train_model(
                args.model,
                args.epochs,
                args.batch_size,
                args.lr,
                args.num_workers,
                args.preprocessed_root,
                augment=not args.disable_augmentation,
                seed=args.seed,
                resume_checkpoint=args.resume_checkpoint,
                patience=args.patience,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
