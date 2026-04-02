# Demo Folders

- Put `.pt` model weights in `demo/weights/`
- Put input `.nii` or `.nii.gz` images in `demo/data/`
- Predictions will be written to `demo/output/`

Example:

```bash
python demo/predict_pt.py --model transformer --image demo/data/case001.nii.gz --weights demo/weights/best.pt
```
