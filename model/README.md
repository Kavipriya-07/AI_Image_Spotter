# Model integration

`model_loader.py` loads the trained `artifacts/real_fake_classifier.joblib` once
per Flask process and applies the same 32 × 32 RGB-plus-grayscale preprocessing
used by `predict.py`. The Flask response contract remains unchanged.
