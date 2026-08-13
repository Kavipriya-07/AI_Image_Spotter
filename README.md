# AI Image Spotter

AI Image Spotter is a Flask dashboard for classifying an uploaded image as
**REAL** or **AI-GENERATED**. It preserves the original upload, history,
statistics, and dashboard workflow.

## Model

The application uses the existing persisted artifact:

`artifacts/real_fake_classifier.joblib`

It is a `StandardScaler + LogisticRegression` classifier. Its exact input
representation is a 32 × 32 RGB image with an additional grayscale channel.
`model/model_loader.py` uses the same preprocessing as `predict.py` and loads
the artifact once per Flask process.

The stored artifact metrics in `artifacts/metrics.json` are:

- Accuracy: 0.5642
- Precision: 0.5646
- Recall: 0.5642
- F1: 0.5635
- Confusion matrix: `[[315, 285], [238, 362]]`

These are the artifact's existing recorded results; they are not universal
AI-image-detection claims. Results may vary for images or generators outside
the dataset and training distribution.

## Run the Flask dashboard

```powershell
cd C:\Users\kavip\OneDrive\Desktop\i
venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000`, choose **Analyze Image**, select a JPG, JPEG,
PNG, or WEBP file up to 10 MB, and click **Analyze Image**.

## API

`POST /api/predict` accepts the file field `image` and returns compatible
fields including `prediction`, `confidence`, `real_probability`, and
`ai_probability`. The dashboard also receives `demo_indicators` for backward
compatibility; the values are real classifier probabilities and the decision
threshold, not fabricated quality metrics.

## Project layout

- `app.py` — Flask entry point
- `backend/routes/prediction.py` — prediction/history/statistics API
- `backend/services/prediction_service.py` — response formatting and threshold
- `model/model_loader.py` — artifact loading and exact preprocessing
- `templates/`, `static/` — preserved dashboard frontend
- `artifacts/` — trained classifier and its recorded metadata

## Dataset and limitations

The local dataset is ignored by Git. This model is trained on its available
real/fake data and is not guaranteed to detect every AI-generated image;
performance can vary for unseen generators, compressed or edited images,
screenshots, and images outside the represented domains.
