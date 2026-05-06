# Heart Attack Risk Classification App

A Streamlit web app that predicts heart attack risk using a Random Forest model.

## Files
- `app.py` — Streamlit application
- `rf_model.pkl` — Trained Random Forest model
- `requirements.txt` — Python dependencies

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (Free)

1. Push all files to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"**
4. Select your repo, branch (`main`), and set **Main file path** to `app.py`
5. Click **"Deploy!"**

Your app will be live at `https://<your-app>.streamlit.app`

## Deploy on Hugging Face Spaces (Free)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Streamlit** as the SDK
3. Upload `app.py`, `rf_model.pkl`, and `requirements.txt`
4. The space auto-builds and goes live

## Deploy on Railway / Render

Create a `Procfile`:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
Then connect your GitHub repo in Railway or Render dashboard.

## Input Features

| Feature | Type | Description |
|---|---|---|
| Age | Number | Patient age (20–100) |
| RestingBP | Number | Resting blood pressure |
| Cholesterol | Number | Serum cholesterol (mg/dl) |
| MaxHR | Number | Maximum heart rate achieved |
| Oldpeak | Number | ST depression induced by exercise |
| FastingBS | Select | Fasting blood sugar > 120 mg/dl (0/1) |
| Gender | Select | M / F |
| ChestPainType | Select | ATA / NAP / ASY / TA |
| RestingECG | Select | Normal / ST / LVH |
| ExerciseAngina | Select | Y / N |
| ST_Slope | Select | Up / Flat / Down |
