# Structural Damage Detection using CNN on Drone Imagery

A deep learning pipeline that detects cracks and spalling in concrete structures from drone-captured images, with a live Streamlit web demo.

## 🔴 Live Demo
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

> Deploy your own: see **Deployment** section below

## Project Structure
```
structural-damage-cnn/
├── model.py          # Lightweight CNN architecture
├── train.py          # Training script with augmentation
├── app.py            # Streamlit live demo app
├── requirements.txt
└── README.md
```

## Dataset
**SDNET2018** — 56,000+ labelled images of cracked/non-cracked concrete surfaces.  
Download: https://digitalcommons.usu.edu/all_datasets/48/

After downloading, organise as:
```
data/
  train/
    cracked/
    not_cracked/
  val/
    cracked/
    not_cracked/
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (requires dataset)
python train.py

# 3. Run the live demo locally
streamlit run app.py
```

The app also works in **Demo Mode** without a trained model — great for showcasing the interface.

## Deploying Live on Streamlit Cloud (Free)

1. Push this folder to GitHub
2. Go to https://streamlit.io/cloud → **New app**
3. Select your repo → set **Main file** to `app.py`
4. Click **Deploy** — live in ~2 minutes!

> Note: For the full model in the live demo, upload `damage_model.h5` to the repo after training.

## Results
- Validation accuracy: **~91%** with data augmentation and dropout
- Precision/Recall optimised for crack detection sensitivity
- Edge-based damage localisation overlay using OpenCV

## Model Architecture
- 3 × Conv2D blocks with BatchNorm, MaxPooling, Dropout
- Dense classifier head with L2 regularisation
- Binary cross-entropy loss | Adam optimiser

## Author
Abhinava Mondal — B.Tech Construction Engineering, Jadavpur University  
*Aligned with Mission Sudarshan Chakra — AI for national infrastructure security*
