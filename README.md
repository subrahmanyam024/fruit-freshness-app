# fruit-freshness-app

# 🍎 Fruit Freshness Prediction App

A deep learning web app that predicts whether a fruit is **Fresh** or **Rotten** using image classification. Built using **TensorFlow**, **Keras**, and **Streamlit**.

---

## 📌 Overview

This project allows users to upload a fruit image and predict its freshness using a pre-trained deep learning model. The model was trained using a custom dataset on **Google Colab**, saved as `best_model.keras`, and used in a local **Streamlit app** developed in **VS Code**.

---

## 🧠 Model Details

- **Model Type**: CNN (Convolutional Neural Network)
- **Framework**: TensorFlow / Keras
- **Trained On**: Google Colab
- **Classes**: `Fresh`, `Rotten`

The model file `best_model.keras` is stored locally in the `model/` directory and is loaded in the Streamlit app for predictions.

---

## 🗂️ Project Structure

fruit-freshness-app/
├── app.py # Streamlit app script
├── model/
│ └── best_model.keras # Trained model saved from Colab
├── requirements.txt # Python dependencies
├── .gitignore # Files to ignore in Git
└── README.md # Project documentation


---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/subrahmanyam024/fruit-freshness-app.git
cd fruit-freshness-app

2. Install dependencies
Make sure you have Python 3.10 or lower (due to TensorFlow compatibility).

pip install -r requirements.txt

3. Run the app

streamlit run app.py

🖼 App Features
📤 Upload an image of a fruit

🔍 Automatically predicts if it’s Fresh or Rotten

📊 Displays model prediction confidence

🧠 Uses a pre-trained model from Google Colab

✅ Notes
No training happens in the app. The model was pre-trained in Google Colab and used as-is.

Make sure the best_model.keras file is present in the model/ folder before running the app.

📌 Future Improvements
Add drag & drop image support

Use a lighter model for mobile-friendly inference

Include more fruit types and multi-class support

👨‍💻 Developed By
Subrahmanyam Thota
GitHub: subrahmanyam024
