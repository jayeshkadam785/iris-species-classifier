# Iris Species Classifier

A machine learning web application that classifies Iris flowers into three species — *Iris setosa*, *Iris versicolor*, and *Iris virginica* — based on sepal and petal measurements. Built with scikit-learn and deployed as a serverless app on Vercel.

**Live Demo:** 

## Overview

This project uses the classic Fisher Iris dataset to train a Support Vector Machine (SVM) classifier that predicts flower species from four measurements: sepal length, sepal width, petal length, and petal width. The trained model is served through a Flask API, with an interactive web interface for real-time predictions.

## Features

- Interactive slider-based UI for entering flower measurements
- Real-time species prediction with confidence scores for all three classes
- Lightweight serverless backend (Flask on Vercel)
- Trained on the full 150-sample Iris dataset with 100% test accuracy

## Tech Stack

| Layer | Technology |
|---|---|
| Model | scikit-learn (SVM, linear kernel) |
| Backend | Flask (Python serverless function) |
| Frontend | HTML, CSS, JavaScript |
| Deployment | Vercel |

## Project Structure

```
iris-species-classifier/
├── api/
│   ├── index.py              # Flask API — loads model and serves predictions
│   ├── model.pkl              # Trained SVM classifier
│   └── label_encoder.pkl      # Encodes/decodes species labels
├── public/
│   └── index.html             # Frontend interface
├── requirements.txt            # Python dependencies
└── vercel.json                 # Deployment configuration
```

## Dataset

The Iris dataset contains 150 samples, 50 from each of three species, with four features per sample:

- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

## Model Performance

| Model | Test Accuracy | 5-Fold CV Accuracy |
|---|---|---|
| Logistic Regression | 96.7% | 97.3% |
| Decision Tree | 93.3% | 95.3% |
| Random Forest | 90.0% | 96.7% |
| K-Nearest Neighbors | 100% | 97.3% |
| **SVM (linear)** | **100%** | **98.0%** |

The final model (SVM, linear kernel) was selected for deployment and retrained on the full dataset for production use.

## API Reference

**POST** `/api/predict`

Request body:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:
```json
{
  "species": "Iris-setosa",
  "confidence": {
    "Iris-setosa": 97.63,
    "Iris-versicolor": 1.52,
    "Iris-virginica": 0.85
  }
}
```

## Running Locally

```bash
git clone https://github.com/jayeshkadam785/iris-species-classifier.git
cd iris-species-classifier
pip install -r requirements.txt
```

Run the API:
```python
from api.index import app
app.run(port=5000, debug=True)
```

Then open `public/index.html` in a browser.

## Deployment

Deployed on Vercel using serverless Python functions. Any push to `main` triggers an automatic redeploy.

```bash
npm install -g vercel
vercel --prod
```

## Author

**Jayesh Kadam**
B.Tech AI & Data Science student at Karmaveer Bhaurao Patil College of Engineering (KBPCOES), Satara
[GitHub](https://github.com/jayeshkadam785)
