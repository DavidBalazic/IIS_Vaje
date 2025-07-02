# PM10 Particle Prediction for Slovenian Air Quality Stations

This project implements an end-to-end machine learning pipeline to predict **PM10 particle concentrations** at multiple air quality monitoring stations across Slovenia. It covers the entire workflow from data fetching, preprocessing, validation, model training, to testing and reporting — all fully automated and reproducible.

---

## 🚀 Project Overview

Air pollution monitoring is critical for public health and environmental management. This project focuses on:

- Automatically fetching air quality data (PM10 and related variables) for multiple Slovenian stations.
- Data cleaning, validation, and preprocessing to ensure high-quality inputs.
- Training machine learning models to accurately predict PM10 particle concentrations.
- Generating detailed validation and testing reports.
- Fully integrating the pipeline with **GitHub Workflows** and **DVC** for versioning, reproducibility, and CI/CD.

---

## 📈 Features

- **Automated data ingestion:** Periodically fetches raw air quality data from official sources.
- **Robust preprocessing pipeline:** Handles missing values, outliers, and feature engineering.
- **Data validation:** Ensures data quality through automated checks and tests.
- **Model training & evaluation:** Trains ML models on prepared data, evaluates performance.
- **Report generation:** Produces validation and test reports accessible online.
- **Reproducibility:** Full pipeline management with DVC and GitHub Actions.
- **Multi-station support:** Predictions tailored to different Slovenian air quality stations.

---

## 📡 Live Reports

Detailed reports on data validation, model testing, and performance are published here:

[https://iis-vaje.netlify.app/](https://iis-vaje.netlify.app/)

---

## 🔧 Tech Stack

- **Python** with libraries: pandas, scikit-learn, numpy, tensorflow, evidently, great-expectations, evidently, etc.
- **DVC** for data and pipeline version control
- **GitHub Actions** for CI/CD automation
- **Machine Learning Models:** time-series forecasting with LSTM
- **Data Sources:** Official Slovenian air quality data API

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- DVC installed (`pip install dvc`)
- Git

### Installation
```bash
git clone https://github.com/DavidBalazic/IIS_Vaje.git
cd IIS_Vaje
dvc pull
dvc repro
```