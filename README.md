# MLOps Lab 5 - Feature Engineering Pipeline

## Description
This project implements a TensorFlow Transform (TFT) pipeline for feature engineering.

## Modifications Made
- Modified scaling for temperature (MinMax scaling instead of Z-score)
- Added new feature: temperature category (above/below mean)
- Added feature interaction between temperature and cloud coverage

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the pipeline using the notebook or script