# MLOps Lab 5 - Feature Engineering Pipeline

## Overview
This project implements a feature engineering pipeline for traffic data using TensorFlow. The pipeline processes raw input features and generates transformed features suitable for model training.

---

## Modifications Made

### 1. Custom Feature Engineering
- Added a new feature **temp_category_xf**  
  - Categorizes temperature as above or below the mean  
  - Helps capture relative temperature patterns  

### 2. Feature Interaction
- Introduced **temp_cloud_interaction_xf**  
  - Combines temperature and cloud coverage  
  - Captures interaction between weather conditions  

### 3. Modified Feature Scaling
- Applied different scaling techniques:
  - Z-score normalization for continuous features  
  - Min-max scaling for range-based features  

### 4. Improved Feature Representation
- Implemented hashing-based encoding for categorical features  
- Preserved time-based features (hour, day, month, etc.)  




## How to Run

1. Install dependencies:
```bash
pip install tensorflow
```
2. Run file 
```bash
python run_pipeline.py
```
