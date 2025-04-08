# PCA and Logistic Regression on Breast Cancer Dataset

## Project Overview

This project, completed as a data analysis task at the Anderson Cancer Center, demonstrates the use of **Principal Component Analysis (PCA)** to identify essential variables from the breast cancer dataset provided by `sklearn.datasets`.**Logistic Regression** is implemented for predictive modeling based on the reduced components.

---

## Files Included

- `Anderson_PCA.py` — Main Python script performing all analysis and saving outputs.
- `README.md` — This documentation file.

---

## Features

1. **Principal Component Analysis (PCA)**  
   - Standardizes features before applying PCA.
   - Reduces the original dataset to 2 principal components.
   - Saves the 2D PCA representation to `pca_components.csv`.

2. **Visualization**  
   - A clear scatter plot visualizing the PCA-reduced data, colored by cancer class.
   - Saved as `pca_scatter.png`.

3. **Logistic Regression (Bonus)**  
   - Trains a logistic regression model on the reduced dataset.
   - Saves predictions and actual labels to `predictions.csv`.

---

## Output Location

All output files (CSV and PNG) are **automatically saved in the same folder** as the Python script, regardless of where the script is executed. This is done by dynamically detecting the script's location using:

python
script_dir = os.path.dirname(os.path.abspath(__file__))

## How to Run
Make sure you have Python 3 and the following libraries installed:


pip install numpy pandas matplotlib seaborn scikit-learn
Run the script:


python Anderson_PCA.py
Check the output files (.csv and .png) in the same folder as the script.
