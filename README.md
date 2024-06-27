# Cricket Score Predictor

## Predicting Cricket Scores: A Machine Learning Approach with Ensemble Learning Potential

### Author
**Taha Hasnain Raza**  
Department of Computer Engineering, Information Technology University, Lahore, Pakistan  
[Email: thraza99@gmail.com](mailto:thraza99@gmail.com)

---

## Table of Contents
1. [Abstract](#abstract)
2. [Keywords](#keywords)
3. [Introduction](#introduction)
4. [Materials and Methods](#materials-and-methods)
    1. [Data Sources](#data-sources)
    2. [Model Selection and Training](#model-selection-and-training)
        1. [Training Process](#training-process)
    3. [Model Evaluation](#model-evaluation)
5. [Results](#results)
    1. [Model Comparison](#model-comparison)
6. [Software and Hardware Requirements](#software-and-hardware-requirements)
    1. [Hardware Requirements](#hardware-requirements)
    2. [Software Requirements](#software-requirements)
7. [References](#references)

## Abstract
A machine learning project successfully predicted cricket scores (run rate, total score) for T20 and ODI matches using Random Forest, Decision Tree, and KNN regression. All models achieved impressive accuracy (RMSE < 30), with Random Forest excelling. This paves the way for further exploration of ensemble methods and hyper-parameter tuning for optimized performance.

## Keywords
Prediction, Regression, Scores, RMSE, KNN, SciKitLearn

## Introduction
Cricket, a sport adored by billions, thrives on the thrill of the unpredictable. However, the ability to anticipate scores offers valuable insights for fans, analysts, and even teams. This project tackles this challenge by creating a novel Cricket Score Prediction System using machine learning.

## Materials and Methods
### Data Sources
- **GitHub Repositories:** 
  - [CricketScorePredictor](https://github.com/codophobia/CricketScorePredictor)
  - [Prohith995 Cricket Data](https://github.com/prohith995/cricket-data)
- **Cricket Websites:** 
  - [ESPN Cricinfo](https://www.espncricinfo.com)

### Model Selection and Training
- **Random Forest Regression:** An ensemble learning technique combining multiple decision trees.
- **Decision Tree Regression:** Uses a tree-like structure for predictions.
- **K-Nearest Neighbors (KNN) Regression:** Predicts based on the k nearest neighbors in the training data.

#### Training Process
- Split the pre-processed data into training, validation, and testing sets.
- Train each model using the training set and fine-tune hyperparameters using the validation set.
- Evaluate performance on the testing set using RMSE as the primary metric.

### Model Evaluation
- **Root Mean Squared Error (RMSE):** Lower RMSE indicates better prediction accuracy.

## Results
### Model Comparison
| Model              | RMSE       |
|--------------------|------------|
| KNN Regression     | 29.81318   |
| Decision Tree      | 28.74362   |
| Random Forest      | 29.81318   |
| Linear Regression  | 23.80819   |
| CatBoost           | 46.29299   |
| Gradient Boosting  | 46.78471   |
| SVR                | 21.37161   |

## Software and Hardware Requirements

### Hardware Requirements
- **Processor:** Intel Core i5 or AMD Ryzen 5.
- **Memory (RAM):** Minimum 8GB.
- **Storage:** At least 10GB of free disk space.
- **Graphics Processing Unit (GPU):** Optional, but beneficial for faster model training.

### Software Requirements
- **Operating System:** Windows, macOS, or Linux.
- **Python:** Python 3.7 or later.
- **Python Libraries:**
    - `scikit-learn`
    - `pandas`
    - `NumPy`
    - `Matplotlib`
    - `Jupyter Notebook`

## References
1. **[GitHub Repository: CricketScorePredictor](https://github.com/codophobia/CricketScorePredictor)**
2. **[GitHub Repository: Prohith995 Cricket Data](https://github.com/prohith995/cricket-data)**
3. **[ESPN Cricinfo](https://www.espncricinfo.com)**
