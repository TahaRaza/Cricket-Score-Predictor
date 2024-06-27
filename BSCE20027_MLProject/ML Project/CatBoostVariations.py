import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('t20_abc.csv')

# Function for data preprocessing (reusable)
def preprocess_data(data):
    """
    Encodes categorical features and separates features and target variable.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        tuple: A tuple containing the preprocessed data (X, y)
    """

    cat_columns = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
    for col in cat_columns:
        data[col] = pd.Categorical(data[col]).codes

    X = data[['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'runs', 'wickets', 'overs', 'striker', 'non-striker']]
    y = data['score_difference']

    return X, y

# Preprocess data
X, y = preprocess_data(data.copy())  # Copy data to avoid modifying original

# Split the data for hyperparameter tuning (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'iterations': [1000, 2000],  # Number of boosting iterations
    'learning_rate': [0.01, 0.001],  # Learning rate
    'depth': [4, 6],  # Maximum tree depth
    'random_strength': [16, 32]  # L2 regularization coefficient
}

# Define the model (without initial hyperparameters)
model = CatBoostRegressor(random_seed=42)

# Perform grid search cross-validation for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)  # Use negative MSE for minimization
grid_search.fit(X_train, y_train, eval_set=(X_val, y_val))  # Evaluate on validation set

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Extract best hyperparameter values
best_params = grid_search.best_params_

print("Best Hyperparameters:", best_params)

# Make predictions on test set using the best model
X_test, y_test = train_test_split(X.drop(X_val.index), y.drop(y_val.index), test_size=0.3, random_state=42)  # Split remaining data for testing
y_pred = best_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", round(rmse, 5))

# Example prediction
example_input = pd.DataFrame({'venue': ["The Rose Bowl"], 'bat_team': ['South Africa'], 'bowl_team': ['England'],
                              'batsman': ['AB de Villiers'], 'bowler': ['Jofra Archer'], 'runs': [50], 'wickets': [3],
                              'overs': [8.5], 'runs_last_5': [35], 'wickets_last_5': [1],
                              'striker': [30], 'non-striker': [20]})
example_input['venue'] = pd.Categorical(example_input['venue'], categories=data['venue'].unique()).codes
example_input['bat_team'] = pd.Categorical(example_input['bat_team'], categories=data['bat_team'].unique()).codes
example_input['bowl_team'] = pd.Categorical(example_input['bowl_team'], categories=data['bowl_team'].unique()).codes
example_input['batsman'] = pd.Categorical(example_input['batsman'], categories=data['batsman'].unique()).codes
example_input['bowler'] = pd.Categorical(example_input['bowler'], categories=data['bowler'].unique()).codes

example_prediction = model.predict(example_input)
print("Predicted the score they can make further:", int(example_prediction[0]))
# Save the best model
model_path = "./catboost_model_tuned"
best_model.save_model(model_path)

print("Model saved successfully at:", model_path)

# Visualize hyperparameter tuning results (optional)
scores = grid_search.cv_results_['mean_test_score']
scores = np.absolute(scores)  # Convert to positive for plotting

for param_name, values in param_grid.items():
    plt.figure()
    plt.scatter(values, scores)
    plt.xlabel(param_name)
    plt.ylabel('Mean Absolute Squared Error (lower is better)')
    plt.title(f'Hyperparameter Tuning: {param_name}')
    plt.show()
