import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('t20_abc.csv')

# Separate features and target variable
X = data[['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'runs', 'wickets', 'overs', 'runs_last_5',
          'wickets_last_5', 'striker', 'non-striker']]
y = data['score_difference']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
categorical_features = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Define hyperparameter grid for tuning
param_grid = {
    'regressor__n_estimators': [10, 20, 30],  # Number of trees in the forest
    'regressor__max_depth': [None, 10, 20, 30]  # Maximum depth of the tree
}

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=2, scoring='neg_mean_squared_error', verbose=1)

# Start the timer
start_time = time.time()

# Fit the model with hyperparameter tuning
grid_search.fit(X_train, y_train)

# Stop the timer
end_time = time.time()

# Calculate training time
training_time = end_time - start_time
print("Training time with GridSearchCV:", round(training_time, 3), "seconds")

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions on test set using the best model
y_pred = best_model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", round(rmse, 5))

# Example prediction
example_input = pd.DataFrame({'venue': ["The Rose Bowl"], 'bat_team': ['South Africa'], 'bowl_team': ['England'],
                              'batsman': ['AB de Villiers'], 'bowler': ['Jofra Archer'], 'runs': [50], 'wickets': [3],
                              'overs': [8.5], 'runs_last_5': [35], 'wickets_last_5': [1],
                              'striker': [30], 'non-striker': [20]})
example_prediction = best_model.predict(example_input)
print("Predicted the score they can make further:", int(example_prediction[0]))

# Extract parameter values and RMSE values from GridSearchCV results
params = grid_search.cv_results_['params']
mean_rmse = np.sqrt(-grid_search.cv_results_['mean_test_score'])

# Print the best parameters found
print("Best Parameters:")
print(grid_search.best_params_)

# Plot RMSE vs. Hyperparameters
plt.figure(figsize=(8, 6))
for i in range(len(params)):
    plt.scatter(params[i]['regressor__n_estimators'], params[i]['regressor__max_depth'], c='blue', s=mean_rmse[i] * 100, label='RMSE')
plt.xlabel('Number of Estimators')
plt.ylabel('Max Depth')
plt.title('RMSE vs. Hyperparameters')
plt.colorbar(label='RMSE')
plt.legend()
plt.show()
