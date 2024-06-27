import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time

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

# Define the model
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', DecisionTreeRegressor(random_state=42))])

# Start the timer
start_time = time.time()

# Fit the model
model.fit(X_train, y_train)

# Stop the timer
end_time = time.time()

# Calculate training time
training_time = end_time - start_time
print("Training time:", round(training_time, 3), "seconds")

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", round(rmse, 5))

# Example prediction
example_input = pd.DataFrame({'venue': ["The Rose Bowl"], 'bat_team': ['South Africa'], 'bowl_team': ['England'],
                              'batsman': ['AB de Villiers'], 'bowler': ['Jofra Archer'], 'runs': [20], 'wickets': [3],
                              'overs': [8.5], 'runs_last_5': [35], 'wickets_last_5': [1],
                              'striker': [10], 'non-striker': [0]})
example_prediction = model.predict(example_input)
print("Predicted the score they can make further:", int(example_prediction[0]))
