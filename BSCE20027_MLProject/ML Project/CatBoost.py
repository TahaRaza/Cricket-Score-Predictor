import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time

# Load data
data = pd.read_csv('t20_abc.csv')

# Encode categorical features
cat_columns = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
for col in cat_columns:
    data[col] = pd.Categorical(data[col]).codes

# Separate features and target variable
X = data[['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'runs', 'wickets', 'overs', 'striker', 'non-striker']]
y = data['score_difference']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model
model = CatBoostRegressor(iterations=10000, learning_rate=0.00001, depth=6, random_seed=42)

# Start the timer
start_time = time.time()

# Fit the model
model.fit(X_train, y_train, cat_features=['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler'])

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

# Define the path where you want to save the model
model_path = "./catboost_model"

# Save the model
model.save_model(model_path)

print("Model saved successfully at:", model_path)
