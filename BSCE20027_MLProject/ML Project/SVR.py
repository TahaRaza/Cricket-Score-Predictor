import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time

# Load data
data = pd.read_csv('t20_abc.csv')

# Encode categorical features
label_encoders = {}
cat_columns = ['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler']
for col in cat_columns:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])

# Separate features and target variable
X = data[['venue', 'bat_team', 'bowl_team', 'batsman', 'bowler', 'runs', 'wickets', 'overs', 'striker', 'non-striker']]
y = data['score_difference']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the SVR model
model = SVR(kernel='rbf', C=100, epsilon=0.1)  # You can adjust the parameters as needed

# Start the timer
start_time = time.time()

# Fit the model
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error:", round(rmse, 5))

# Example prediction
example_input = pd.DataFrame({'venue': ["The Rose Bowl"], 'bat_team': ['South Africa'], 'bowl_team': ['England'],
                              'batsman': ['AB de Villiers'], 'bowler': ['Jofra Archer'], 'runs': [50], 'wickets': [3],
                              'overs': [9],
                              'striker': [30], 'non-striker': [20]})

# Encode the example input using the same label encoders
for col in cat_columns:
    example_input[col] = example_input[col].map(lambda x: label_encoders[col].transform([x])[0]
                                                if x in label_encoders[col].classes_ else -1)

# Make predictions for the example input
example_prediction = model.predict(example_input)
print("Predicted the score they can make further:", int(example_prediction[0]))

# Stop the timer
end_time = time.time()

# Calculate training time
training_time = end_time - start_time
print("Training time in total:", round(training_time, 3), "seconds")

# Save the SVR model using joblib
filename = 'my_svr_model.pkl'  # Adjust the filename as needed
joblib.dump(model, filename)
print(f"Model saved to {filename}")
