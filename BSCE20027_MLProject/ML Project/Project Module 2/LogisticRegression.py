import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle  # Import the pickle module

# Load data
data = pd.read_csv('t20.csv')

# Define a threshold for classification (for example, the median total score)
threshold = data['total'].median()

# Classify matches based on the threshold
data['class'] = (data['total'] > threshold).astype(int)

# Separate features and target variable
X = data[['bat_team', 'bowl_team', 'venue', 'runs', 'wickets', 'overs',
          'runs_last_5', 'wickets_last_5', 'run_rate', 'striker', 'strike_rate', 'non-striker']]
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Define the model
model = LogisticRegression(random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 5))

# Example prediction
example_input = pd.DataFrame({'venue': ["Lahore"], 'bat_team': ['Australia'], 'bowl_team': ['India'],
                              'runs': [129], 'wickets': [5], 'overs': [10], 'runs_last_5': [0], 'wickets_last_5': [0],
                              'run_rate': [12.9], 'striker': [0], 'strike_rate': [0], 'non-striker': [0]})
example_prediction = model.predict(example_input)
print("Predicted class:", example_prediction[0])

# Define the path where you want to save the model
model_path = "./logistic_regression_model"

# Save the model
with open(model_path, 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully at:", model_path)
