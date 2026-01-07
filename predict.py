import numpy as np
import pandas as pd
import pickle

# Load Dataset
heart_df = pd.read_csv("heart.csv")

# Separate X and y
X = heart_df.drop('HeartDisease', axis=1)
y = heart_df['HeartDisease']

# Load Saved Logistic Regression Model
with open('LogisticR.pkl', 'rb') as file:
    lr_model = pickle.load(file)

print("✅ Logistic Regression model loaded successfully.")

# Take input (sample patient)
input_data = (63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0)
input_data = np.asarray(input_data).reshape(1, -1)

# Prediction
prediction = lr_model.predict(input_data)

if prediction[0] == 1:
    print("⚠️ Person has Heart Disease")
else:
    print("✅ Person does NOT have Heart Disease")
