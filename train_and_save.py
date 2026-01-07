import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
heart_df = pd.read_csv("heart.csv")

# Convert text columns to numbers
encoder = LabelEncoder()
for col in heart_df.columns:
    if heart_df[col].dtype == 'object':
        heart_df[col] = encoder.fit_transform(heart_df[col])

# Separate features & target
X = heart_df.drop("HeartDisease", axis=1)
y = heart_df["HeartDisease"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open("LogisticR.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained & LogisticR.pkl created successfully!")
