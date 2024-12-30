import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Initialize label encoders
label_encoders = {}

# Encode categorical features
categorical_columns = ['Gender', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Blood Pressure', 'Cholesterol Level', 'Disease']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare features and target for disease prediction
X_disease = df.drop(['Disease', 'Charges'], axis=1)
y_disease = df['Disease']

# Train disease prediction model
disease_model = RandomForestClassifier()
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(X_disease, y_disease, test_size=0.3, random_state=42)
disease_model.fit(X_train_disease, y_train_disease)

# Prepare features and target for cost prediction
X_cost = df.drop(['Disease', 'Charges'], axis=1)
y_cost = df['Charges']

# Train cost prediction model
cost_model = RandomForestRegressor()
X_train_cost, X_test_cost, y_train_cost, y_test_cost = train_test_split(X_cost, y_cost, test_size=0.3, random_state=42)
cost_model.fit(X_train_cost, y_train_cost)

# Save models and label encoders
with open('disease_model.pkl', 'wb') as file:
    pickle.dump(disease_model, file)

with open('cost_model.pkl', 'wb') as file:
    pickle.dump(cost_model, file)

with open('label_encoders.pkl', 'wb') as file:
    pickle.dump(label_encoders, file)

print("Models and label encoders saved successfully!")
