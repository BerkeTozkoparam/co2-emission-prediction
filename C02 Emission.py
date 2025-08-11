#!/usr/bin/env python
# coding: utf-8

# # Random Forest Regression Model for Predicting Vehicle CO2 Emissions
# 
# This script loads a vehicle emissions dataset, selects relevant features,
# trains a Random Forest regression model to predict CO2 emissions (g/km),
# evaluates the model performance, and saves the trained model as a pickle file.
# 
# 

# In[88]:


import os
import pandas as pd
import kagglehub

# Download dataset
path = kagglehub.dataset_download("brsahan/vehicle-co2-emissions-dataset")

# Find first CSV file in the directory
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

if not csv_files:
    raise FileNotFoundError("No CSV file found in the dataset folder.")

csv_path = os.path.join(path, csv_files[0])

# Load the CSV
df = pd.read_csv(csv_path)
print(f"Loaded file: {csv_files[0]}")
print(df.head())


# # Streamlit Application for Vehicle CO₂ Emission Prediction and Carbon Tax Calculation
# 
# This app provides an interactive web interface where users can:
# - Select a vehicle's make and model from the dataset
# - View engine specifications and fuel consumption
# - Input their annual driving distance
# - Get a predicted CO₂ emission value using a pre-trained Random Forest model
# - Calculate an estimated annual carbon tax based on emissions and usage
# - Receive an eco-friendly warning if emissions are above a threshold
# 
# This transition to Streamlit allows for easy user interaction and visualization 
# without requiring direct code execution, making the model accessible to non-technical users.
# 
# 

# In[90]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the dataset
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Select relevant features and target
features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']
target = 'CO2 Emissions(g/km)'

# Drop rows with missing values in selected columns
df = df.dropna(subset=features + [target])

# Define input features (X) and target variable (y)
X = df[features]
y = df[target]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict CO2 emissions on the test set
y_pred = rf.predict(X_test)

# Calculate performance metrics: RMSE and R² score
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²) Score: {r2:.2f}")

# Save the trained model to a pickle file
with open('rf_co2_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Model successfully saved as 'rf_co2_model.pkl'!")


# In[86]:


import streamlit as st
import pandas as pd
import pickle

# Load dataset (update path if needed)
df = pd.read_csv(csv_path)

# Clean column names (remove whitespace)
df.columns = df.columns.str.strip()

# Features to use for prediction
features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']

# Load pre-trained model (pickle file)
with open('rf_co2_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Vehicle CO₂ Emission Prediction and Carbon Tax Calculator")

# Select vehicle make
selected_make = st.selectbox("Select Vehicle Make", sorted(df['Make'].unique()))

# Filter models by selected make
models_for_make = sorted(df[df['Make'] == selected_make]['Model'].unique())
selected_model = st.selectbox("Select Vehicle Model", models_for_make)

# Get specs of the selected vehicle
car_row = df[(df['Make'] == selected_make) & (df['Model'] == selected_model)].iloc[0]

st.write(f"**Engine Size:** {car_row['Engine Size(L)']} L")
st.write(f"**Number of Cylinders:** {car_row['Cylinders']}")
st.write(f"**Average Fuel Consumption:** {car_row['Fuel Consumption Comb (L/100 km)']} L/100km")

# Input for annual kilometers driven
annual_km = st.number_input("Annual Average Kilometers Driven", min_value=0, step=100)

if st.button("Predict Emissions and Calculate Tax"):
    # Prepare input for the model
    input_df = pd.DataFrame([[car_row['Engine Size(L)'], car_row['Cylinders'], car_row['Fuel Consumption Comb (L/100 km)']]], columns=features)
    
    # Predict CO2 emissions
    predicted_co2 = model.predict(input_df)[0]
    
    # Calculate carbon tax (example formula)
    tax = (predicted_co2 * annual_km / 1_000_000) * 65
    
    st.write(f"**Predicted CO₂ Emissions:** {predicted_co2:.2f} g/km")
    st.write(f"**Annual Carbon Tax:** €{tax:.2f}")
    
    if predicted_co2 > 150:
        st.warning("Your CO₂ emissions are high. Consider a more eco-friendly vehicle or public transport.")


# In[ ]:




