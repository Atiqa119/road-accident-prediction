import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
st.title("Road Accident Risk Prediction")
st.write("This is a simple Streamlit app for predicting accident risk.")


# Load trained model
model = pickle.load(open("linear_model.pkl", "rb"))

# Upload dataset for visualization
uploaded_file = st.file_uploader("Upload CSV file for trend analysis", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Display dataset summary
    st.write("### Dataset Summary")
    st.write(df.describe())

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Pairplot visualization
    st.write("### Feature Pairplot")
    fig = sns.pairplot(df)
    st.pyplot(fig)

    # Scatter plot of user-selected features
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    x_feature = st.selectbox("Select Feature for X-Axis", numeric_cols)
    y_feature = st.selectbox("Select Feature for Y-Axis", numeric_cols)

    if x_feature and y_feature:
        st.write(f"### Scatter Plot: {x_feature} vs {y_feature}")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x=df[x_feature], y=df[y_feature], ax=ax)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"{x_feature} vs {y_feature}")
        st.pyplot(fig)


# User Inputs
road_type = st.selectbox("Road Type", ["Urban", "Rural", "Highway"])
traffic_volume = st.number_input("Traffic Volume (vehicles/hour):", min_value=0)
speed_limit = st.number_input("Speed Limit (km/h):", min_value=10)
speed_violations = st.number_input("Speed Violations (per day):", min_value=0)
num_lanes = st.number_input("Number of Lanes:", min_value=1)
road_surface = st.selectbox("Road Surface Condition", ["Dry", "Wet", "Icy"])
weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Foggy", "Snowy"])
visibility = st.number_input("Visibility Distance (meters):", min_value=0)
temperature = st.number_input("Temperature (°C):", min_value=-30, max_value=50)
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
past_accidents = st.number_input("Past Accident Count:", min_value=0)
past_severity = st.number_input("Severity of Past Accidents (1-10):", min_value=1, max_value=10)
response_time = st.number_input("Emergency Response Time (minutes):", min_value=1)

# Encode categorical inputs as numerical (similar to training data)
road_type_map = {"Urban": 0, "Rural": 1, "Highway": 2}
road_surface_map = {"Dry": 0, "Wet": 1, "Icy": 2}
weather_map = {"Clear": 0, "Rainy": 1, "Foggy": 2, "Snowy": 3}
time_of_day_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

# Convert categorical inputs to numerical
road_type_num = road_type_map[road_type]
road_surface_num = road_surface_map[road_surface]
weather_num = weather_map[weather]
time_of_day_num = time_of_day_map[time_of_day]

# Prediction Button
if st.button("Predict"):
    input_data = np.array([[road_type_num, traffic_volume, speed_limit, speed_violations, 
                            num_lanes, road_surface_num, weather_num, visibility, 
                            temperature, time_of_day_num, past_accidents, past_severity, 
                            response_time]])
    
    prediction = model.predict(input_data)
    st.success(f"Predicted Accident Probability: {prediction[0]:.2f}")