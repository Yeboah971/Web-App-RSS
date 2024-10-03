
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb  # Updated for XGBoost
from PIL import Image

st.title("Residual Frictional Angle Prediction")
nav = st.sidebar.radio("Navigation",["Home","Prediction","Info"])

if nav == "Home":
    image = Image.open('image.png') 
    st.image(image, width=800)
    st.markdown(
        """
        📈**Residual Shear Strength Prediction**
### **Predicting Residual Frictional Angle with ML**

**This web application predicts the residual frictional angle of soil based on key soil index properties: Liquid Limit, Plasticity Index, Change in Plasticity Index, and Clay Fraction.**

##  **How It's Built**

- **Data Collection and Preprocessing** 
- **Training Machine Learning Model** 
- **Building Web Application with Streamlit** 

The app workflow is:
1. Enter numeric values for LL, PI, Change in PI, and CF.
2. Click the **Predict** button to get the result.
3. Results are given in degrees.

##  **Key Features**

- **Real-time data** - Instantly predict the residual frictional angle by entering the soil index properties.
- **Accurate Predictions** - Model has the highest accuracy rate.
"""
    )

if nav == "Prediction":
    #st.header("Predict Your Frictional Angle with Index Properties")
    st.markdown(
        """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Streamlit Geotech Machine Learning App</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Function to predict the residual frictional angle
    def predict_residues(Liquid_Limit, Plasticity_Index, Change_in_Plasticity_Index, Clay_Fraction):
        # Load the XGBoost model
        model = xgb.XGBRegressor()
        model.load_model("XGBoost.json")  # Load the XGBoost model

        # Prepare the input without scaling
        input_data = np.array([[Liquid_Limit, Plasticity_Index, Change_in_Plasticity_Index, Clay_Fraction]])

        # Make a prediction using the XGBoost model
        prediction = model.predict(input_data)
        
        return prediction[0]  # Return the prediction (single value)

    # Function to handle user input and prediction
    def main():
        Liquid_Limit = st.text_input("Liquid Limit (%)", "")
        Plasticity_Index = st.text_input("Plasticity Index (%)", "")
        Change_in_Plasticity_Index = st.text_input("Change in Plasticity Index (%)", "")
        Clay_Fraction = st.text_input("Clay Fraction (%)", "")

        try:
            Liquid_Limit = float(Liquid_Limit)
            Plasticity_Index = float(Plasticity_Index)
            Change_in_Plasticity_Index = float(Change_in_Plasticity_Index)
            Clay_Fraction = float(Clay_Fraction)
        except ValueError:
            st.error("Please enter numeric value percentages for LL, PI, DPI, and CF.")
            st.stop()

        result = ""
        if st.button("Predict"):
            result = predict_residues(Liquid_Limit, Plasticity_Index, Change_in_Plasticity_Index, Clay_Fraction)
            st.success(f"The Predicted Frictional Angle is {result:.2f} degrees.")

    # Call the main function to display the input section and prediction output
    main()

elif nav == "Info":
    st.markdown(
        """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">University of Mines and Technology</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Geological Engineering Department")
    image = Image.open('Lmg.jpg') 
    st.image(image, width=800)
