import pandas as pd
import streamlit as st
import joblib

st.title("Loan Default Prediction App")


model = joblib.load('model.pkl')

def main():
    with st.form("input_form"):
        st.write("Input Data")
        employed = st.checkbox("Employed")
        bank_balance = st.number_input(
    "Bank Balance", value=None, placeholder="Type a number...")
        annual_salary = st.number_input(
    "Annual Salary", value=None, placeholder="Type a number...")

        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            input_data = pd.DataFrame({
                'Employed': [1 if employed == "Yes" else 0],
                'Bank Balance': [float(bank_balance)],
                'Annual Salary': [float(annual_salary)]
            })

            # Make prediction
            prediction = model.predict(input_data)
            
            result =  "Default" if prediction[0] == 1 else "No Default"

            st.write(f"Prediction: {result}")
            st.write(f"Probability of Default: {prediction[1]:.2f}")
    



if __name__ == "__main__":
    main()
