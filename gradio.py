
import gradio as gr
import pandas as pd
import joblib
import logging


model = joblib.load('model.pkl')

def run_prediction(bank_balance, annual_salary, employed):

    print("Received inputs:", bank_balance, annual_salary, employed)
    # Convert inputs to a DataFrame
    input_data = pd.DataFrame({
        'Employed': [1 if employed == "Yes" else 0],
        'Bank Balance': [float(bank_balance)],
        'Annual Salary': [float(annual_salary)]
    })
    
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return "Default" if prediction[0] == 1 else "No Default"




demo = gr.Interface(
    fn=run_prediction,
    inputs=["text", "text", gr.Dropdown(choices=["Yes", "No"])],
    outputs=["text"],
)

demo.launch(share=True)

