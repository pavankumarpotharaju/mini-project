from flask import Flask, render_template, request
import pandas as pd
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
model_path = 'kmodel.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])
def submit():
    if request.method == 'POST':
        try:
            # Reading the inputs given by the user
            input_feature = [request.form.get(name) for name in ['year', 'month', 'day', 'hour', 'temp', 'humidity']]
            logging.debug(f"Received input: {input_feature}")  # Log input values
            
            # Ensure all inputs are present
            if not all(input_feature):
                return render_template("inner-page.html", result="Error: Please provide all 6 input values.")
            
            # Convert to float
            input_feature = [float(x) for x in input_feature]
            
            # Define column names
            names = ['Year', 'Month', 'Day', 'Hour', 'temp', 'humidity']
            data = pd.DataFrame([input_feature], columns=names)
            
            # Predictions using the loaded model file
            prediction = model.predict(data)
            rounded_prediction = round(prediction[0])  # Round the prediction to a whole number
            output = f"Predicted Carbon Monoxide Level: {rounded_prediction} ppm."
        
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            output = f"Error in prediction: {str(e)}"
        
        return render_template("last-page.html", result=output)
    return render_template("inner-page.html", result="")

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Create a contact.html and render it here

if __name__ == '__main__':
    app.run(debug=True, port=5000)
