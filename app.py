from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load the trained model
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json
        
        # Debug: Log incoming data
        print("Received data:", data)
        
        # Convert data to DataFrame
        input_data = pd.DataFrame([data])
        
        # Define columns in the correct order
        columns = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 
            'Oldpeak', 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina'
        ]
        
        # Reorder and fill missing columns with default values
        input_data = input_data.reindex(columns=columns, fill_value=0)
        
        # Debug: Log processed input data
        print("Processed input data:\n", input_data)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        print(prediction)
        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        # Log the error and return a 500 error response
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
