from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback

app = Flask(__name__)

# Load the saved model
model = joblib.load("best_landslide_model.pkl")
# print("Model loaded", model)

@app.route('/')
def home():
    return "Landslide Prediction API"

# Define the endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input JSON data
        data = request.get_json()

        # Convert JSON to DataFrame
        input_data = pd.DataFrame([data])

        # Ensure input columns match model training features
        required_features = model.feature_names_in_
        input_data = input_data[required_features]  # Select only required columns

        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data).tolist()

        # Return response
        response = {
            "prediction": int(prediction[0]),
            "probability": prediction_proba
        }
        print(response)
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
