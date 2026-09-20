from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
try:
    loaded_model = joblib.load('logistic_regression_model.pkl')
    loaded_scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    loaded_model = None
    loaded_scaler = None

@app.route('/predict', methods=['POST'])
def predict():
    if loaded_model is None or loaded_scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    try:
        # Get data from POST request
        data = request.get_json(force=True)

        # Convert json data to pandas DataFrame
        # Ensure the order of features is consistent with training data
        feature_names = X.columns.tolist() # X is from the training phase
        input_df = pd.DataFrame(data, index=[0])
        input_df = input_df[feature_names] # Ensure correct column order

        # Scale the input features
        scaled_input = loaded_scaler.transform(input_df)

        # Make prediction
        prediction = loaded_model.predict(scaled_input)
        prediction_proba = loaded_model.predict_proba(scaled_input)

        # Return prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'probability_class_0': float(prediction_proba[0][0]),
            'probability_class_1': float(prediction_proba[0][1])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# To run the Flask app, you'd typically use:
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

print("Flask API setup complete. To run, execute the commented out app.run() line in a local environment or deploy.")
