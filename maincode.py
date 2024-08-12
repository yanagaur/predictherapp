from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Load the model from the .sav file
file_path = 'symptoms.sav'
model = joblib.load(file_path)

app = Flask(__name__)

# Allow CORS for both Vercel and ngrok during development
CORS(app, resources={r"/*": {"origins": ["https://pcosapp.vercel.app", "https://e8de-14-97-132-203.ngrok-free.app"]}})

# Define the BMI insights function
def BMI_Insights(x):
    if x < 18:
        return 'A'
    elif x >= 18 and x < 26:
        return 'B'
    else:
        return 'C'

@app.route('/noreport', methods=['POST'])
def noreport():
    try:
        # Log the content type
        content_type = request.content_type
        print("Content-Type:", content_type)

        # Ensure the content type is application/json
        if 'application/json' not in content_type:
            return jsonify({'error': '415 Unsupported Media Type: Content-Type must be application/json'}), 415

        # Get the data from the POST request
        data = request.json

        # Log the received data
        print("Received data:", data)

        # Validate and extract input data
        try:
            features = [
                int(data['age']),
                int(data['Cycle(R/I)'].lower() == 'regular'),
                int(data['cycleLength']),
                float(data['waist']),
                int(data['weightGain'] == 'Y'),
                int(data['hairGrowth'] == 'Y'),
                int(data['skinDarkening'] == 'Y'),
                int(data['hairLoss'] == 'Y'),
                int(data['pimples'] == 'Y'),
                float(data['fastFood'] == 'Y'),
                int(data['regExercise'] == 'Y')
            ]

            # Calculate BMI category
            bmi_value = float(data['bmi'])
            bmi_category = BMI_Insights(bmi_value)

            # Initialize BMI category variables
            bmi_a = 1.0 if bmi_category == 'A' else 0.0
            bmi_b = 1.0 if bmi_category == 'B' else 0.0
            bmi_c = 1.0 if bmi_category == 'C' else 0.0

            # Append BMI category variables to the features list
            features.extend([bmi_a, bmi_b, bmi_c])

        except (KeyError, ValueError) as e:
            return jsonify({'error': f'Invalid input data: {str(e)}'}), 400

        # Convert to a 2D numpy array
        features = np.array(features, dtype=float).reshape(1, -1)

        # Use the model to make predictions
        predictions = model.predict(features)

        # Return the predictions and other results as a JSON response
        return jsonify({
            'predictions': predictions.tolist(),
            'bmi_a': bmi_a,
            'bmi_b': bmi_b,
            'bmi_c': bmi_c
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
