from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the request
        input_data = request.json.get('input_data')
        if not input_data:
            return jsonify({"status": "error", "message": "No input data provided"}), 400

        # Convert input data to JSON string to pass to the Python script
        input_json = json.dumps(input_data)

        # Call the Python script and pass the input JSON
        command = f"python3 predict.py '{input_json}'"
        result = subprocess.check_output(command, shell=True, text=True)

        # Decode the output JSON from the Python script
        prediction_result = json.loads(result)

        # Check if the prediction exists in the result
        if 'prediction' in prediction_result:
            return jsonify({"status": "success", "prediction": prediction_result['prediction']})
        else:
            return jsonify({"status": "error", "message": "Prediction not found in the response"}), 500

    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": "Failed to execute prediction script", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": "An error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
