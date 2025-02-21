from flask import Flask, request, jsonify
import subprocess
import json

app = Flask(__name__)

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate the incoming JSON data
        input_data = request.get_json()
        if not input_data:
            return jsonify({"status": "error", "message": "Invalid input. JSON data is required."}), 400

        # Convert input data to JSON string
        input_json = json.dumps(input_data)

        # Call the Python script with input data
        command = f"python3 predict.py '{input_json}'"
        process = subprocess.run(command, shell=True, capture_output=True, text=True)

        # Check for errors in the script execution
        if process.returncode != 0:
            return jsonify({"status": "error", "message": f"Error executing script: {process.stderr.strip()}"}), 500

        # Decode the output JSON from the Python script
        result = json.loads(process.stdout.strip())
        if 'prediction' in result:
            return jsonify({"status": "success", "prediction": result['prediction']})
        else:
            return jsonify({"status": "error", "message": "Prediction key not found in script output."}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
