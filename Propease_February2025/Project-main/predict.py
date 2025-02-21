import sys
import json

def predict(data):
    # Replace this with your model's prediction logic
    return [x * 2 for x in data]  # Example: doubling the input numbers

if __name__ == "__main__":
    # Parse input JSON from the command line
    input_json = sys.argv[1]
    input_data = json.loads(input_json)

    # Perform prediction
    prediction = predict(input_data)

    # Output the prediction as JSON
    output = {"prediction": prediction}
    print(json.dumps(output))
