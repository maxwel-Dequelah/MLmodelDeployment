from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model
model = pickle.load(open('model2.pkl', "rb"))

@app.route("/", methods=["GET"])
def home():
    return "<h1>Hello world</h1>"

@app.route("/predict/", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()

        # Validate input
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid input. Expecting a dictionary with keys 'account', 'toaccount', and 'amount'."}), 400

        required_keys = {"account", "toaccount", "amount"}
        if not required_keys.issubset(data.keys()):
            return jsonify({"error": f"Input must contain keys: {required_keys}"}), 400

        # Prepare input
        inputs = [data["account"], data["toaccount"], data["amount"]]
        # Update this based on your model type
        concatenated_input = " ".join(inputs)
        processed_input = [concatenated_input]  # Adjust this for numeric models

        # Predict
        prediction = model.predict(processed_input)[0]

        # Return result
        return jsonify({
            "input": inputs,
            "prediction": "Malicious" if prediction == 1 else "Not Malicious"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
