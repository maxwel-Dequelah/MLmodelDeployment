from flask import Flask, request, jsonify
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the model
@app.route("/", methods=["GET"])
def home():
    return "<h1>Hello world</h1>"

model=pickle.load(open('model2.pkl',"rb"))

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

        # Extract and combine input values
        inputs = [data["account"], data["toaccount"], data["amount"]]
        concatenated_input = " ".join(inputs)

        # Preprocess input
        # Assuming the model handles tokenization and preprocessing internally
        prediction = model.predict([concatenated_input])[0]

        # Return result
        return jsonify({
            "input": inputs,
            "prediction": "Malicious" if prediction == 1 else "Not Malicious"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
