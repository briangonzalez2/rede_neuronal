from flask import Flask, request, jsonify
from logica.predictor import predict_complexity

app = Flask(__name__)

@app.route("/predict-complexity", methods=["POST"])
def predict():
    code = request.json.get("code", "")
    if not code:
        return jsonify({"error": "No code provided"}), 400
    prediction = predict_complexity(code)
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
