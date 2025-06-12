from flask import Flask, request, jsonify
from logic.predictor import predict_complexity

app = Flask(__name__)

@app.route("/")
def home():
    return "🚀 API de Red Neuronal para Complejidad Algorítmica"

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json()
    if not data or "codigo" not in data:
        return jsonify({"error": "Falta el campo 'codigo'"}), 400

    resultado = predict_complexity(data["codigo"])

    return jsonify({
        "O": resultado["O"],
        "Ω": resultado["Ω"],
        "Θ": resultado["Θ"]
    })

if __name__ == "__main__":
    app.run(debug=True)

