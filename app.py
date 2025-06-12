from flask import Flask, render_template, request
from logic.predictor import predict_complexity

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        user_code = request.form["code"]
        prediction = predict_complexity(user_code)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
