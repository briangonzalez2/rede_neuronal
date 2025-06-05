from flask import Flask, request, render_template
from logic.predictor import predict_complexity

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    code = ""
    if request.method == "POST":
        code = request.form.get("code", "")
        if code.strip():
            result = predict_complexity(code)
    return render_template("index.html", result=result, code=code)

if __name__ == "__main__":
    app.run(debug=True)
