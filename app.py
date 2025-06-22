import pickle
import pandas as pd 
from flask import Flask, request, jsonify

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Love Score Prediction API is Live"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        df_input = pd.DataFrame([data])
        prediction = model.predict(df_input)[0]
        return jsonify({"predicted_love_score": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)