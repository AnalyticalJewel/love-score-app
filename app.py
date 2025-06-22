import pickle
import pandas as pd 
from flask import Flask, request, jsonify,render_template

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Serve the frontend

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    df_input = pd.DataFrame([data])
    prediction = model.predict(df_input)[0]
    return jsonify({"predicted_love_score": round(prediction, 2)})
