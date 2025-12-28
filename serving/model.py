import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load SavedModel using TFSMLayer (Keras 3 compatible)
model = tf.keras.layers.TFSMLayer(
    "/model",
    call_endpoint="serve"
)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["instances"]
    x = np.array(data).reshape(-1, 784)
    preds = model(x).numpy().tolist()
    return jsonify({"predictions": preds})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)