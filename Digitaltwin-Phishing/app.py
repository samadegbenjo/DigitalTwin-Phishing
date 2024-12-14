from flask import Flask, request, jsonify
import onnxruntime as rt
import numpy as np
import joblib

app = Flask(__name__)

# Load models
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
sess = rt.InferenceSession("ridge_model.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    data = np.array(data).reshape(1, -1)
    scaled_data = scaler.transform(data)
    reduced_data = pca.transform(scaled_data)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: reduced_data.astype(np.float32)})[0]
    return jsonify({'phishing': int(pred_onx[0])})

if __name__ == '__main__':
    app.run(debug=True)