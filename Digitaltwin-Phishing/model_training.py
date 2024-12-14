import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load reduced dataset
df = pd.read_csv('reduced_dataset.csv')

# Split data
X = df.drop('phishing', axis=1)
y = df['phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Ridge Regression model
model = Ridge()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'ridge_model.pkl')

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("ridge_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())