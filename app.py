import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
data = pd.read_csv(r"C:\Users\sheth\Desktop\python\ml\housing.csv")

# Define features and target
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'guestroom', 'airconditioning', 'prefarea', 'mainroad', 'furnishingstatus']
X = data[features].copy()
y = data['price']

# Encode categorical variables
label_enc = LabelEncoder()
for col in ['guestroom', 'airconditioning', 'prefarea', 'mainroad', 'furnishingstatus']:
    X[col] = label_enc.fit_transform(X[col])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        guestroom = 1 if request.form['guestroom'] == 'yes' else 0
        mainroad = 1 if request.form['mainroad'] == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
        prefarea = 1 if request.form['prefarea'] == 'yes' else 0
        furnishingstatus = request.form['furnishingstatus']

        # Encode furnishing status
        furnishing_encoded = label_enc.fit_transform(['semi-furnished', 'furnished', 'unfurnished'])
        furnishing_dict = dict(zip(['semi-furnished', 'furnished', 'unfurnished'], furnishing_encoded))
        furnishing_status_encoded = furnishing_dict[furnishingstatus]

        # Prepare input data
        user_data = np.array([[area, bedrooms, bathrooms, stories, guestroom, airconditioning, prefarea, mainroad, furnishing_status_encoded]])
        
        # Predict house price
        predicted_price = model.predict(user_data)[0]

        return render_template('index.html', prediction=f"🏠 Predicted House Price: ₹{predicted_price:.2f}")

    except Exception as e:
        return render_template('index.html', error=f"❌ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
