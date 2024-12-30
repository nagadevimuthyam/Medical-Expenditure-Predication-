from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load the models and label encoders
with open('disease_model.pkl', 'rb') as file:
    disease_model = pickle.load(file)

with open('cost_model.pkl', 'rb') as file:
    cost_model = pickle.load(file)

with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form data
        features = [
            int(request.form['age']),
            int(request.form['Gender']),
            int(request.form['Fever']),
            int(request.form['Cough']),
            int(request.form['Fatigue']),
            int(request.form['Difficulty Breathing']),
            int(request.form['Blood Pressure']),
            int(request.form['Cholesterol Level'])
        ]

        final = np.array(features).reshape((1, -1))

        # Predict disease
        disease_pred = disease_model.predict(final)[0]

        # Predict cost
        cost_pred = cost_model.predict(final)[0]

        # Convert numerical prediction to string
        if 'Disease' in label_encoders:
            disease_pred_str = list(label_encoders['Disease'].inverse_transform([disease_pred]))[0]
        else:
            disease_pred_str = 'Unknown'

        # Handle negative cost predictions
        if cost_pred < 0:
            return render_template('op.html', pred_disease=f'Expected disease: {disease_pred_str}', pred_amount='Error calculating Amount!')
        else:
            return render_template('op.html', pred_disease=f'Expected disease: {disease_pred_str}', pred_amount=f'Expected amount: {cost_pred:.2f}')
    except Exception as e:
        return render_template('op.html', pred_disease='Error calculating Disease!', pred_amount='Error calculating Amount!')

if __name__ == '__main__':
    app.run(debug=True)
