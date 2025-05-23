from flask import Flask, request, render_template
import pickle
import numpy as np


app = Flask(__name__)

model = pickle.load(open("regression_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours_studied = float(request.form['hours_studied'])
        previous_scores = float(request.form['previous_scores'])
        extracurricular = request.form['extracurricular']
        sleep_hours = float(request.form['sleep_hours'])
        sample_question_papers = float(request.form['sample_question_papers'])

        # Encode categorical feature
        activity_encoded = encoder.transform([extracurricular])[0]

        # Prepare features and scale
        features = np.array([[hours_studied, previous_scores, activity_encoded, sleep_hours, sample_question_papers]])
        scaled_features = scaler.transform(features)

        prediction = model.predict(scaled_features)[0]

        return render_template('index.html', prediction_text=f'Predicted Performance Index: {prediction:.2f}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
