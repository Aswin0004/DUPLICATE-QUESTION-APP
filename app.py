from flask import Flask, request, render_template, redirect, url_for, send_file
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and vectorizer
with open('duplicate_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def make_features(q1, q2):
    q1_vec = vectorizer.transform([q1])
    q2_vec = vectorizer.transform([q2])
    return np.hstack([q1_vec.toarray(), q2_vec.toarray()])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        q1 = request.form['question1']
        q2 = request.form['question2']
        features = make_features(q1, q2)
        prediction = model.predict(features)[0]
        result = 'Duplicate' if prediction == 1 else 'Not Duplicate'
    return render_template('index.html', result=result)

@app.route('/bulk', methods=['GET', 'POST'])
def bulk():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)

            test_df = pd.read_csv(filepath)
            test_df = test_df.dropna(subset=['question1', 'question2'])

            X_test = np.array([make_features(q1, q2)[0] for q1, q2 in zip(test_df['question1'], test_df['question2'])])
            predictions = model.predict(X_test)

            test_df['is_duplicate_pred'] = predictions
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_output.csv')
            test_df[['test_id', 'is_duplicate_pred']].to_csv(output_path, index=False)

            return send_file(output_path, as_attachment=True)

    return render_template('bulk.html')



if __name__ == '__main__':
    app.run(debug=True)
