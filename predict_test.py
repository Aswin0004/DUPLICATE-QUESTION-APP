import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

# Load model and vectorizer
with open('duplicate_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def make_features(q1, q2):
    q1_vec = vectorizer.transform([q1])
    q2_vec = vectorizer.transform([q2])
    return np.hstack([q1_vec.toarray(), q2_vec.toarray()])

chunk_size = 10000  # Adjust based on your memory availability
results = []

# Open the output file and write headers
output_file = 'test_predictions.csv'
with open(output_file, 'w') as f_out:
    f_out.write('test_id,is_duplicate_pred\n')

# Read test.csv in chunks
for chunk in tqdm(pd.read_csv('test.csv', chunksize=chunk_size)):
    # Fill NaNs if any to avoid errors
    chunk['question1'] = chunk['question1'].fillna('')
    chunk['question2'] = chunk['question2'].fillna('')

    # Prepare features for this chunk
    X_chunk = np.array([make_features(q1, q2)[0] for q1, q2 in zip(chunk['question1'], chunk['question2'])])

    # Predict on chunk
    preds = model.predict(X_chunk)

    # Append results to output csv
    chunk_results = pd.DataFrame({
        'test_id': chunk['test_id'],
        'is_duplicate_pred': preds
    })

    chunk_results.to_csv(output_file, mode='a', header=False, index=False)

print(f"Predictions saved to {output_file}")
