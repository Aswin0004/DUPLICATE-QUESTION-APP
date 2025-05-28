import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack

# Load and clean data
train_df = pd.read_csv('train.csv')
train_df.dropna(subset=['question1', 'question2'], inplace=True)

# Optional: Sample subset to avoid memory issues
# train_df = train_df.sample(n=100000, random_state=42)  # Uncomment if needed

# Fit vectorizer on combined questions
all_questions = pd.concat([train_df['question1'], train_df['question2']]).dropna().astype(str).unique()
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
vectorizer.fit(all_questions)

# Feature engineering using sparse matrix
def make_features_sparse(q1, q2):
    q1_vec = vectorizer.transform([str(q1)])
    q2_vec = vectorizer.transform([str(q2)])
    return hstack([q1_vec, q2_vec])

X_sparse = []
y = train_df['is_duplicate'].values

print("[INFO] Creating features...")
for q1, q2 in tqdm(zip(train_df['question1'], train_df['question2']), total=len(train_df)):
    X_sparse.append(make_features_sparse(q1, q2))

# Stack all sparse features
from scipy.sparse import vstack
X = vstack(X_sparse)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("[INFO] Training model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
print(classification_report(y_val, y_pred))

# Save model and vectorizer
pickle.dump(model, open('duplicate_model.pkl', 'wb'))
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))
print("[INFO] Model and vectorizer saved.")
