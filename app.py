import tkinter as tk
from tkinter import scrolledtext
import joblib
import os
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.replace(r'[^a-z0-9\\s]', ' ', regex=True).str.replace(r'\\s+', ' ', regex=True).str.strip()


def ensure_model():
    """Load model and tfidf if present; otherwise train and save them."""
    model_path = 'model.pkl'
    tfidf_path = 'tfidf.pkl'

    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf

    # Train fallback
    df = pd.read_csv('spam.csv', encoding='latin-1')
    # drop extra unnamed columns if present
    for c in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    df['cleaned'] = clean_text(df['message'])

    tfidf = TfidfVectorizer(stop_words='english')
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label_num']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    joblib.dump(tfidf, tfidf_path)
    return model, tfidf


print('app.py: starting', flush=True)
try:
    print('app.py: loading or training model...', flush=True)
    model, tfidf = ensure_model()
    print('app.py: model ready', flush=True)
except Exception as e:
    print('app.py: error during model load/train ->', e, flush=True)
    raise


def predict_text(raw_text: str):
    if raw_text.strip() == '':
        return None, None
    cleaned = re.sub(r'[^a-z0-9\\s]', ' ', raw_text.lower())
    cleaned = re.sub(r'\\s+', ' ', cleaned).strip()
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = None
    if hasattr(model, 'predict_proba'):
        prob = model.predict_proba(vec)[0][1]
    return pred, prob


def build_ui():
    # --- Tkinter UI ---
    window = tk.Tk()
    window.title('Spam Classifier')
    window.geometry('600x360')

    lbl = tk.Label(window, text='Enter message to classify:')
    lbl.pack(anchor='w', padx=10, pady=(10, 0))

    text = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=8)
    text.pack(fill='both', expand=False, padx=10, pady=5)


    def on_predict():
        msg = text.get('1.0', tk.END).strip()
        pred, prob = predict_text(msg)
        if pred is None:
            result_label.config(text='Please enter a message', fg='orange')
            prob_label.config(text='')
            return
        if pred == 1:
            result_label.config(text='SPAM 🚨', fg='red')
        else:
            result_label.config(text='HAM ✅', fg='green')
        if prob is not None:
            prob_label.config(text=f'Spam probability: {prob:.3f}')
        else:
            prob_label.config(text='')


    def on_clear():
        text.delete('1.0', tk.END)
        result_label.config(text='')
        prob_label.config(text='')


    btn_frame = tk.Frame(window)
    btn_frame.pack(pady=10)

    predict_btn = tk.Button(btn_frame, text='Predict', command=on_predict, width=12)
    predict_btn.pack(side='left', padx=6)

    clear_btn = tk.Button(btn_frame, text='Clear', command=on_clear, width=12)
    clear_btn.pack(side='left', padx=6)

    nonlocal_placeholders = {}
    # place result and prob labels after defining them so nested functions can close over
    result = tk.Label(window, text='', font=('Arial', 14))
    result.pack(pady=(6, 2))
    prob = tk.Label(window, text='', font=('Arial', 10))
    prob.pack()

    # Expose labels to outer scope names used earlier
    global result_label, prob_label
    result_label = result
    prob_label = prob

    print('app.py: launching GUI', flush=True)
    window.mainloop()


if __name__ == '__main__':
    build_ui()