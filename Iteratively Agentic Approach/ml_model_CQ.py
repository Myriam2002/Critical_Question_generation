import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
import joblib
import os

# ===== Step 1: Load Data =====
df = pd.read_csv("D:\\My_working_area\\Masters\\Semester 2\\NLP804\\Project\\Critical_Question_generation\\data_splits\\df_processed.csv")

# ===== Step 2: Define structured features =====
structured_features = [
    "question_word_count",
    "question_char_count",
    "bm25_similarity",
    "word_overlap",
    "max_similarity"
]

df = df.dropna(subset=structured_features)

# ===== Step 3: Combine text and structured features =====
df["text_input"] = df["intervention"] + " " + df["question"]
X_text = df["text_input"]
X_struct = df[structured_features]
y = df["label_numeric"]

# ===== Step 4: Train/Test Split =====
X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_text, X_struct, y, test_size=0.2, random_state=42
)

# ===== Step 5: TF-IDF Vectorization (with bigrams) =====
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_text_train_vec = vectorizer.fit_transform(X_text_train)
X_text_test_vec = vectorizer.transform(X_text_test)

# ===== Step 6: Standardize Structured Features =====
scaler = StandardScaler()
X_struct_train_scaled = scaler.fit_transform(X_struct_train)
X_struct_test_scaled = scaler.transform(X_struct_test)

# ===== Step 7: Combine TF-IDF + Structured Features =====
X_train_combined = hstack([X_text_train_vec, X_struct_train_scaled])
X_test_combined = hstack([X_text_test_vec, X_struct_test_scaled])

# ===== Step 8: Train Logistic Regression Model =====
model = LogisticRegression(max_iter=1000, multi_class='multinomial', class_weight='balanced')
model.fit(X_train_combined, y_train)

# ===== Step 9: Predict and Evaluate =====
y_pred = model.predict(X_test_combined)
y_proba = model.predict_proba(X_test_combined)
usefulness_scores = y_proba[:, 2]  # probability of being 'Useful'

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Invalid", "Unhelpful", "Useful"]))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# ===== Step 10: Save Model and Tools =====
# Save to the same directory as the script (optional: define a folder like 'models/')
save_dir = "models"
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, "logistic_model_with_features.joblib"))
joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer_with_bigrams.joblib"))
joblib.dump(scaler, os.path.join(save_dir, "structured_feature_scaler.joblib"))

print("\n💾 Model, vectorizer, and scaler saved successfully to ./models/")