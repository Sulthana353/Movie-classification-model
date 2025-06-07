import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load your custom formatted .txt file
file_path = r"D:\source\archive\Genre Classification Dataset\train_data.txt"

# Use custom separator and set column names
df = pd.read_csv(file_path, sep=':::', engine='python', header=None, names=['id', 'title', 'genre', 'plot'])

# Step 2: Check the structure (optional)
print(df.head())
print("Columns:", df.columns.tolist())

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(df['plot'], df['genre'], test_size=0.2, random_state=42)

# Step 4: TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 7: Test prediction
while True:
    user_input = input("\nEnter a plot summary (or 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        break
    user_tfidf = tfidf.transform([user_input])
    prediction = model.predict(user_tfidf)
    print("Predicted Genre:", prediction[0])
