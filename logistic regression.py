import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

file_path = "train_data.txt"

df = pd.read_csv(file_path, sep=':::', engine='python', header=None, names=['id', 'title', 'genre', 'plot'])

print(df.head())
print("Columns:", df.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(df['plot'], df['genre'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

while True:
    user_input = input("\nEnter a plot summary (or 'exit' to quit):\n> ")
    if user_input.lower() == 'exit':
        break
    user_tfidf = tfidf.transform([user_input])
    prediction = model.predict(user_tfidf)
    print("Predicted Genre:", prediction[0])
