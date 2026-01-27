from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset converted to text
sentences = [
    "Study hour is Low and attendance is High",
    "Study hour is Low and attendance is Low",
    "Study hour is Low and attendance is High",
    "Study hour is High and attendance is Low",
    "Study hour is High and attendance is High",
    "Study hour is High and attendance is High",
    "Study hour is High and attendance is Low",
    "Study hour is High and attendance is Low",
    "Study hour is High and attendance is High",
    "Study hour is Low and attendance is Low"
]

# Labels: Yes = 1, No = 0
labels = [0,0,0,1,1,1,1,1,1,0]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    sentences, labels, test_size=0.3, random_state=42
)

# Bag of Words
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Logistic Regression model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction
predictions = model.predict(X_test_vec)

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Display predictions
for s, p in zip(X_test, predictions):
    print(s, "->", "Yes" if p == 1 else "No")


"""
OUTPUT:

Accuracy: 1.0

Study hour is High and attendance is Low -> Yes
Study hour is Low and attendance is Low -> No
Study hour is High and attendance is Low -> Yes
"""
+