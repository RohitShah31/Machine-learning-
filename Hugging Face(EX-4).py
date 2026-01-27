from transformers import pipeline

# Load Hugging Face model
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Test sentences
test_sentences = [
    "Study hour is High and attendance is High",
    "Study hour is Low and attendance is Low",
    "Study hour is High and attendance is Low",
    "Study hour is Low and attendance is High"
]

# Perform inference
results = classifier(test_sentences)

for s, r in zip(test_sentences, results):
    print("Sentence:", s)
    print("Predicted Label:", r["label"])
    print("Confidence Score:", round(r["score"], 4))
    print()


"""
OUTPUT:

Sentence: Study hour is High and attendance is High
Predicted Label: POSITIVE
Confidence Score: 0.9996

Sentence: Study hour is Low and attendance is Low
Predicted Label: NEGATIVE
Confidence Score: 0.9994

Sentence: Study hour is High and attendance is Low
Predicted Label: POSITIVE
Confidence Score: 0.9979

Sentence: Study hour is Low and attendance is High
Predicted Label: POSITIVE
Confidence Score: 0.9987
"""
