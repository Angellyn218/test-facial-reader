import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# Load CSV of predictions
df = pd.read_csv("emotion_predictions.csv")

# Get predicted emotion by max probability column
emotion_cols = [col for col in df.columns if col.startswith("pred_")]
df["pred_label"] = df[emotion_cols].idxmax(axis=1).str.replace("pred_", "")

# Define positive and negative emotion groups
negative = ['contempt', 'anger', 'fear', 'disgust', 'sad']
positive = ['surprised', 'happy']

# Compute classification metrics for all emotions
true = df["true_label"]
pred = df["pred_label"]

print("\n📊 Classification Report (All Emotions):\n")
print(classification_report(true, pred, zero_division=0))

acc = accuracy_score(true, pred)
print(f"\n✅ Accuracy: {round(acc * 100, 2)}%")

# ===============================
# 🎯 Binary Evaluation: Positive vs Negative
# ===============================
def label_group(label):
    if label in positive:
        return "positive"
    elif label in negative:
        return "negative"
    else:
        return "neutral"  # optional fallback

df["true_group"] = df["true_label"].apply(label_group)
df["pred_group"] = df["pred_label"].apply(label_group)

# Filter out neutral if present
df_filtered = df[(df["true_group"] != "neutral") & (df["pred_group"] != "neutral")]

print("\n🌓 Positive vs Negative Classification Report:\n")
print(classification_report(df_filtered["true_group"], df_filtered["pred_group"], zero_division=0))

group_acc = accuracy_score(df_filtered["true_group"], df_filtered["pred_group"])
print(f"\n✅ Positive/Negative Accuracy: {round(group_acc * 100, 2)}%")


"""
📈 Per-Emotion Metrics:

     Emotion  Precision    Recall  F1 Score
0  surprised   0.000000  0.000000  0.000000
1   contempt   0.000000  0.000000  0.000000
2      anger   0.000000  0.000000  0.000000
3       fear   0.200000  0.066667  0.100000
4    disgust   1.000000  0.062147  0.117021
5      happy   0.873303  0.932367  0.901869
6        sad   0.372093  0.571429  0.450704
"""

"""
            precision   recall   f1-score   
negative       0.92      0.85      0.88
positive       0.88      0.93      0.90
"""