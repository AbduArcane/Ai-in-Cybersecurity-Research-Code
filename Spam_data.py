# Spam_data.py
# Author and Editor: Abdullah Bin Asif
# Student ID: 2023882
# For: Ai in Cybersecurity (Rule Based Ai Research Paper)
# Topic: Spam Detection
# Note: Assistance provided by ChatGPT (Code Structure, rules & evaulation)


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd


# Load the Dataset with the following code

df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
# Code converts all label to numerics
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# This converts all messages to lowercase
df['message'] = df['message'].str.lower()

print("\nData prepared successfully.")
print(df.head())  # Prints the dataset

# ----- RULE BASED SYSTEM 1 (Keyword-Based) -----

# Common spam keywords that are found in messages to be used in the system
keywords = ["win", "prize", "free", "click", "cash",
            "offer", "urgent", "claim", "limited time", "reward"]


# Code below check for spam messages or excessive exclamation marks (!)
def keyword_rule(msg):
    if any(word in msg for word in keywords):
        return 1  # 1 means "spam"
    if msg.count('!') >= 3:
        return 1  # This also returns as "spam"
    else:
        return 0  # if both conditions are not met, it is met with "ham"


# This code applies the rules to every message in the dataset
df['pred_keyword'] = df['message'].apply(keyword_rule)

print("\nSample predictions (1 = spam & 0 = ham):")
print(df[['message', 'pred_keyword']].head(10))

#  ---- Evaulate The System ----

print("\n--- Evaluation: Keyword Rule System ---")
print(classification_report(df['label'], df['pred_keyword'], digits=3))
print("Confusion Matrix:")
print(confusion_matrix(df['label'], df['pred_keyword']))
print("Accuracy:", round(accuracy_score(df['label'], df['pred_keyword'])))


# ----- RULE BASED SYSTEM 2 (Pattern-Based) -----

# This system focuses on the message structure or patterns


def pattern_rules(msg):
    # This code checks if the message starts with common spam terms
    if msg.startswith(("free", "congratulations", "dear user", "attention")):
        return 1  # Returns as "Spam"

    digits = sum(c.isdigit() for c in msg)
    # Counts how many digits appear in a message
    if digits >= 5:
        return 1
    # This code check for "call" or "text" in messages aas they often appear in spam
    if len(msg) < 25 and ("call" in msg or "text" in msg):
        return 1  # Returns as "Spam"

    return 0  # Returns not "Spam"


# Applies these rules to every other message
df['pred_pattern'] = df['message'].apply(pattern_rules)

#  ---- Evaulate The System ----

print("\n--- Evaluation: Pattern Rule System ---")
print(classification_report(df['label'], df['pred_pattern'], digits=3))
print("Confusion Matrix:")
print(confusion_matrix(df['label'], df['pred_keyword']))
print("Accuracy:", round(accuracy_score(df['label'], df['pred_keyword'])))
