import pandas as pd
import io
import requests
from transformers import pipeline

# Read clean data (rows code XX removed) file from GitHub repo pxtextmining
# https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url
url = "https://raw.githubusercontent.com/CDU-data-science-team/pxtextmining/development/datasets/text_data.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')), encoding='utf-8')

print(df.isna().sum())
df = df[df['feedback'].notna()] # Remove records with no feedback text
print(df.isna().sum())

# Zero-shot pipeline
# https://colab.research.google.com/drive/1jocViLorbwWIkTXKwxCOV9HLTaDDgCaw?usp=sharing
classifier = pipeline("zero-shot-classification")
sequences = df.feedback.tolist()
candidate_labels = df.label.unique()
candidate_labels.sort()

zs = classifier(sequences[0:3], candidate_labels, multi_label=True) # A list of dicts with "sequence", "labels" and "scores".

scores = []
class_pred = []
for i in range(0, len(zs)):
    scores.append(pd.DataFrame([zs[i]['scores']], columns=zs[i]['labels']))
    max_score = max(zs[i]['scores'])
    max_index = zs[i]['scores'].index(max_score)
    class_pred.append(zs[i]['labels'][max_index])
scores = pd.concat(scores)
scores = scores.reindex(sorted(scores.columns), axis=1)
scores['class_pred'] = class_pred
print(scores)
scores.to_csv('zs_preds.csv')