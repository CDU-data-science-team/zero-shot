# -*- coding: utf-8 -*-
"""Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1LFh2bP4G-acCaea5mj0tgYOoJfTYkRik
"""

!pip install transformers

"""##Import needed library"""

import transformers
import torch
print(transformers.__version__) #print the tranformer version
from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer   #the API for training model in transormer since PyTorch does not provide a training loop
# from time import time

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder

# # Needed to mount google drive on Colab platform
# from google.colab import drive
# drive.mount('/content/drive')

# # To used code to  upload files on Colab platform
# # from google.colab import files
# # files.upload()

"""##Loading and split the dataset"""

url = 'https://raw.githubusercontent.com/CDU-data-science-team/pxtextmining/main/datasets/text_data.csv'
data = pd.read_csv(url, usecols=['feedback', 'label'],  encoding='utf-8')#, nrows=100)
print(f'{data.head()} \n\nno. of feedbacks: {len(data)} \n')
df = data.loc[:8999,:]         # the remaining data is USED FOR TESTING final prediction

#  take a sample to easily test pipeline
# df = df.sample(1000).reset_index(drop=True)

#fill missing value has this causes runtime error while fiting the model 
print(f'Missing values\n{df.isna().sum()}')
df['feedback'].fillna('Nothing', inplace = True)
print(df.isna().sum())
print(f'{df.head()} \n\nno. of feedbacks: {len(df)} \n')

"""# Needed Function and Classes"""
# Turn the labels and encodings into a Dataset object (using pytorch). 
class feedbkDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

# This will allow us to feed batches of sequences into the model at the same time)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])      # encode the label before doing this
        return item

    def __len__(self):
        return len(self.labels)

### Define a function for evaluating the model
#'weighted' produced higher metrics than 'macro'
def compute_metrics(pred):
    r"""The function that will be used to compute metrics at evaluation. Must take a
    `transformers.EvalPrediction` and return a dictionary string to metric values
    Arguments:
    pred (:obj:`transformers.EvalPrediction`):
        model prediction from `transformers.EvalPrediction`
    Returns:
    :obj:`dict`: 
        dictionary string to metric values
    """ 
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')  # set average to 'weighted' (to take label imbalance into account.) | 'macro' (to not take label imbalance into account)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

"""#Preprocessing the data"""
# 1. Encode the label
# 2. tokenize the text feature
# 3. Combine the label and text together and convert them into a Dataset object
#  (`torch.utils.data.Dataset` or `torch.utils.data.IterableDataset` object)

# # Needed to map label2id and id2label in the model configuration (Very useful while using model in Zeroshot or test classification pipeline)
label2id = {}
id2label = {}
for v, k in enumerate(sorted( df.label.unique())):
  label2id[k] = v
  id2label[v] = k

"""### Define the model"""
# Define the model_checkpoint. Tokenizer and the model
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, num_labels=len(set(df.label)))
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, id2label=id2label, label2id = label2id, num_labels=len(set(df.label)))

# Encode the label and create label2id and id2label - to map label2id and id2label in the model configuration (possible to be able to use model for Zeroshot)
df['label'] = LabelEncoder().fit_transform(df.label)

# Split the dataset
seed = 0
train_texts, train_labels = list(df.feedback), list(df.label)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2, random_state=seed) # create validation set
# train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=.2, random_state=seed)   # create validation set 

print(f'No of feedbacks in\nTrain set: {len(train_texts)}\nValidation set: {len(val_texts)}') #\nTest set: {len(test_texts)} 

# (truncation=True, padding=True will ensure that all of our sequences are padded to the same length and 
# are truncated to be no longer than model’s maximum input length (512 in this case). 
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
# test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = feedbkDataset(train_encodings, train_labels)
val_dataset = feedbkDataset(val_encodings, val_labels)
# test_dataset = feedbkDataset(test_encodings, test_labels)

# The steps above prepared the datasets in the way that the trainer expected. Now all we need to do is create a model to fine-tune, define the TrainingArguments and instantiate a Trainer.

"""#Fine-tuning the model """
### Instantiate a TrainingArguments to hold all the hyperparameters we can tune for the Trainer
training_args = TrainingArguments(
    output_dir='/output',             # output directory
    num_train_epochs=2,               # total number of training epochs (2 appeared to be the best from trial)
    per_device_train_batch_size=16,   # batch size per device during training
    per_device_eval_batch_size=64,    # batch size for evaluation
    learning_rate=3e-5,               # (3e-5 appeared to be the best from trial)
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,                # strength of weight decay
    evaluation_strategy = 'epoch',
    eval_steps = 10,                  # useful if evaluation_strategy="steps"`
    do_eval = True,                   # to run eval on the dev set.
    # do_train = True,                  # to train the model.
    # label_names = list(label2id.keys()), # The list of keys in our dictionary of inputs that correspond to the labels.
    # logging_steps=10,
    # logging_dir='./logs',             # directory for storing logs
)

print(f'No of training Samples: {len(train_dataset)}')
training_args.device          # return the device used by this process
# training_args.n_gpu         # the number of GPUs used by this process

"""### Instantiate a trainer"""
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics,     # used to compute metrics at evaluation
    tokenizer=tokenizer,                 # to preprocess the dataset and make it easier to rerun an uninterrupted training or reuse the fine-tuned model.
)

"""### Train the model """
trainer.train()   # will use GPU as long as you have access to a GPU (i.e to run script with gpu, no addition code is needed)

"""### Run model evaluation (on the test data - if available)
No need for this if no test data because `.evaluate()` uses self.eval_dataset by default and if compute metrics has been assigned to `Trainer` object the evaluation metrics on eval_data set will be returned anyways with train()
"""
# trainer.evaluate(test_dataset)

"""# Save the FineTuned Model
Save a model and its configuration file to a directory, so that it can be re-loaded using the
        ``:func:~transformers.PreTrainedModel.from_pretrained`` class method.
"""
save_path = '/content/drive/MyDrive/Colab Notebooks/best_model/'
trainer.save_model(save_path) # Will save the model (and tokenizer if specified when initializing the trainer object), so you can reload it using from_pretrained()

# Use below codes to save the model and tokenizer seperately
# model.save_pretrained(save_path)   # same as above but only save the model
# tokenizer.save_pretrained(save_path) # same as above but save only the tokenizer