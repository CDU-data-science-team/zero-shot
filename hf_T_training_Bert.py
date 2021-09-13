# %%
"""
<a href="https://colab.research.google.com/github/CDU-data-science-team/zero-shot/blob/feature-Huggingface_transformer/hf_T_training_Bert.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

# %%
!pip install transformers
!pip install datasets #needed for loading metric

# %%
#? not sure this is needed. all i needed is means of evaluating
#from datasets import load_metric    # to instantiate a metric 
# metric = load_metric('Accuracy')
#ex
# fake_preds = np.random.randint(0, 2, size=(64,))
# fake_labels = np.random.randint(0, 2, size=(64,))
# metric.compute(predictions=fake_preds, references=fake_labels)

# %%
"""
##Import needed library
"""

# %%
import pandas as pd
import io
import requests
import transformers  
print(transformers.__version__) # print the transformer version
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer    #the API for training model in transormer since PyTorch does not provide a training loop
from sklearn.model_selection import train_test_split
from nlp import ClassLabel
import datasets # CAUTION: it took me a while to realize that "datasets" and "from nlp import Dataset" aren't exactly
                # the same. I had all sorts of troubles with renaming and/or removing columns in Dataset objects
                # before switching from nlp's Dataset to datasets. Don't know what the difference is- something to
                # look into in the future.
import numpy as np
from datasets import load_metric

# %%
"""
##Loading and split the dataset
"""

# %%
# Read clean data (rows code XX removed) file from GitHub repo pxtextmining
# https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url
url = "https://raw.githubusercontent.com/CDU-data-science-team/pxtextmining/development/datasets/text_data.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')), encoding='utf-8')

#fill missing value has this causes runtime error while fiting the model 
print(df.isna().sum())
df = df[df['feedback'].notna()]
print(df.isna().sum())

# In an effort to avoid non-intuitive errors from trainer.train() about strings, I have converted our multiclass
# problem into a binary one, where the targets are integers (NOTE: numpy integers won't work). It seems like this fixes
# the problem. Problem is, I'm not sure what causes the problem. Is it that our themes are strings? Is it that they are
# too many? Is it both? Is it something else?
# Therefore, once we have confirmed that trainer runs with binary integer classes, we can try two alternatives:
# 1. Try trainer with the original 'label' target variable and see what happens.
# 2. If 1 fails, convert the themes into integers, from 0 to 9.
df['label'].loc[df['label'] == 'Access'] = 0
df['label'].loc[df['label'] != 0] = 1

# %%
#Split the dataset
seed = 0
candidate_labels = df.label.unique()
num_label = len(candidate_labels)

train_feedback, test_feedback, train_label, test_label = train_test_split(df['feedback'], 
                                                                          df['label'], test_size=0.4, random_state=seed,
                                                                          stratify=df['label']) # Stratified split

# Function to convert our pd.DataFrame into Dataset.
def toDataset(dataset, text_col=None, class_col=None):
    all_classes = dataset[class_col].unique()
    dataset_nlp = datasets.Dataset.from_pandas(df[[text_col, 'label']])
    dataset_nlp.features[class_col] = ClassLabel(num_classes=len(all_classes), names=list(all_classes), names_file=None,
                                                 id=None)
    return dataset_nlp

train_nlp = toDataset(pd.concat([train_feedback, train_label], axis=1), text_col='feedback', class_col='label')
test_nlp = toDataset(pd.concat([test_feedback, test_label], axis=1), text_col='feedback', class_col='label')

# %%
# #seperate the labels from the feedbacks
# sequence = df.feedback
# candidate_labels = df.label.unique()

# # split the sequence
# train = sequence.iloc[:701]
# eval = sequence.iloc[701:1001]
# test = sequence.iloc[1001:1301]

# df.reset_index(inplace=True, drop=True)
# print(train.tail())
# eval.head()
# test.tail()

# %%
print(train_feedback.shape)
test_feedback.shape

# %%
"""
#Preprocessing the data
"""

# %%
model_checkpoint = 'bert-base-cased'
#model_checkpoint = "distilbert-base-uncased"

#instantiate our tokenizer. 
try:
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, num_labels=num_label, use_fast=True) 
except:
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, num_labels=len(candidate_labels))   # remove  use_fast=True if error is thrown from above

# Define function for tokenizing
def tokenize_function(dataset):
    return tokenizer(dataset['feedback'], padding='max_length', truncation=True)

# Tokenize the data
tokenized_train = train_nlp.map(tokenize_function, batched=True)
tokenized_train = tokenized_train.rename_column('label', 'labels') # Rename target so HF understands it.
tokenized_train = tokenized_train.remove_columns(['__index_level_0__']) # Remove index (video: https://huggingface.co/transformers/training.html#preparing-the-datasets)

tokenized_test = test_nlp.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.rename_column('label', 'labels')
tokenized_test = tokenized_test.remove_columns(['__index_level_0__'])

# %%
"""
# Define our model
"""

# %%
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_label)

# %%
"""
#Fine-tuning the model
"""

# %%
"""
### Instantiate a TrainingArguments 
to hold all the hyperparameters we can tune for the Trainer

"""

# %%
metric_name = 'accuracy'

#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics  #!define compute_metrics function
# )
batch_size = 10 # DON'T KNOW WHAT NUMBER TO SET IT TO!

#this uses all the defaults
args = TrainingArguments(
    "train-model",    # <output_dir: str> required to save the checkpoints of the model, all other arguments are optional
    evaluation_strategy = "epoch",  #to fine-tune your model and regularly report the evaluation metrics at the end of each epoch 
    # learning_rate=2e-5,
    per_device_train_batch_size=batch_size,   # batch size per device during training
    per_device_eval_batch_size=batch_size,    # batch size for evaluation
    # num_train_epochs=5,                       # total number of training epochs
    # weight_decay=0.01,
    # load_best_model_at_end=True,
    metric_for_best_model=metric_name, #
    label_names = list(candidate_labels) # The list of keys in our dictionary of inputs that correspond to the labels.
)

# %%
"""
### Define a function for evaluating the model
needs to takes predictions and labels (grouped in a namedtuple called EvalPrediction)
"""

# %%
metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# %%
"""
### Instantiate a trainer
"""

# %%
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics  #!define compute_metrics function
)

# %%
"""
### Train the model 
to do the actual fine tuning of the model
"""

# %%
trainer.train()

# Andreas' fixes end here #
########################################################################################################################
# %%
"""

"""

# %%
trainer.evaluate()


#predictions = trainer.predict(tokenized_test)["logits"]

# %%
"""
#Hyperparameter search

Ignore this section for now
"""

# %%
"""
The hyperparameter_search method returns a BestRun objects, which contains the value of the objective maximized (by default the sum of all metrics) and the hyperparameters it used for that run.
"""

# %%
#Hyperparameter search (The Trainer supports hyperparameter search using optuna or Ray Tune. )
# ! pip install optuna
# ! pip install ray[tune]    #run either

#needed because the Trainer will run several trainings
def model_init():
    return AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
best_run = trainer.hyperparameter_search(n_trials=10, direction="maximize")   #ustomize the search space by passing a hp_space argument
best_run

# %%
#to reproduce the best training, just set the hyperparameters in your TrainingArgument before creating a Trainer:
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

trainer.train()

# %%
#EvalPrediction
from datasets import load_metric
metric = load_metric('glue', actual_task)
metric.compute(predictions=predictions, references=labels)