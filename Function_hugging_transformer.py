# """#install needed packages"""
# !pip install transformers

"""#Load important libraries"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from transformers import pipeline

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from warnings import filterwarnings
filterwarnings("once", category=DeprecationWarning) # Display just the first matching deprecation warnings (e.g. warning that multi_class is changing to multi_label)

# Needed Functions
def class_accuracy (cat, y_true, y_pred):
  """
  Function for individual Class Accuracy score
  :param string cat: name of class.
  :param array y_true: True classes, shape = [n_samples].
  :param array y_pred: Predicted classes, shape = [n_samples].
  :return: cba (`float`): The individual Class Accuracy score (returned as percentage)
  """
  total = list(y_true).count(cat)
  y_true_ = list(y_true)
  y_pred_ = list(y_pred)
  correct = [i for i, x in enumerate(y_true_) if x == str(cat) and y_true_[i] == y_pred_[i]]
  if total == 0:  
    acc = None
  else:
    acc = len(correct)/total
    acc = round((acc*100),2)
  return acc


def plot_conf_mat (y_true, y_pred, cmap="YlGnBu", ax=None, cm_perc=True):
    """
    Function for ploting confusion matrix
    :param array y_true: True classes, shape = [n_samples].
    :param array y_pred: Predicted classes, shape = [n_samples]. 
    :param string cmap: color palette.
    :param  Axes object ax: subplot axis.
    :param bool cm_perc: if true the percentage accuracy is plot.
    :return: (`matplotlib plot object`): the confusion matrix plot
    """
    mat = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    mat_sum = np.sum(mat, axis=1, keepdims=True)      #sum across rows (actual label) and retain the matrix dimesion in the return array
    mat_perc = mat / mat_sum.astype(float) * 100

    if cm_perc:
        mat_df = pd.DataFrame(mat_perc, index=np.unique(y_true), columns=np.unique(y_true))
    else:
        mat_df = pd.DataFrame(mat, index=np.unique(y_true), columns=np.unique(y_true))
    mat_df.index.name = 'Actual'
    mat_df.columns.name = 'Predicted'

    sns.heatmap(mat_df,
                cmap= cmap,                             # color palette
                annot=True,                             # to display the values 
                fmt='.1f',                              # round up the values
                linewidths=.5,                          # Add lines between the cells
                annot_kws={"size": 15},                 # font size
                ax=ax,
                cbar_kws={"shrink": .70,                # adjust the size of the sidebar 
#                              "orientation": "horizontal"  # change the sidebar position
                }, 
    )

# List of models with predictions above random chance (best performing models)
        # best peforming model
#         [     # 'valhalla/distilbart-mnli-12-1',
#               # 'valhalla/distilbart-mnli-12-6', 
#               # 'facebook/bart-large-mnli',
#               # 'cross-encoder/nli-deberta-base'    # Fastest model
#                 ]
    
def main (filepath, feedback_col, target_col, model='valhalla/distilbart-mnli-12-9', no_row=None, gpu=False):
    """#Load Data"""
    # Choose to read CSV from folder or the table directly
    try: 
        if isinstance(filepath, str):                         # if the filepath is a string read the data as csv file
            df = pd.read_csv(filepath, encoding='utf-8')
        else:
            df = filepath                                   # accept the data as a dataframe
    except:
      raise TypeError('filepath can only be a a pandas dataframe or .csv filepath (Ensure you specify the filepath correctly)!') 

    df = df.rename(columns={feedback_col: 'feedback', target_col: 'label'})

    """# Prepare data for prediction"""   
    df.dropna(inplace=True)                               # drop rows with missing values
    no_row = len(df) if no_row == None else no_row        # if number of row is not specified, use all the data
    sequence = df.feedback[:no_row].values                # first n rows of feedbacks to be classify
    candidate_labels = df.label.unique()

    """#Prediction"""
    class_dict ={}
    feedback_predict = []; feedback_predict_score = []    # list to hold the predicted class and its score

    # instantiate a zeroshot classifier object    
    classifier = pipeline("zero-shot-classification", model, device=0) if gpu else pipeline("zero-shot-classification", model) # utilize GPU/CPU

    # Actual prediction
    for i in range(no_row):
      result = classifier(sequence[i], candidate_labels)  # to do multiclass classification set <multi_class=True>
      feedback_predict.append(result['labels'][0])
      feedback_predict_score.append(result['scores'][0]) 

    # Model evaluation
    y_true = df.label[:no_row].values
    y_pred = feedback_predict
    balanced_score = balanced_accuracy_score(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)

    # append the class score to the class dictionary
    for i in range(len(candidate_labels)):
      # check if the class is already a key in the dictionary
      if candidate_labels[i] not in class_dict: 
        class_dict[candidate_labels[i]] = [class_accuracy(candidate_labels[i], y_true, y_pred)]
      else: 
        class_dict[candidate_labels[i]].append(class_accuracy(candidate_labels[i], y_true, y_pred))

    # Update model performance tracking lists
    model_score = round(acc_score*100, 2)   
    model_balanced_score = round(balanced_score*100, 2)    
    model_average_feedback_score = round(np.mean(feedback_predict_score)*100, 2)

    # create the model performance table and populate it for model comparism
    Model_performance_df = pd.DataFrame.from_dict(class_dict)
    Model_performance_df['accuracy_score'] = model_score
    Model_performance_df['balanced_score'] = model_balanced_score
    Model_performance_df['average_feedback_score'] = model_average_feedback_score

    # rearrange the column names to ensure model name come first
    model_df = Model_performance_df[['accuracy_score', 'balanced_score', 'average_feedback_score',
           'Couldn\'t be improved', 'Environment/ facilities', 'Access', 'Communication', 'Dignity', 'Staff', 
           'Care received', 'Transition/coordination', 'Miscellaneous']]
    print(model_df)
    print('\n')
    
    """# Visualize the data and the confusion matrix"""#  
    fig, axs = plt.subplots(2,  
                        figsize=(10,7),     # create a 10x7 figure size for each plot 
                        constrained_layout=True)
    
    model_df[['accuracy_score', 
              'balanced_score', 'average_feedback_score']]\
              .plot.bar(rot = 90, ax=axs[0], figsize=(12,8), fontsize=12, width=0.9)
     
    plot_conf_mat (y_true, y_pred, ax=axs[1])
    axs[1].set_title('Confusion Matrix')


# if __name__ == '__main__':
#     main()    
# ============================================================================================================================================