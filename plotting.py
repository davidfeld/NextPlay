import matplotlib.pyplot as plt  # to plot
import pandas as pd  # to handle data
import statsmodels.api as sm  # to explore model
import itertools
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score  # metrics
from sklearn.linear_model import LogisticRegression  # model
from sklearn.tree import DecisionTreeClassifier  # model
from sklearn.ensemble import RandomForestClassifier  # model
from sklearn.cross_validation import cross_val_score # model
from sklearn.neighbors import KNeighborsClassifier  # model
import pylab as pl
import numpy as np  # functions to work with numerical data
import seaborn as sns  # pretty plots
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import svm, datasets


#function to plot play distributions
def playdistplt(df):
    y = list(df['PlayType'])
    yset = list(set(y))
    counts = []
    for i in yset:
        counts.append(y.count(i))
    colors = ['r','k','b','yellow','g', 'magenta','orange']
    fig, ax = plt.subplots()
    width = 0.4

    dic = {i:(yset[i],colors[i]) for i in range(len(yset))}
    ax1 = plt.subplot(111)
    xval = range(len(yset))

    for j in range(len(xval)):
        ax1.bar(xval[j], counts[j], width=0.8, bottom=0.0, align='center', color=dic[xval[j]][1], alpha=0.6, label=dic[xval[j]][0])
    ax1.set_xticks(xval)
    ax1.set_xticklabels([dic[i][0] for i in xval])
    #ax1.set_facecolor('forestgreen')
    ax1.legend()
    plt.show()
    fig.savefig('playdist.png',figsize=(14, 14))

#function to plot 4th down play distributions
def playdist4dplt(df):
    t1 = df[df['down'] == 4].PlayType
    t1 = list(t1)
    t1set = list(set(t1))
    counts = []
    for i in t1set:
        counts.append(t1.count(i))
    colors = ['r','k','b','yellow','g', 'magenta','orange']
    fig, ax = plt.subplots()
    width = 0.4
    dic = {i:(t1set[i],colors[i]) for i in range(len(t1set))}
    ax1 = plt.subplot(111)
    xval = range(len(t1set))

    for j in range(len(xval)):
        ax1.bar(xval[j], counts[j], width=0.8, bottom=0.0, align='center', color=dic[xval[j]][1], alpha=0.6, label=dic[xval[j]][0])
    ax1.set_xticks(xval)
    ax1.set_xticklabels([dic[i][0] for i in xval])
    ax1.legend()
    plt.title('4th Down')
    plt.show()
    fig.savefig('playdist4d.png',figsize=(14, 14))


# Plot the importance of features
def plot_importance(clf,X):
    feature_importances = clf.feature_importances_
    #print len(feature_importances)
    ziptry = zip(feature_importances,list(X))
    ziptry.sort()
    #for i in ziptry:
        #print i
    feature_importances.sum()
    features = pd.DataFrame()
    features['features'] = list(X)
    features['importance'] = feature_importances
    features.sort_values(by=['importance'],ascending=False,inplace=True)
    plt.figure()
    fig,ax= plt.subplots()
    fig.set_size_inches(20,10)
    plt.xticks(rotation=60)
    ax.tick_params(labelsize=20)
    #sns.barplot(data=features.head(10),x="features", y="importance",ax=ax,orient="v",color='g')
    sns.barplot(data=features.head(10), x="importance", y="features",ax=ax,orient="h",color='g')
    plt.title('Feature Importance', fontsize = 20)
    plt.show()
    #print(features[['features', 'importance']])
    fig.savefig('featimport.png',figsize=(14, 14))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 17)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

