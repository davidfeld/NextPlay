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


### Function to make dummy columns. Need dummies for categorical features to feed into classifiers.
def make_dummies(X,column,df):
    dummy = pd.get_dummies(df[column],prefix = column)
    X = X.join(dummy.astype(int), lsuffix= 'P')
    X = X.drop([column], axis =1)
    return X

### Function to engineer features.
def addfeatures(df):
    # Add 'SOFpos4d' feature
    df['SOFpos4d'] = 0
    df['SOFpos4d'][ (df['SideofField'] == df['posteam']) & (df['down'] == 4) ] =1
    df['SOFpos4d'][ (df['SideofField'] != df['posteam']) & (df['down'] == 4) ] =2
    return df
    
### Make df with just predictor columns. PlayType is the target column.
def makefeatures(df):
    X = df.drop(['PlayType'], axis =1)

    ## Columns where you need to make dummies
    dummy_columns = ['down','HomeTeam','AwayTeam']
    # Make dummies
    for column in dummy_columns:
        X = make_dummies(X,column,df)
    return X

# Make target classification array
def maketarget(df):
    y = df['PlayType']
    dic = {'Run':'Run','Pass':'Pass','Field Goal': 'FG', 'Punt': 'Punt'}
    ### Essentially just doing this to abbreviate Field Goal as FG.
    y = y.map(dic)
    return y

# Make an oversampled dataframe
def make_oversample(df,classes):
    #Find which class has most values
    maxim, max_class = 0, ''
    for label in classes:
        if len(df.loc[df['PlayType' ]== label]) > maxim:
            maxim = len(df.loc[df['PlayType' ]== label])
            max_class = label
    
    df_os = df.loc[df['PlayType' ]== max_class]
    classes.remove(max_class)
    
    #Duplicate other classes until there is as many as the max column
    for label in classes:
        df_temp = df.loc[df['PlayType' ]== label]
        while len(df_temp) < maxim:
            df_temp = df_temp.append([df_temp],ignore_index=True)
        df_temp = df_temp.iloc[:maxim]
        df_os = df_os.append([df_temp],ignore_index=True)
    
    #Put dataframe in random order
    df_os = df_os.sample(frac=1)

    return df_os

#Make a Random Forest
def makeRF(X,y):
    #Split dataset into train and test dataset
    train_percentage = 0.7
    train_x, test_x, train_y, test_y = train_test_split(X, y,
                                                            train_size=train_percentage)


    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_x, train_y)
    return clf, test_x, test_y

#Make a Random Forest for oversampled dataframe. This needs a seperate function because I am oversampling
#while cross validating.
def oversample_clf(clf, df, nfolds):
    df = df.sample(frac=1)  ## randomize order
    accuracy = []
    test_length = len(df)/nfolds

    #nfolds is the number of cross validation folds.
    #For each fold I run oversample a portion of the data, train a RF, and test it on the rest of the data.
    for i in range(nfolds):
        print "Fold #"+str(i+1)
        # Make test and training dataframes
        test_df = df.iloc[i*test_length:(i+1)*test_length]
        train_df1 = df.iloc[:i*test_length]
        train_df2 = df.iloc[(i+1)*test_length:]
        train_df = train_df1.append([train_df2],ignore_index=True)
        
        if nfolds == 1: train_df = test_df

        #Oversample training dataframe
        classes = ['Pass','Run','Punt','Field Goal']
        train_df = make_oversample(train_df, classes)

        train_x = makefeatures(train_df) 
        train_y = maketarget(train_df)
        test_x = makefeatures(test_df) 
        test_y = maketarget(test_df)

        #clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train_x, train_y)
        accuracy.append(clf.score(test_x,test_y))
    return accuracy, clf, test_x, test_y
