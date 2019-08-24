import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv('b33.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 31].values

# displaying few rows of the dataset
dataset.head()

# there is no missing data 

# taking the count of class
dataset['diagnosis'].value_counts()
#plotting the graph for classes
dataset['diagnosis'].value_counts().plot('bar')


# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 15
plt.rcParams["figure.figsize"] = fig_size


#plotting all the columns
dataset.hist()
plt.show()

#Response variable for regression
plt.hist(dataset['radius_mean'], bins = 10 , color = 'red' )


# resizing the plots again for feature plots
fig_size[0] = 15
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size

# feature plots
import seaborn as sns
classy = ['B', 'M']
for x in dataset:
    if(x == 'diagnosis'):
        continue
    if(x == 'id'):
        continue
    for classname in classy:
        subset = dataset[dataset['diagnosis'] == classname]
        #draw the density plot 
        sns.distplot(subset[x],hist = False , kde =True,
                     kde_kws = {'linewidth' : 3},
                     label = classname)
        plt.legend(prop={'size': 16}, title = 'Class')
        plt.title('Density Plot with different classes')
        plt.xlabel(x)
        plt.ylabel('Density') 
    plt.show() 
#feature scaling
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=10)
Y_sklearn = sklearn_pca.fit_transform(X_std)  ## now Y_sklearn has 10 main columns 
explained_variance = sklearn_pca.explained_variance_ratio_
import plotly.plotly as py

import plotly 
plotly.tools.set_credentials_file(username='malvikabhalla99', api_key='CWol1Dp0UR0K9IOh22fs')

import plotly 
plotly.tools.set_config_file(world_readable= True,
                             sharing='public')
data = []
colors = {'B': '#0D76BF', 
          'M': '#00cc96'}
for name, col in zip(('B', 'M'), colors.values()):

    trace = dict(
        type='scatter',
        x=Y_sklearn[y==name,0],
        y=Y_sklearn[y==name,1],
        mode='markers',
        name=name,
        marker=dict(
            color=col,
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5),
            opacity=0.8)
    )
    data.append(trace)

layout = dict(
        xaxis=dict(title='PC1', showline=False),
        yaxis=dict(title='PC2', showline=False)
)
fig = dict(data=data, layout=layout)
plot_url = py.iplot(fig, filename='basic-line')

# encoding the y 
#encoding the categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_Y = LabelEncoder()
y=labelencoder_Y.fit_transform(y) # 1 is for malignant and 0 is for benign

# now for multipke regression our X = Y_sklearn and y = y
X=Y_sklearn
y=y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# calculating mse
mse = np.mean((y_pred - y_test)**2)
mse

# evaluation using r-square

regressor.score(X_test,y_test)

#residual plot

x_plot = plt.scatter(y_pred, (y_pred - y_test), c='b')

plt.hlines(y=0, xmin= -0.8, xmax=1.8)

plt.title('Residual plot')

from sklearn.linear_model import Ridge

## training the model

ridgeReg = Ridge(alpha=0.05, normalize=True)

ridgeReg.fit(X_train,y_train)

pred = ridgeReg.predict(X_test)

#calculating mse

mse = np.mean((pred - y_test)**2)

mse  
## calculating score
ridgeReg.score(X_test,y_test) 

# residual plots
from yellowbrick.regressor import ResidualsPlot

# Instantiate the linear model and visualizer
ridge = Ridge()
visualizer = ResidualsPlot(ridge)

visualizer.fit(X_train, y_train)  # Fit the training data to the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof()        

#prediction plots
from sklearn.linear_model import Lasso
from yellowbrick.regressor import PredictionError
lasso = Lasso()
visualizer = PredictionError(lasso)
visualizer.fit(X_train, y_train)  # Fit the training data to the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.poof() 
##Apply different algos as on X_train,X_test,y_train,y_test

#applying decision tree algorithm
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state = 0)
tree.fit(X_train,y_train)

print('Accuracy of the training subset : ',format(tree.score(X_train,y_train)))
print('Accuracy of the test subset : ',format(tree.score(X_test,y_test)))

#tree = DecisionTreeClassifier(max_depth = 4,random_state = 0)
#tree.fit(X_train,y_train)

print('Accuracy of the training subset : ',format(tree.score(X_train,y_train)))
print('Accuracy of the test subset : ',format(tree.score(X_test,y_test)))

y_predtree = tree.predict(X_test)
import graphviz
data2 = [0,1,2,3,4,5,6,7,8,9]
from sklearn.tree import export_graphviz

dot_data = export_graphviz(tree, out_file = 'myoutput.dot' , feature_names = data2,class_names = ['M', 'B'],
                impurity=False,filled=True)

# run the following commands on command prompt to run convert the dot file to png
# first to to the desired folder using cd 
# dot -Tpng myoutput.dot -o myoutput.png

#confusion matrix for decision tree
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predtree)
print(cm)
#plt.matshow(cm)
#plt.colorbar()
plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Positive','Negative']
plt.title('Confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TP','FN'], ['FP', 'TN']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_predtree))

from sklearn.metrics import precision_score
precision_score(y_test,y_predtree,average = 'macro')

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predtree)

from sklearn.metrics import recall_score
recall_score(y_test,y_predtree)












    from sklearn.metrics import precision_recall_curve
precision , recall , thresholds = precision_recall_curve(y_test,y_predtree)
from matplotlib import pyplot
pyplot.plot(recall,precision,'r--',marker = '.')
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
pyplot.title('Precison-Recall')
pyplot.legend(loc = 'upper right' )
pyplot.show()


def plot_prec_recall_thres(precision,recall,thresholds):
    plt.plot(thresholds, precision[:-1],'b--',label = 'precision')
    plt.plot(thresholds, recall[:-1],'g--',label = 'recall')
    plt.xlabel('Threshold')
    plt.legend(loc = 'bottom right')
    
plot_prec_recall_thres(precision,recall,thresholds)
plt.show()




