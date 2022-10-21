import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("PimaIndiansDiabetes2.csv")
df=df.dropna() # Remove NaN elements

df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #Remove indexing column which has no information content
mapping = {'pos': 1, 'neg': 0}
df=df.replace({"diabetes":mapping}) #Replace "pos" and "neg" by 1 and 0
train, test = train_test_split(df, test_size=1/3,random_state=101)

print(train)
print(test)

#Check that approximately the same number of people have diabetes in the train and test set.
vals=train["diabetes"].value_counts()
print(np.array(vals)/len(train["diabetes"]))
vals=test["diabetes"].value_counts()
print(np.array(vals)/len(test["diabetes"]))
train_X=np.array(train.loc[:, train.columns != 'diabetes'])
train_Y=np.array(train.loc[:, train.columns == 'diabetes'])
test_X=np.array(test.loc[:, test.columns != 'diabetes'])
test_Y=np.array(test.loc[:, test.columns == 'diabetes'])
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(train_X) #"teach" scaler the correct scaling
train_X=scaler.transform(train_X)
test_X=scaler.transform(test_X)

"""5 Fold Cross validation"""
from sklearn.model_selection import KFold #Cross validation
num_cv=5 #5 Folds
num_neighbours=np.array(np.linspace(1,50,50),dtype=int) #Consider 1 to 20 neighbours
CV_errs=np.zeros((len(num_neighbours),num_cv))
kf = KFold(n_splits=num_cv)
i=0
j=0
for train_index,test_index in kf.split(train_X):
    i=0
    train_X_CV,test_X_CV=train_X[train_index],train_X[test_index]
    train_Y_CV,test_Y_CV=train_Y[train_index],train_Y[test_index]
    for num_neighbour in num_neighbours:
        neigh = KNeighborsClassifier(n_neighbors=num_neighbour)
        neigh.fit(train_X_CV, np.array(train_Y_CV).ravel())
        y_predict_CV=neigh.predict(test_X_CV)
        CV_errs[i,j]=accuracy_score(y_predict_CV,test_Y_CV)
        i+=1
    j+=1
five_fold_accuracy=np.mean(CV_errs,axis=1) #The CV error
five_fold_test_errors=np.zeros(len(five_fold_accuracy)) #The actual test error
for i,k in enumerate(num_neighbours):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_X, np.array(train_Y).ravel())
    y_predict=neigh.predict(test_X)
    five_fold_test_errors[i]=accuracy_score(y_predict,test_Y)
best_k=num_neighbours[np.argmax(five_fold_accuracy)]
print(best_k)
neigh = KNeighborsClassifier(n_neighbors=best_k)
neigh.fit(train_X, np.array(train_Y).ravel())
y_predict=neigh.predict(test_X)
print(accuracy_score(y_predict,test_Y))

"""LOO Cross validation"""
num_cv=len(train_X[:,0]) #CV folds
CV_errs=np.zeros((len(num_neighbours),num_cv))
kf = KFold(n_splits=num_cv)
i=0
j=0
for train_index,test_index in kf.split(train_X):
    i=0
    train_X_CV,test_X_CV=train_X[train_index],train_X[test_index]
    train_Y_CV,test_Y_CV=train_Y[train_index],train_Y[test_index]
    for num_neighbour in num_neighbours:
        neigh = KNeighborsClassifier(n_neighbors=num_neighbour)
        neigh.fit(train_X_CV, np.array(train_Y_CV).ravel())
        y_predict_CV=neigh.predict(test_X_CV)
        CV_errs[i,j]=accuracy_score(y_predict_CV,test_Y_CV)
        i+=1
    j+=1
LOO_accuracy=np.mean(CV_errs,axis=1)
plt.plot(num_neighbours,five_fold_test_errors,label="Test error ")
plt.plot(num_neighbours,five_fold_accuracy,label="5-Fold test error")
plt.plot(num_neighbours,LOO_accuracy,label="LOO test error")
plt.xlabel("Number of nearest neighbours k")
plt.ylabel("Classification accuracy")
plt.title("kNN accuracy as function of k")
plt.legend()
plt.tight_layout()
plt.savefig("Test errors.pdf")
plt.show()
sys.exit(1)
best_k=num_neighbours[np.argmax(five_fold_accuracy)]
print(best_k)
neigh = KNeighborsClassifier(n_neighbors=best_k)
neigh.fit(train_X, np.array(train_Y).ravel())
y_predict=neigh.predict(test_X)
print(accuracy_score(y_predict,test_Y))

best_k=num_neighbours[np.argmax(LOO_accuracy)]
print(best_k)
neigh = KNeighborsClassifier(n_neighbors=best_k)
neigh.fit(train_X, np.array(train_Y).ravel())
y_predict=neigh.predict(test_X)
print(accuracy_score(y_predict,test_Y))

for i in range(1,21):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(train_X, np.array(train_Y).ravel())
    train_predict=neigh.predict(train_X)
    test_predict=neigh.predict(test_X)
    print("Number of neighbors: %d"%i)
    print("Train correctness: %f"%(accuracy_score(train_predict,train_Y)))
    print("Test correctness: %f"%(accuracy_score(test_predict,test_Y)))
#K nearest neighbors Classifier
from sklearn import tree
for k,i in enumerate(np.logspace(-3,-1,10)):
    baum = tree.DecisionTreeClassifier(ccp_alpha=i)
    baum.fit(train_X, np.array(train_Y).ravel())
    train_predict=baum.predict(train_X)
    test_predict=baum.predict(test_X)
    print("Penalty: %f"%i)
    print("Train correctness: %f"%(accuracy_score(train_predict,train_Y)))
    print("Test correctness: %f"%(accuracy_score(test_predict,test_Y)))
    tree.plot_tree(baum)
    plt.show()
