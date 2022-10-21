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
df=pd.read_csv("PimaIndiansDiabetes.csv")
#df=pd.read_csv("PimaIndiansDiabetes2.csv")
#df=df.dropna() # Remove NaN elements

df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #Remove indexing column which has no information content
mapping = {'pos': 1, 'neg': 0}
df=df.replace({"diabetes":mapping}) #Replace "pos" and "neg" by 1 and 0
init_random=42
train, test = train_test_split(df, test_size=1/3,random_state=init_random)

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
"""
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
"""



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
    #tree.plot_tree(baum)
    #plt.show()

#Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
train_X=train.loc[:, train.columns != 'diabetes']
train_Y=train.loc[:, train.columns == 'diabetes']
test_X=test.loc[:, test.columns != 'diabetes']
test_Y=test.loc[:, test.columns == 'diabetes']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(train_X) #"teach" scaler the correct scaling
train_X=scaler.transform(train_X)
test_X=scaler.transform(test_X)
train_X=np.array(train_X)
test_X=np.array(test_X)
train_Y=np.array(train_Y)
test_Y=np.array(test_Y)
print("Bagging")
bagging = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=3),n_estimators=1000, random_state=init_random).fit(train_X, train_Y.ravel())
train_predict=bagging.predict(train_X)
test_predict=bagging.predict(test_X)
print("Train correctness: %f"%(accuracy_score(train_predict,train_Y)))
print("Test correctness: %f"%(accuracy_score(test_predict,test_Y)))
print("Boosting")
boosting = AdaBoostClassifier(n_estimators=1000, random_state=init_random,learning_rate=0.01).fit(train_X, train_Y.ravel())
print(boosting.estimator_errors_)
train_predict=boosting.predict(train_X)
test_predict=boosting.predict(test_X)
print("Train correctness: %f"%(accuracy_score(train_predict,train_Y)))
print("Test correctness: %f"%(accuracy_score(test_predict,test_Y)))

sys.exit(1)
import pygam
from pygam import LogisticGAM, s, f
from pygam.datasets import wage
train_X=train.loc[:, train.columns != 'diabetes']
train_Y=train.loc[:, train.columns == 'diabetes']
test_X=test.loc[:, test.columns != 'diabetes']
test_Y=test.loc[:, test.columns == 'diabetes']
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(train_X) #"teach" scaler the correct scaling
train_X=scaler.transform(train_X)
test_X=scaler.transform(test_X)
train_X=np.array(train_X)
test_X=np.array(test_X)
"""
n_splines=10
formula = s(0, n_splines)
for i in range(1, train_X.shape[1]):
    if i==2 or i==6 or i==7:
        n_splines_i=n_splines
        #n_splines_i=len(Counter(train_X[:,i]).keys()) #Number of
    else:
        n_splines_i=n_splines
    formula = formula + s(i, n_splines_i)
#lams=np.array([1,1,1,1,1,1,1,1])
"""
np.random.seed(init_random)
lams=np.exp(12*np.random.rand(40, 8)+2)



gam = LogisticGAM().gridsearch(train_X,train_Y,lam=lams,objective="AIC")
summary=gam.summary()
print(gam.statistics_["AIC"])
train_Y_predict=gam.predict(train_X)
test_Y_predict=gam.predict(test_X)
print("Train:")
print(accuracy_score(train_Y_predict,train_Y))
print("Test:")
print(accuracy_score(test_Y_predict,test_Y))


def backward_selection_pyGAM(X,y,criterion="AIC"):
    """
    Perform backward elimination to obtain the best model.

    Input:
        X: An array containing the predictors
        y: An array dataframe containing the response
        criterion: The selection criterion. Should be "AIC" or one of the ones supported in pyGAM.
    """
    full_model=X
    all_variables=list(range(X.shape[1])) #all variables. lol
    current_X=full_model
    lams=np.exp(10*np.random.rand(10, 8)-2)
    current_model = LogisticGAM().gridsearch(full_model,y,lam=lams,objective="AIC")
    if criterion=="AIC":
        best_crit=current_model.statistics_["AIC"]
    curr_crit=best_crit*10
    removed=[] #Columns so far removed
    curent_columns=list(range(X.shape[1]))
    while best_crit<curr_crit:
        curr_crit=best_crit #Best model is current model

        new_crits=[]
        for i in range(len(curent_columns)):
            new_columns=curent_columns.copy()
            del new_columns[i] # Delete i'th element from list (e.g. iterate through all)
            new_X=full_model[:, new_columns]
            print(new_X.shape)
            new_model = LogisticGAM().gridsearch(new_X,y,lam=lams[:,new_columns],objective="AIC")
            if criterion=="AIC":
                new_crit=new_model.statistics_["AIC"]
                print(new_columns,new_crit)
            new_crits.append(new_crit)
        best_crit=min(new_crits) #Best model with this number of variables is the one with the lowest AIC/BIC
        if best_crit>=curr_crit:
            return current_model, curent_columns #Current model is the best
        best_crit_index_temp=np.argmin(new_crits)
        best_crit_index=curent_columns[best_crit_index_temp]
        to_remove=best_crit_index
        curent_columns.remove(to_remove)
        current_X=full_model[:, curent_columns]
        current_model = LogisticGAM().gridsearch(current_X,y,lam=lams[:,curent_columns],objective="AIC")
        if current_X.shape[1]==0: #If the best model is the constant model
            return current_model
best_model,curent_columns=backward_selection_pyGAM(train_X,train_Y)
summary=best_model.summary()
gam=best_model
print(curent_columns)
print(best_model.statistics_["AIC"])
train_Y_predict=best_model.predict(train_X[:,curent_columns])
test_Y_predict=best_model.predict(test_X[:,curent_columns])
print("Train:")
print(accuracy_score(train_Y_predict,train_Y))
print("Test:")
print(accuracy_score(test_Y_predict,test_Y))
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.show()
