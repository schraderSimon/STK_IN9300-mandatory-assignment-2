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
from sklearn.metrics import accuracy_score, zero_one_loss
#df=pd.read_csv("PimaIndiansDiabetes.csv")
df=pd.read_csv("PimaIndiansDiabetes2.csv")
df=df.dropna() # Remove NaN elements

df = df.loc[:, ~df.columns.str.contains('^Unnamed')] #Remove indexing column which has no information content
mapping = {'pos': 1, 'neg': 0}
df=df.replace({"diabetes":mapping}) #Replace "pos" and "neg" by 1 and 0
init_random=11 #Two sets with almost the same number of diabetic people
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
train_X=scaler.transform(train_X) #Scale data for k-nearest neighbours
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
        CV_errs[i,j]=1-accuracy_score(y_predict_CV,test_Y_CV)
        i+=1
    j+=1
five_fold_accuracy=np.mean(CV_errs,axis=1) #The CV error
five_fold_test_errors=np.zeros(len(five_fold_accuracy)) #The actual test error
for i,k in enumerate(num_neighbours):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_X, np.array(train_Y).ravel())
    y_predict=neigh.predict(test_X)
    five_fold_test_errors[i]=1-accuracy_score(y_predict,test_Y)
best_k_5Fold=num_neighbours[np.argmin(five_fold_accuracy)]-1
neigh = KNeighborsClassifier(n_neighbors=best_k_5Fold)
neigh.fit(train_X, np.array(train_Y).ravel())
y_predict=neigh.predict(test_X)

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
        CV_errs[i,j]=1-accuracy_score(y_predict_CV,test_Y_CV)
        i+=1
    j+=1
LOO_accuracy=np.mean(CV_errs,axis=1)
best_k_LOO=num_neighbours[np.argmin(LOO_accuracy)]-1

plt.plot(num_neighbours,five_fold_test_errors,label="Test error ")
plt.plot(num_neighbours,five_fold_accuracy,label="5-Fold error")
plt.plot(num_neighbours,LOO_accuracy,label="LOO error")
plt.plot(num_neighbours[best_k_LOO],LOO_accuracy[best_k_LOO],"o",color="red",label="best k")
plt.plot(num_neighbours[best_k_5Fold],five_fold_accuracy[best_k_5Fold],"o",color="red",)
plt.xlabel("Number of nearest neighbours k")
plt.ylabel("Misclassification error")
plt.title("kNN misclassification error as function of k")
plt.legend()
plt.tight_layout()
plt.savefig("Test_errors_kNN_nan.pdf")
plt.show()
neigh = KNeighborsClassifier(n_neighbors=best_k_5Fold)
neigh.fit(train_X, np.array(train_Y).ravel())
y_predict_train=neigh.predict(train_X)
y_predict_test=neigh.predict(test_X)

five_fold_train_error=1-accuracy_score(y_predict_train,train_Y)
five_fold_test_error=1-accuracy_score(y_predict_test,test_Y)

neigh = KNeighborsClassifier(n_neighbors=best_k_LOO)
neigh.fit(train_X, np.array(train_Y).ravel())
y_predict_train=neigh.predict(train_X)
y_predict_test=neigh.predict(test_X)

LOO_train_error=1-accuracy_score(y_predict_train,train_Y)
LOO_test_error=1-accuracy_score(y_predict_test,test_Y)

print("5-fold train error: %f"%five_fold_train_error)
print("5-fold test error: %f"%five_fold_test_error)
print("LOO train error: %f"%LOO_train_error)
print("LOO test error: %f"%LOO_test_error)

"""Tree-based methods"""
from sklearn import tree
for k,i in enumerate([0.01]):
    baum = tree.DecisionTreeClassifier(ccp_alpha=i)
    baum.fit(train_X, np.array(train_Y).ravel())
    train_predict=baum.predict(train_X)
    test_predict=baum.predict(test_X)
    print("Tree with Penalty alpha: %f"%i)
    print("Train error: %f"%(1-accuracy_score(train_predict,train_Y)))
    print("Test error: %f"%(1-accuracy_score(test_predict,test_Y)))
    #tree.plot_tree(baum)
    #plt.show()
#Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

train_X=np.array(train_X)
test_X=np.array(test_X)
train_Y=np.array(train_Y)
test_Y=np.array(test_Y)
print("Bagging")
bagging = BaggingClassifier(base_estimator=tree.DecisionTreeClassifier(min_samples_leaf=10),n_estimators=2000, random_state=init_random).fit(train_X, train_Y.ravel())
bagging_estimators=bagging.estimators_
train_predict_voting=np.zeros(len(train_Y))
train_predict_probability=np.zeros(len(train_Y))
test_predict_voting=np.zeros(len(test_Y))
test_predict_probability=np.zeros(len(test_Y))
for tree in bagging_estimators: #iterate over trees
    probs_train = tree.predict_proba(train_X)[:,1]
    train_predict_voting+=np.rint(probs_train) # Round to int, e.g. vote
    train_predict_probability+=probs_train
    probs_test= tree.predict_proba(test_X)[:,1]
    test_predict_voting+=np.rint(probs_test) # Round to int, e.g. vote
    test_predict_probability+=probs_test
#Divide by number of trees
train_predict_voting/=len(bagging_estimators)
train_predict_probability/=len(bagging_estimators)
test_predict_voting/=len(bagging_estimators)
test_predict_probability/=len(bagging_estimators)
train_predict_voting=np.rint(train_predict_voting)
train_predict_probability=np.rint(train_predict_probability)
test_predict_voting=np.rint(test_predict_voting)
test_predict_probability=np.rint(test_predict_probability)

print("Train error voting: %f"%(1-accuracy_score(train_predict_voting,train_Y)))
print("Test error voting: %f"%(1-accuracy_score(test_predict_voting,test_Y)))
print("Train error probability: %f"%(1-accuracy_score(train_predict_probability,train_Y)))
print("Test error probabibility: %f"%(1-accuracy_score(test_predict_probability,test_Y)))

print("Random forest")
forest = RandomForestClassifier(min_samples_leaf=10,max_features=3,n_estimators=2000, random_state=init_random).fit(train_X, train_Y.ravel())
train_predict=forest.predict(train_X)
test_predict=forest.predict(test_X)
print("Train error: %f"%(1-accuracy_score(train_predict,train_Y)))
print("Test error: %f"%(1-accuracy_score(test_predict,test_Y)))

print("Boosting")
boosting = AdaBoostClassifier(n_estimators=2000, random_state=init_random,learning_rate=0.01).fit(train_X, train_Y.ravel())
train_predict=boosting.predict(train_X)
test_predict=boosting.predict(test_X)
print("Train error: %f"%(1-accuracy_score(train_predict,train_Y)))
print("Test error: %f"%(1-accuracy_score(test_predict,test_Y)))


"""GAM"""
import pygam
from pygam import LogisticGAM, s, f
from pygam.datasets import wage

np.random.seed(init_random)
lams_linear=np.broadcast_to(np.logspace(0,4,10),(8,10)).T #Lamdas (same for all variables) from 1e-1 to 1e4
lams_random=np.exp(7*np.random.rand(300, 8)+2)
lams=np.concatenate((lams_random,lams_linear),axis=0)
criterion="AIC"
gam = LogisticGAM().gridsearch(train_X,train_Y,lam=lams,objective=criterion)
summary=gam.summary()
train_Y_predict=gam.predict(train_X)
test_Y_predict=gam.predict(test_X)
print("FULL GAM")
print("Train:error: %f"%(1-accuracy_score(train_Y_predict,train_Y)))
print("Test error: %f"%(1-accuracy_score(test_Y_predict,test_Y)))
def backward_selection_pyGAM(X,y,lambda_values,criterion="AIC"):
    """
    Perform backward elimination to obtain the best model.

    Input:
        X: An array containing the predictors
        y: An array dataframe containing the response
        lambda_values: The penalty parameters to iterate over
        criterion: The selection criterion. Should be "AIC" or one of the ones supported in pyGAM.
    """
    lams=lambda_values
    full_model=X
    all_variables=list(range(X.shape[1])) #all variables. lol
    current_X=full_model
    #lams=np.exp(7*np.random.rand(100, 8)+2) #Convergence issues when lambdas are too small
    current_model = LogisticGAM().gridsearch(full_model,y,lam=lams,objective=criterion)
    best_crit=current_model.statistics_[criterion]
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
            new_model = LogisticGAM().gridsearch(new_X,y,lam=lams[:,new_columns],objective=criterion)
            new_crit=new_model.statistics_[criterion]
            new_crits.append(new_crit)
        best_crit=min(new_crits) #Best model with this number of variables is the one with the lowest criterion
        if best_crit>=curr_crit:
            return current_model, curent_columns #Current model is the best
        best_crit_index_temp=np.argmin(new_crits)
        best_crit_index=curent_columns[best_crit_index_temp]
        to_remove=best_crit_index
        curent_columns.remove(to_remove)
        current_X=full_model[:, curent_columns]
        current_model = LogisticGAM().gridsearch(current_X,y,lam=lams[:,curent_columns],objective=criterion)
        if current_X.shape[1]==0: #If the best model is the constant model
            return current_model
best_model,current_columns=backward_selection_pyGAM(train_X,train_Y,lams,criterion)
summary=best_model.summary()
gam=best_model
train_Y_predict=best_model.predict(train_X[:,current_columns])
test_Y_predict=best_model.predict(test_X[:,current_columns])
print("Best GAM")
print("Train:")
print(1-accuracy_score(train_Y_predict,train_Y))
print("Test:")
print(1-accuracy_score(test_Y_predict,test_Y))
XX_plot=np.zeros((100,8))

for i in range(len(current_columns)):
    XX = gam.generate_X_grid(term=i)
    XX_plot[:,current_columns[i]]=XX[:,i]
plotty=scaler.inverse_transform(XX_plot)
fig,axes=plt.subplots(2,2)
axes_helper=[]
for i in range(2):
    for j in range(2):
        axes_helper.append(axes[i,j])
units=["n. preg.","concentration"," pressure (mm HG)", "thickness (mm)","insulin (μU/mL)","BMI","ped. func.","age (years)"]
for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue
    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    axes_helper[i].plot(plotty[:,current_columns[i]], pdep)
    axes_helper[i].plot(plotty[:,current_columns[i]], confi, c='r', ls='--')
    axes_helper[i].set_title(train.columns[current_columns[i]])
    axes_helper[i].set_ylabel("Effect of variable")
    axes_helper[i].set_xlabel(units[current_columns[i]])
plt.tight_layout()
plt.savefig("Effect of variables_nan.pdf")
plt.show()
