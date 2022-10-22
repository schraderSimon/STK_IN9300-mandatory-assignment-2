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
def forward_selection(X,y,criterion="AIC"):
    """
    Perform forward selection to obtain the best model.

    Input:
        X: A Pandas dataframe containing the predictors
        y: A Pandas dataframe containing the response
        criterion: The selection criterion. Can be "AIC" or "BIC".
    """
    full_model=X
    all_variables=list(full_model.columns)
    empty_model=full_model.loc[:, :'const'] #Keep only the intercept
    current_X=empty_model
    current_model = sm.OLS(y, current_X)
    current_est = current_model.fit()
    if criterion=="AIC":
        best_crit=current_est.aic
    elif criterion=="BIC":
        best_crit=current_est.bic
    curr_crit=best_crit*10
    while best_crit<curr_crit:
        curr_crit=best_crit #Best model is current model
        curent_columns=list(current_X.columns)
        not_added_yet = list(set(all_variables) - set(curent_columns))
        if len(not_added_yet)==0: #If the best model is the full model
            return current_est
        new_crits=[]
        for predictor in not_added_yet:
            new_columns=curent_columns+[predictor]
            new_X=full_model.loc[:, new_columns]
            new_model = sm.OLS(y, new_X)
            new_est = new_model.fit()
            if criterion=="AIC":
                new_crit=new_est.aic
            elif criterion=="BIC":
                new_crit=new_est.bic
            new_crits.append(new_crit)
        best_crit=min(new_crits) #Best model with this number of variables is the one with the lowest AIC/BIC
        if best_crit>=curr_crit:
            return current_est #Current model is the best
        best_crit_index=np.argmin(new_crits)
        current_X=full_model.loc[:, curent_columns+[not_added_yet[best_crit_index]]]
        current_model = sm.OLS(y, current_X)
        current_est = current_model.fit()
def backward_selection(X,y,criterion="AIC"):
    """
    Perform backward elimination to obtain the best model.

    Input:
        X: A Pandas dataframe containing the predictors
        y: A Pandas dataframe containing the response
        criterion: The selection criterion. Can be "AIC" or "BIC".
    """
    full_model=X
    all_variables=list(full_model.columns)
    current_X=full_model
    current_model = sm.OLS(y, current_X)
    current_est = current_model.fit()
    if criterion=="AIC":
        best_crit=current_est.aic
    elif criterion=="BIC":
        best_crit=current_est.bic
    curr_crit=best_crit*10
    removed=[] #Columns so far removed
    while best_crit<curr_crit:
        curr_crit=best_crit #Best model is current model
        curent_columns=list(current_X.columns)
        not_removed_yet = list(set(all_variables) - set(removed))
        if len(not_removed_yet)==1: #If the best model is the constant model
            return current_est
        new_crits=[]
        for predictor in not_removed_yet:
            new_columns=list(set(curent_columns) - set([predictor]))
            new_X=full_model.loc[:, new_columns]
            new_model = sm.OLS(y, new_X)
            new_est = new_model.fit()
            if criterion=="AIC":
                new_crit=new_est.aic
            elif criterion=="BIC":
                new_crit=new_est.bic
            if predictor=="const": #Not removing the intercept, can be done by setting the criterion without the intercept artificially high
                new_crit=1e100
            new_crits.append(new_crit)
        best_crit=min(new_crits) #Best model with this number of variables is the one with the lowest AIC/BIC
        if best_crit>=curr_crit:
            return current_est #Current model is the best
        best_crit_index=np.argmin(new_crits)

        to_remove=not_removed_yet[best_crit_index]
        curent_columns_new=list(set(curent_columns) - set([to_remove]))
        current_X=full_model.loc[:, curent_columns_new]
        current_model = sm.OLS(y, current_X)
        current_est = current_model.fit()
        removed.append(not_removed_yet[best_crit_index]) #Add the removed variable to the list of removed variables

colnames=["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040","LC50"]
df=pd.read_csv("qsar_aquatic_toxicity.csv",delimiter=";",names=colnames)
df=sm.add_constant(df) #Add intercept
df_dich=df.copy() #Create new copy of data in order to access
df_dich["H050"][df_dich["H050"]>=1]=1; #Dichotomize
df_dich["nN"][df_dich["nN"]>=1]=1;
df_dich["C040"][df_dich["C040"]>=1]=1;


n_runs=200
init_random=42
np.random.seed(seed=init_random) #reproducibility
mse_test_list=np.zeros(n_runs)
mse_test_dich_list=np.zeros(n_runs)
mse_train_list=np.zeros(n_runs)
mse_train_dich_list=np.zeros(n_runs)
def MSE(A,B):
    err=(np.array(A).flatten()-np.array(B).flatten())**2
    return np.sum(err)/len(err) #divide by number of data
for i in range(n_runs):
    train, test = train_test_split(df, test_size=1/3,random_state=i+init_random) #Split into train and test data with random state $i$ for reproducibility
    train_dich, test_dich = train_test_split(df_dich, test_size=1/3,random_state=i+init_random) #Apply same split to dichotomized data for comparability
    train_X=train.loc[:, train.columns != 'LC50']
    train_Y=train.loc[:, train.columns == 'LC50']
    test_X=test.loc[:, test.columns != 'LC50']
    test_Y=test.loc[:, test.columns == 'LC50']
    train_X_dich=train_dich.loc[:, train_dich.columns != 'LC50']
    test_X_dich=test_dich.loc[:, test_dich.columns != 'LC50']
    train_Y_dich=train_dich.loc[:, train_dich.columns == 'LC50']
    test_Y_dich=test_dich.loc[:, test_dich.columns == 'LC50']
    model1 = sm.OLS(train_Y, train_X)
    model2 = sm.OLS(train_Y_dich,train_X_dich)
    est1 = model1.fit()
    est2 = model2.fit()
    y_test_predict=est1.predict(test_X)
    y_test_predict_dich=est2.predict(test_X_dich)
    y_train_predict=est1.predict(train_X)
    y_train_predict_dich=est2.predict(train_X_dich)
    mse_test_list[i]=MSE(y_test_predict,test_Y)
    mse_test_dich_list[i]=MSE(y_test_predict_dich,test_Y_dich)
    mse_train_list[i]=MSE(y_train_predict,train_Y)
    mse_train_dich_list[i]=MSE(y_train_predict_dich,train_Y_dich)
    if i==0:
        print(est1.summary())
        print("Train error: %f"%mse_train_list[i])
        print("Test error: %f"%mse_test_list[i])
        print("End of Non-dichotomized data")
        print("-------------------------------------")
    if i==0:
        print(est2.summary())
        print("Train error: %f"%mse_train_dich_list[i])
        print("Test error: %f"%mse_test_dich_list[i])
        print("End of Dichotomized data")
        print("-------------------------------------")
print("Averaged over %d runs, the mean train and test errors are:"%n_runs)
print("Non-dichotomized data:")
print("Test error: %f"%np.mean(mse_test_list))
print("Train error: %f"%np.mean(mse_train_list))
print("dichotomized data:")
print("Test error: %f"%np.mean(mse_test_dich_list))
print("Train error: %f"%np.mean(mse_train_dich_list))

#Return to original train_test_split
train, test = train_test_split(df, test_size=1/3,random_state=init_random) #Split into train and test data with random state $i$ for reproducibility
train_X=train.loc[:, train.columns != 'LC50']
train_Y=train.loc[:, train.columns == 'LC50']
test_X=test.loc[:, test.columns != 'LC50']
test_Y=test.loc[:, test.columns == 'LC50']
best_model_FW_AIC=forward_selection(train_X,train_Y,criterion="AIC")
best_model_BW_AIC=backward_selection(train_X,train_Y,criterion="AIC")
best_model_FW_BIC=forward_selection(train_X,train_Y,criterion="BIC")
best_model_BW_BIC=backward_selection(train_X,train_Y,criterion="BIC")
print("Forward AIC")
print(best_model_FW_AIC.summary())
print("BW AIC")
print(best_model_BW_AIC.summary())
print("Forward BIC")
print(best_model_FW_BIC.summary())
print("BW BIC")
print(best_model_BW_BIC.summary())
predict_test_y_reduced=best_model_BW_BIC.predict(test_X.loc[:,list(dict(best_model_BW_BIC.params).keys())])
predict_train_y_reduced=best_model_BW_BIC.predict(train_X.loc[:,list(dict(best_model_BW_BIC.params).keys())])
print("Train error BW BIC: %f"%MSE(predict_train_y_reduced,train_Y))
print("Test error BW BIC: %f"%MSE(predict_test_y_reduced,test_Y))

from sklearn.preprocessing import StandardScaler
train_X=np.array(train_X)[:,1:]
test_X=np.array(test_X)[:,1:]

train_Y=np.array(train_Y)
test_Y=np.array(test_Y)
train_Y_mean=np.mean(train_Y)
train_Y=train_Y-train_Y_mean
test_Y=test_Y-train_Y_mean
scaler=StandardScaler()
scaler.fit(train_X) #"teach" scaler the correct scaling
train_X=scaler.transform(train_X)
test_X=scaler.transform(test_X)



from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold
#regr = RidgeCV(fit_intercept=False)
#regr.fit(train_X, train_Y)
#y_pred = regr.predict(test_X)
#print(MSE(y_pred,test_Y))

alpha_vals=np.logspace(-1,np.log10(40),100)
num_cv=10 #CV folds
CV_errs=np.zeros((len(alpha_vals),num_cv))
kf = KFold(n_splits=num_cv)
i=0
j=0
for train_index,test_index in kf.split(train_X):
    i=0
    train_X_CV,test_X_CV=train_X[train_index],train_X[test_index]
    train_Y_CV,test_Y_CV=train_Y[train_index],train_Y[test_index]
    for alpha_val in alpha_vals:
        regr = Ridge(alpha=alpha_val,fit_intercept=False)
        regr.fit(train_X_CV, train_Y_CV)
        y_predict_CV=regr.predict(test_X_CV)
        CV_errs[i,j]=MSE(y_predict_CV,test_Y_CV)
        i+=1
    j+=1
err_lambda_CV=np.mean(CV_errs,axis=1)
best_lambda=alpha_vals[np.argmin(err_lambda_CV)]
min_lambda_err_CV=np.argmin(err_lambda_CV)
regr = Ridge(alpha=best_lambda,fit_intercept=False)
regr.fit(train_X, train_Y)
y_pred_test = regr.predict(test_X)
y_pred_train = regr.predict(train_X)
print("Optimal Lambda (CV): %f"%alpha_vals[min_lambda_err_CV])

print("Ridge train error (CV): %f"%MSE(y_pred_train,train_Y))
print("Ridge test error (CV): %f"%MSE(y_pred_test,test_Y))

N_bootstrap=1000 #Some random large number :)
bootstrap_errs=np.zeros((len(alpha_vals),N_bootstrap))
indices_list=list(range(train_X.shape[0]))
for j in range(N_bootstrap):
    train_index=np.random.choice(indices_list,len(indices_list))
    test_index=list(set(indices_list)-set(train_index)) #Indices that are not train indices
    train_X_BS,test_X_BS=train_X[train_index],train_X[test_index]
    train_Y_BS,test_Y_BS=train_Y[train_index],train_Y[test_index]
    for i,alpha_val in enumerate(alpha_vals):
        regr = Ridge(alpha=alpha_val,fit_intercept=False)
        regr.fit(train_X_BS, train_Y_BS)
        y_predict_BS=regr.predict(test_X_BS)
        bootstrap_errs[i,j]=MSE(y_predict_BS,test_Y_BS)
err_lambda_BS=np.mean(bootstrap_errs,axis=1)
best_lambda=alpha_vals[np.argmin(err_lambda_BS)]
min_lambda_err_BS=np.argmin(err_lambda_BS)
regr = Ridge(alpha=best_lambda,fit_intercept=False)
regr.fit(train_X, train_Y)
y_pred_test = regr.predict(test_X)
y_pred_train = regr.predict(train_X)
print("Optimal Lambda (BS): %f"%alpha_vals[min_lambda_err_BS])
print("Ridge train error (BS): %f"%MSE(y_pred_train,train_Y))
print("Ridge test error (BS): %f"%MSE(y_pred_test,test_Y))
plt.plot(alpha_vals,err_lambda_BS,label="Boostrap error")
plt.plot(alpha_vals[min_lambda_err_BS],err_lambda_BS[min_lambda_err_BS],"o",color="red",label=r"Best $\lambda$")
plt.plot(alpha_vals,err_lambda_CV,label="Cross validation error")
plt.plot(alpha_vals[min_lambda_err_CV],err_lambda_CV[min_lambda_err_CV],"o",color="red")

#plt.plot(alpha_vals,err_lambda_test,label="Test error")
plt.xscale("log")
plt.xlabel(r"Penalty $\lambda$")
plt.ylabel("MSE")
plt.title("CV/BS errors for toxicity data set")
plt.legend()
plt.savefig("Ridge.pdf")
plt.show()
df=pd.read_csv("qsar_aquatic_toxicity.csv",delimiter=";",names=colnames)
df=sm.add_constant(df) #Add intercept

train, test = train_test_split(df, test_size=1/3,random_state=init_random) #Split into train and test data with random state $i$ for reproducibility
train_X=np.array(train.loc[:, train.columns != 'LC50'])
train_Y=np.array(train.loc[:, train.columns == 'LC50'])
test_X=np.array(test.loc[:, test.columns != 'LC50'])
test_Y=np.array(test.loc[:, test.columns == 'LC50'])


#Smoothing splines

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df=pd.read_csv("qsar_aquatic_toxicity.csv",delimiter=";",names=colnames)
#df=sm.add_constant(df) #Add intercept
train, test = train_test_split(df, test_size=1/3,random_state=init_random) #Split into train and test data with random state $i$ for reproducibility
scaler.fit(train[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]])
mean_train_Y=np.mean(train[["LC50"]])
train[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]]=scaler.transform(train[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]])
test[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]]=scaler.transform(test[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]])
train[["LC50"]]=train[["LC50"]]-mean_train_Y
test[["LC50"]]=test[["LC50"]]-mean_train_Y

train.to_csv("train_set.csv")
test.to_csv("test_set.csv")

import pygam
from pygam import LinearGAM, s, f
from pygam.datasets import wage

train_X=np.array(train.loc[:, train.columns != 'LC50'])
train_Y=np.array(train.loc[:, train.columns == 'LC50'])
test_X=np.array(test.loc[:, test.columns != 'LC50'])
test_Y=np.array(test.loc[:, test.columns == 'LC50'])
## model
n_splines=int(train_X.shape[0]/train_X.shape[1])+15 #+15 to others because we end up using less splines for C050, H040 and nN
formula = s(0, n_splines)
from collections import Counter
for i in range(1, train_X.shape[1]):
    if i==2 or i==6 or i==7:
        n_splines_i=len(Counter(train_X[:,i]).keys()) #Number of different values equal to number of knots
    else:
        n_splines_i=n_splines
    formula = formula + s(i, n_splines_i)
lams=np.array([1,1,1,1,1,1,1,1])
lams_spec=lams=np.array([1,0.01,1,10,1,1,1,1])
for lam in np.logspace(1,5,2):
    print("Lam:%f"%lam)
    gam = LinearGAM(formula,lam=lam*lams).fit(train_X,train_Y)
    #gam.summary()
    train_Y_predict=gam.predict(train_X)
    test_Y_predict=gam.predict(test_X)
    print("Train:")
    print(MSE(train_Y_predict,train_Y))
    print("Test:")
    print(MSE(test_Y_predict,test_Y))

## plotting
#print("Lam: Specific"%lam)
#gam = LinearGAM(formula,lam=lams_spec).fit(train_X,train_Y)
#train_Y_predict=gam.predict(train_X)
#test_Y_predict=gam.predict(test_X)
#print("Train:")
#print(MSE(train_Y_predict,train_Y))
#print("Test:")
#print(MSE(test_Y_predict,test_Y))
np.random.seed(seed=0) #reproducibility
lams=np.exp(10*(np.random.rand(500, 8)) )
gam = LinearGAM(formula).gridsearch(train_X,train_Y,lam=lams) #Find best parameters. Rather random search...
#gam.summary()
train_Y_predict=gam.predict(train_X)
test_Y_predict=gam.predict(test_X)
print("Optimal lambda")
print("Train:")
print(MSE(train_Y_predict,train_Y))
print("Test:")
print(MSE(test_Y_predict,test_Y))


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree


alpha_vals=np.logspace(-1.1,0,100)
num_cv=5 #CV folds
CV_errs=np.zeros((len(alpha_vals),num_cv))
kf = KFold(n_splits=num_cv)
i=0
j=0
for train_index,test_index in kf.split(train_X):
    i=0
    train_X_CV,test_X_CV=train_X[train_index],train_X[test_index]
    train_Y_CV,test_Y_CV=train_Y[train_index],train_Y[test_index]
    for alpha_val in alpha_vals:
        CV_tree = DecisionTreeRegressor(random_state=init_random,ccp_alpha=alpha_val)
        CV_tree.fit(train_X_CV, train_Y_CV)
        y_predict_CV=CV_tree.predict(test_X_CV)
        CV_errs[i,j]=MSE(y_predict_CV,test_Y_CV)
        i+=1
    j+=1
err_lambda=np.sum(CV_errs,axis=1)
best_lambda=alpha_vals[np.argmin(err_lambda)]
#print(best_lambda)
tree = DecisionTreeRegressor(random_state=init_random,ccp_alpha=best_lambda)
tree.fit(train_X, train_Y)
y_pred_test = tree.predict(test_X)
y_pred_train = tree.predict(train_X)

print("Tree train error (CV): %f"%MSE(y_pred_train,train_Y))
print("Tree test error (CV): %f"%MSE(y_pred_test,test_Y))
plot_tree(tree,feature_names=["intercept","TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040","LC50"])
plt.savefig("tree.pdf")
plt.show()

#Ignore from here on.
"""
train_X=train.loc[:, train.columns != 'LC50']
train_Y=train.loc[:, train.columns == 'LC50']
test_X=test.loc[:, test.columns != 'LC50']
test_Y=test.loc[:, test.columns == 'LC50']
from rpy2 import robjects
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

import rpy2.robjects.numpy2ri as n2r
rpy2.robjects.numpy2ri.activate()
rpy2.robjects.pandas2ri.activate()

n2r.activate()
r = ro.r
stats = importr('stats')
base = importr('base')
glmnet = importr('glmnet')
gam = importr('gam')

grdevices = importr('grDevices')
r.library("glmnet")
r.library("gam")
with localconverter(ro.default_converter + pandas2ri.converter):
  r_train = ro.conversion.py2rpy(train)
  r_test = ro.conversion.py2rpy(train)

model_gam = gam("LC50 ~TPSA",data=base.as_symbol(r_train)) #Ridge
#nr,nc = train_X.shape
#train_Xr = ro.r.matrix(train_X, nrow=nr, ncol=nc)
#nr,nc = test_X.shape

#test_Xr = ro.r.matrix(test_X, nrow=nr, ncol=nc)

#train_Yr=FloatVector(list(train_Y))
#lambda_vals=np.logspace(-3,3,100)
#ro.r.assign("train_X", train_Xr)
#ro.r.assign("test_X", test_Xr)


#model_lambda = r['cv.glmnet'](train_Xr, train_Yr,alpha=1) #Ridge
#lambda_min=model_lambda.rx2("lambda.min")
#print(lambda_min)
#y_predict=robjects.r.predict(model_lambda,test_Xr)
#print(MSE(y_predict,test_Y))
#grdevices.png(file="file.png", width=512, height=512)
#base.plot(model_lambda)
#grdevices.dev_off()


df=pd.read_csv("qsar_aquatic_toxicity.csv",delimiter=";",names=colnames)
df=sm.add_constant(df) #Add intercept
train, test = train_test_split(df, test_size=1/10,random_state=init_random) #Split into train and test data with random state $i$ for reproducibility
train_X=train.loc[:, train.columns != 'LC50']
train_Y=train.loc[:, train.columns == 'LC50']
test_X=test.loc[:, test.columns != 'LC50']
test_Y=test.loc[:, test.columns == 'LC50']


#GAM according to https://www.statsmodels.org/dev/gam.html
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
x_spline = train[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]]
degree=list(np.ones(8,dtype=int)*3)
dfv=list(np.ones(8,dtype=int)*30)
#dfv=list(np.ones(8,dtype=int)*30)

print(dfv)
print(degree)
bs = BSplines(x_spline, df=dfv, degree=degree)
alpha = np.ones(8)*1e4 #Some random number, really
gam_bs = GLMGam.from_formula('LC50 ~ TPSA + SAacc + H050 + MLOGP + RDCHI + GATS1p + nN + C040', data=train,smoother=bs, alpha=alpha)
res_bs = gam_bs.fit()
print(res_bs.summary())
y_pred=res_bs.predict(exog=test[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]],exog_smooth = np.asarray(test[["TPSA","SAacc","H050","MLOGP","RDCHI","GATS1p","nN","C040"]]))
print("GLM error")
print(MSE(y_pred,test_Y))

train_X=np.array(train_X)[:,1:]
test_X=np.array(test_X)[:,1:]

train_Y=np.array(train_Y)
test_Y=np.array(test_Y)
train_Y_mean=np.mean(train_Y)
train_Y=train_Y-train_Y_mean
test_Y=test_Y-train_Y_mean
scaler=StandardScaler()
scaler.fit(train_X) #"teach" scaler the correct scaling
train_X=scaler.transform(train_X)
test_X=scaler.transform(test_X)
"""
