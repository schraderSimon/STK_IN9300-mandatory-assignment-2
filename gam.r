library(gam)
library(gamair)
#library(mgcv)
train <- read.csv(file = 'projects/STK-IN9300/oblig_2/train_set.csv')
test <- read.csv(file = 'projects/STK-IN9300/oblig_2/test_set.csv')

print(class(dfval))
fit=gam(LC50 ~TPSA+SAacc+H050+MLOGP+RDCHI+GATS1p+nN+C040, data = train)
#model_gam=gam(LC50 ~s(TPSA, df=5)+s(SAacc, bs = "cr")+s(H050, df=2)+s(MLOGP, bs = "cr")
#              +s(RDCHI, bs = "cr")+s(GATS1p, bs = "cr")+s(nN, df=3)+s(C040, df=2), data = train)
fit<-gam(LC50 ~s(TPSA,df=30)+s(SAacc,df=30)+s(H050,df=30)+s(MLOGP,df=30)+s(RDCHI,df=30)+s(GATS1p,df=30)+s(nN,df=30)+s(C040,df=30),data=train)
y_train_predict<-predict.Gam(fit,train)
y_test_predict<-predict.Gam(fit,test)
mean((y_train_predict-train$LC50)^2)
mean((y_test_predict-test$LC50)^2)

summary(fit)

plot(fit)# 
