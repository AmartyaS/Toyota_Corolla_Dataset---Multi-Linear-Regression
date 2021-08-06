# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 16:05:54 2021

@author: amart
"""
from math import sqrt
from sklearn.metrics import mean_squared_error 
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sb

dat = pd.read_csv("F:\Data Science Assignments\Python-Assignment\Multi-Linear Regression\ToyotaCorolla.csv",encoding='latin1')
new=dat.loc[:, {'Price','Age_08_04','KM','cc','HP','Doors','Gears','Quarterly_Tax','Weight'}]
sb.pairplot(new)
new.corr()

model1=smf.ols('Price~Age_08_04+KM+cc+HP+Doors+Gears+Quarterly_Tax+Weight',data=new).fit()
model1.summary()             #Rsquare =0.864  and Adj Rsquare =0.863
pred1=model1.predict(new)                                           
rmse=sqrt(mean_squared_error(new.Price,pred1))
rmse                         #1338.2584236201515
                                             
cc_model=smf.ols('Price~cc',data=new).fit()
cc_model.summary()                               
doors_model=smf.ols('Price~Doors',data=new).fit()
doors_model.summary()                          
tot_model=smf.ols('Price~cc+Doors',data=new).fit()
tot_model.summary()                       
                                             
sm.graphics.influence_plot(model1)
                                          
new_1=new.drop(new.index[[80,221]],axis=0)
model2=smf.ols('Price~Age_08_04+KM+cc+HP+Doors+Gears+Quarterly_Tax+Weight',data=new_1).fit()
model2.summary()             #Rsquare =0.878    and  Adj Rsquare = 0.877
pred2=model2.predict(new_1)
sm.graphics.influence_plot(model2)
sm.graphics.plot_partregress_grid(model2)
rmse=sqrt(mean_squared_error(new_1.Price,pred2))
rmse                         #1265.720000354872
tab=new_1.corr()
tab=pd.DataFrame(tab)
tab

rsq_Age=smf.ols('Age_08_04~KM+cc+HP+Doors+Gears+Quarterly_Tax+Weight',data=new_1).fit().rsquared
vif_Age=(1/(1-rsq_Age))
rsq_km=smf.ols('KM~Age_08_04+cc+HP+Doors+Gears+Quarterly_Tax+Weight',data=new_1).fit().rsquared
vif_km=(1/(1-rsq_km))
rsq_cc=smf.ols('cc~KM+Age_08_04+HP+Doors+Gears+Quarterly_Tax+Weight',data=new_1).fit().rsquared
vif_cc=(1/(1-rsq_cc))
rsq_HP=smf.ols('HP~KM+cc+Age_08_04+Doors+Gears+Quarterly_Tax+Weight',data=new_1).fit().rsquared
vif_HP=(1/(1-rsq_HP))
rsq_Doors=smf.ols('Doors~KM+cc+HP+Age_08_04+Gears+Quarterly_Tax+Weight',data=new_1).fit().rsquared
vif_Doors=(1/(1-rsq_Doors))
rsq_Gears=smf.ols('Gears~KM+cc+HP+Doors+Age_08_04+Quarterly_Tax+Weight',data=new_1).fit().rsquared
vif_Gears=(1/(1-rsq_Gears))
rsq_Quarter=smf.ols('Quarterly_Tax~KM+cc+HP+Doors+Gears+Age_08_04+Weight',data=new_1).fit().rsquared
vif_Quarter=(1/(1-rsq_Quarter))
rsq_Weight=smf.ols('Weight~KM+cc+HP+Doors+Gears+Quarterly_Tax+Age_08_04',data=new_1).fit().rsquared
vif_Weight=(1/(1-rsq_Weight))

ds={'Index':['Age','KM','CC','HP','Doors','Gears','Quarterly_Tax','Weight'],'VIF':[vif_Age,vif_km,vif_cc,vif_HP,vif_Doors,vif_Gears,vif_Quarter,vif_Weight]}
ds=pd.DataFrame(ds)          #Weight has the highest VIF=3.19

model3=smf.ols('Price~Age_08_04+KM+cc+HP+Doors+Gears+Quarterly_Tax',data=new_1).fit()
model3.summary()             #Rsquare =0.840    and  Adj Rsquare = 0.839
pred3=model3.predict(new_1)
rmse=sqrt(mean_squared_error(new_1.Price,pred3))
rmse                         #1450.5000058805758

#Model2 is best as rmse value is less and has better Rsq and adj.rsq values.