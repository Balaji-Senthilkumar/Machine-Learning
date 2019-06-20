""" Prediction of Cancer death Rate using Linear Regression and 
used advanced boosting techniques like Gradient Boosting"""
=======================================================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import metrics
from sklearn import linear_model
from sklearn import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import explained_variance_score
from statistics import stdev
from sklearn.ensemble import AdaBoostRegressor
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
 
=========================================================================================================
 
"""Reading dataset from the file """
 
df=pd.read_csv(r'D:\\data.csv' , encoding = 'latin')
 
 
df.head()  
df.describe()  
df.info()
df.shape
 
#Duplicate column check
df.columns.duplicated()
 
#Duplicate row check
df['duplicate']=df.duplicated()
df.duplicate.value_counts()
df.drop('duplicate',axis=1,inplace=True)
 
===========================================================================================================
 """ Plots """
 fff
sns.distplot(df['incidenceRate'])
sns.distplot(df['medIncome'])
sns.distplot(df['povertyPercent'])
sns.distplot(df['PctHS25_Over'])
sns.distplot(df['PctBachDeg25_Over'])
sns.distplot(df['PctEmployed16_Over'])
sns.distplot(df['PctUnemployed16_Over'])
sns.distplot(df['PctPrivateCoverage'])
sns.distplot(df['PctPublicCoverage'])
 
sns.pairplot(df , x_vars= ['incidenceRate'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['medIncome'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['povertyPercent'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['PctHS25_Over'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['PctBachDeg25_Over'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['PctEmployed16_Over'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['PctUnemployed16_Over'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['PctPrivateCoverage'],y_vars= ['deathRate'],size = 5)
sns.pairplot(df , x_vars= ['PctPublicCoverage'],y_vars= ['deathRate'],size = 5)
 
===========================================================================================================
 
""" Finding missing values """
 
df.isnull().sum()
df.isnull().count().sort_values(ascending=False) ##Total number of records
(df.isnull().sum()/df.isnull().count()).sort_values(ascending=False) ##Percentage of null values
 
 
========================================================================================================
 
""" Treating missing Values """
 
sns.boxplot(df['PctPrivateCoverageAlone'])
sns.boxplot(df['PctEmployed16_Over'])
 
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1 #Interquartile range
 
df = df[~((df[['PctPrivateCoverageAlone']] < (Q1 - 1.5 * IQR)) | (df[['PctPrivateCoverageAlone']] > (Q3 + 1.5 * IQR))).any(axis=1)]
df = df[~((df[['PctEmployed16_Over']] < (Q1 - 1.5 * IQR)) | (df[['PctEmployed16_Over']] > (Q3 + 1.5 * IQR))).any(axis=1)]
 
 
df = df.drop('PctSomeCol18_24' , axis =1)
df['PctPrivateCoverageAlone'].fillna(df['PctPrivateCoverageAlone'].mean(),inplace= True)
df['PctEmployed16_Over'].fillna(df['PctEmployed16_Over'].mean(),inplace= True)
 
 
=======================================================================================================
 
""" New data frame with split value columns """
 
a = df["Geography"].str.split(",", n = 1, expand = True)
 
df['County'] = a[0]
df['State'] = a[1]
 
=========================================================================================================
 
 """ Dropping County column """
 
df = df.drop(['County'] ,axis =1)
df = df.drop(['Geography'] ,axis =1)
df = df.drop(['binnedInc'] ,axis =1)
 
=========================================================================================================
 
"""  Labelencoding """
 
labelencoder = LabelEncoder()
df["State"] = labelencoder.fit_transform(df["State"])
df["State"] = df["State"].astype('category')
 
========================================================================================================
 
""" Checking the correlation """
sns.set(rc={'figure.figsize':(25,25)})
sns.heatmap(df.corr(), annot=True, cmap='Reds')
 
b =df.corr()
print(b)
 
===========================================================================================================
 
 """ Standardization """
 
df1 = df.drop(['State' , 'deathRate'] , axis=1)
cols = df1.columns
 
 
for col in cols:
    col_zscore = col + '_zscore'
    df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)
   
df
 
========================================================================================================
 
""" VIF """
 
#Vif
x=df.drop(['deathRate','State'],axis=1)
x.info()
#vif
from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif_(x, thresh):
    variables = list(range(x.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(x.iloc[:, variables].values, ix)
               for ix in range(x.iloc[:, variables].shape[1])]
 
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + x.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True
    print('Remaining variables:')
    print(x.columns[variables])
    return x.iloc[:, variables]
 
calculate_vif_(x, thresh=10.0)
 
======================================================================================================
 
 """ removing variables with high correlation """
corr_matrix = df[remaining_feature_after_vif].corr().abs()
 
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
 
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.80)]
 
=====================================================================================================
 
 """ Removing Outliers """
 
df = df[~((df[['incidenceRate']] < -3 | (df[['incidenceRate']] > 3))).any(axis=1)]
df = df[~((df[['medIncome']] < -3 | (df[['medIncome']] > 3))).any(axis=1)]
df = df[~((df[['povertyPercent']] < -3 | (df[['povertyPercent']] > 3))).any(axis=1)]
df = df[~((df[['PctHS25_Over']] < -3 | (df[['PctHS25_Over']] > 3))).any(axis=1)]
df = df[~((df[['PctBachDeg25_Over']] < -3 | (df[['PctBachDeg25_Over']] > 3))).any(axis=1)]
df = df[~((df[['PctEmployed16_Over']] < -3 | (df[['PctEmployed16_Over']] > 3))).any(axis=1)]
df = df[~((df[['PctUnemployed16_Over']] < -3 | (df[['PctUnemployed16_Over']] > 3))).any(axis=1)]
df = df[~((df[['PctPrivateCoverage']] < -3 | (df[['PctPrivateCoverage']] > 3))).any(axis=1)]
df = df[~((df[['PctPublicCoverage']] < -3 | (df[['PctPublicCoverage']] > 3))).any(axis=1)]
 
==========================================================================================================

"""Recursive feature Elimination"""

from sklearn.feature_selection import RFE
lr = linear_model.LinearRegression()
 
 
# Initializing the RFE object, one of the most important arguments is the estimator, in this case is RandomForest
rfe = RFE(estimator=lr, n_features_to_select=12)
 
 
# Fit the origial dataset
rfe = rfe.fit(x_train, y_train)

===========================================================================================================
 
 """ Splitting the traind and test dataset """
 
train ,test = train_test_split(df ,  test_size=0.3,random_state = 123 )
x_train = train.drop(['deathRate'] , axis =1 )
x_test = test.drop(['deathRate'], axis =1)
y_train = train['deathRate']
y_test = test['deathRate']

x_tr = x_train[fff]
x_te = x_test[fff]
y_tr = y_train
y_te = y_test

 
===========================================================================================================
 
 """ Runnig the regression model """
 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
reg=regressor.fit(X_train_RFE, y_train)
 
 
#coefficient and intercept values
print('coefficient values',regressor.coef_)
print('intercept value',regressor.intercept_)
 
# Predicting the Test set results
r_y_pred = regressor.predict(X_test_RFE)
r_y_pred =pd.DataFrame(r_y_pred)
 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, r_y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, r_y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, r_y_pred)))
 
 
metrics.r2_score(y_test, r_y_pred)

sns.boxplot(data = df , x = ['deathRate'])
 
 
==========================================================================================================
 
 """ Model building using OLS Model """
 
incidenceRate+PctPublicCoverageAlone+PctHS25_Over+PctHS18_24+PctUnemployed16_Over
lm=smf.ols('deathRate~State+avgAnnCount+avgDeathsPerYear+incidenceRate+PctHS18_24+PctBachDeg25_Over+PctUnemployed16_Over+PctPublicCoverageAlone+PctWhite+PctBlack+PctOtherRace+PctMarriedHouseholds', train).fit()
 ### RFE
 
lm=smf.ols('deathRate~incidenceRate+PctBachDeg25_Over+PctPublicCoverage+PctPublicCoverageAlone+PctMarriedHouseholds', train).fit()
 
 
 
print(lm.summary())
 
o_pred = lm.predict(test)
 
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(test.deathRate, o_pred))
print('MSE:', metrics.mean_squared_error(test.deathRate, o_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test.deathRate, o_pred)))
 
fig, ax = plt.subplots()
ax.scatter(test.deathRate, o_pred, edgecolors=(0, 0, 0))
ax.plot([test.deathRate.min(), test.deathRate.max()], [test.deathRate.min(), test.deathRate.max()], 'k--', lw=4)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title("Ground Truth vs Predicted")
plt.show()
 
====================================================================================================
""" Gradient Boosting """
 
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
params = {'n_estimators': 500, 'max_depth': 10,
        'learning_rate': 0.01, 'loss': 'huber','alpha':0.95}
clf = GradientBoostingRegressor(**params).fit(x_tr, y_tr)
 
 
gb_predictions = clf.predict(x_te)
 
metrics.r2_score(test.deathRate, gb_predictions)
 
print('MAE:', metrics.mean_absolute_error(test.deathRate, gb_predictions))
print('MSE:', metrics.mean_squared_error(test.deathRate, gb_predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(test.deathRate, gb_predictions)))
 
=======================================================================================================
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
"""Recursive feature Elimination"""

from sklearn.feature_selection import RFE
lr = linear_model.LinearRegression()
 
 
# Initializing the RFE object, one of the most important arguments is the estimator, in this case is RandomForest
rfe = RFE(estimator=lr, n_features_to_select=12)
 
 
# Fit the origial dataset
rfe = rfe.fit(x_train, y_train)
 
print("Best features chosen by RFE: \n")
 
for i in x_train.columns[rfe.support_]:
    print(i)
 
##Testing
# Generating x_train and x_test based on the best features given by RFE
 
X_train_RFE = rfe.transform(x_train)
X_test_RFE = rfe.transform(x_test)  
 
# Fitting the Random Forest
RandForest_RFE = RandForest_RFE.fit(X_train_RFE, y_train)
 
# Making a prediction and calculting the accuracy
y_pred = RandForest_RFE.predict(X_test_RFE)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',accuracy)
        



































































































