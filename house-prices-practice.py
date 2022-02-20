# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# #  **References**
# * [COMPREHENSIVE DATA EXPLORATION WITH PYTHON](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python)
# * [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard)

# # House Prices - Advanced Regression Techniques
# ### Predict sales prices and practice feature engineering, RFs, and gradient boosting



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

df_train = pd.read_csv('/Users/chrisoh/Desktop/kaggle/house-prices-advanced-regression-techniques/train.csv')

df_train.columns

df_train

# # 1. Check the flavour of our dataset
#
# * **Variable** - variable name.
# * **Type** - numerical / categorical
# * **Segment** - building(physical characteristics of the building e.g. 'OverallQual') / space(space properties e.g. 'TotalBsmtSF') / location(information about the place where it is located e.g. 'Neighborhood')
# * **Expectation** - High / Medium / Low
# * **Conclusion** - high / Medium / Low
# * **Comments** - general comments
#
# To fill the 'Expectations', we should consider 
# - Do we think this variable when we are buying a house?
# - If so, how important would this variable be?
# - Is this information already described in any other variable?
#
# Then, we can rush into some scatter plots between those variables 'SalePrice', filling in the 'Conclusion' column which is just the correction of our expectations.
#
# Variables those play an important role in this problem.
# * OverallQual
# * YearBuilt
# * TotalBsmtSF
# * QrLivArea
#
# This is just a thinking process at the starting point of this project.
#

# # 2. Analysing 'SalePrice'
#
# Getting know about 'SalePrice'

df_train['SalePrice'].describe()

# Good - minimum price is larger than zero.

#histogram
sns.distplot(df_train['SalePrice'])

# * Deviate from the normal distribution.
# * Have appreciable postive skewness.
# * Show peakedness

#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())

# ### Relationship with numerical variables

#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# **Linear Relationship** can be seen in here.

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# Not a perfect linear relation.

# ### Relationship with categorical features

#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.subplots(figsize=(10, 6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)

# Evident

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.subplots(figsize=(16,8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#Using xticks, make x name visible.

# Even though it's not strong, new stuff tends to be more expensive.

# ### In summary
#
# * 'GrLivArea' and 'TotalBsmtSF' seem to be linearly related with 'SalePrice'. Positive relationships. In the case of 'TotalBsmtSF', the slop of the linear relationship is particularly high.
# * 'OverallQual' and 'YearBuilt' also seem to be related with 'SalePrice'. Especially stronger in the case of 'OverallQual'
#
# Like this process, we should put more attention to the choice of the right features and not the definition of complex relationships between all of the features.

# # 3. Keep calm and work smart

# So far, we just followed our intuition and analysed just the variables we thought were important.
# But we should do a more objective analysis.
#
# * Correlation matrix (heatmap style).
# * 'SalePrice' correlation matrix (zoomed heatmap style)
# * Scatter plots between the most correlated variables.

# ### Correlation matrix (heatmap style)

#correlation matrix
corrmat = df_train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)

# Best for the quick overview
#
# At first sight, two red colored squares can be seen. The first one refers to the 'TotalBsmtSF' and '1stFlrSF' variables. and the second one refers to the 'GarageX' variables. Correlations of both cases are so strong that it can indicate a situation of multicollinearity. If we think about these variables, we can conclude that they give almost the same information so multicollinearity actually occurs. Heatmaps are great to find this kind of situations and in problems dominated by feature selection.
#
# Another attention was 'SalePrice' correlations. 'GrLivArea', 'TotalBsmtSF', and 'OverallQual' our well-known factors can be seen. But also many other variables should be taken into account.

# ### 'SalePrice'correlation matrix (zoomed heatmap style)

#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size' : 10}, yticklabels=cols.values,
                 xticklabels=cols.values)
plt.show()

# reference https://hong-yp-ml-records.tistory.com/33

# According to map above, these are the variables most correlated with 'SalePrice'.
#
# * 'OverallQual', 'GrlivArea', and 'TotalBsmtSF' are strongly correlated with 'SalePrice'.
# * 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. However, as it was discussed in the last sub-point, the number of cars that fit into the garage is a consequence of the garage area. 'GarageCars' and 'GarageArea' are like twin brothers. It is impossible to distinguish them. So, just one of these variables will be needed in our analysis.(keep 'GarageCars' since its correlation with 'SalePrice' is higher).
# * 'TotalBsmtSF' and '1stFloor' also seem to be twin brothers.(keep 'TotalBsmtSF' just to say or first quess was right)
# * 'FullBath' ??
# * 'TotRmsAbvGrd' and 'GrLivArea', twin brothers again.
# * 'YearBuilt' is slightly correlated with 'SalePrice'

# ### Scatter plots between 'SalePrice' and correlated variables

#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();

# 왜 그래프가 다르게 나오지?

# ### Just copy
#
# Although we already know some of the main figures, this mega scatter plot gives us a reasonable idea about variables relationships.
#
# One of the figures we may find interesting is the one between 'TotalBsmtSF' and 'GrLiveArea'. In this figure we can see the dots drawing a linear line, which almost acts like a border. It totally makes sense that the majority of the dots stay below that line. Basement areas can be equal to the above ground living area, but it is not expected a basement area bigger than the above ground living area (unless you're trying to buy a bunker).
#
# The plot concerning 'SalePrice' and 'YearBuilt' can also make us think. In the bottom of the 'dots cloud', we see what almost appears to be a shy exponential function (be creative). We can also see this same tendency in the upper limit of the 'dots cloud' (be even more creative). Also, notice how the set of dots regarding the last years tend to stay above this limit (I just wanted to say that prices are increasing faster now).
#
# Ok, enough of Rorschach test for now. Let's move forward to what's missing: missing data!

# # 4.Missing data
#
# * How prevalent is the missing data?
# * Is missing data random or does it have a pattern?
#
# It is important to ensure that the missing data process is not biased and hiding other truths

#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

# Let's say that when more than 15% of the data is missing, we should delete the corresponding variable.
# -> Not gonna try any tricks to fill the missing data in these cases
#
# This means we should delete a set of variables (e.g. 'PoolQC', 'MiscFeature', 'Alley', etc).
# -> None on these variables seem to be very important, since most of them are not aspects we think about wehn buying a house. Moreover, we could say that variables like 'PoolQC', 'MiscFeature', and 'FireplaceQu' are strong candidates for outliers.
#
# Next, we can see that 'GarageX' variables have the same number of missing data. Since the most important information regarding garages is expressed by 'GarageCars' and considering that we are just talking about 5% of missing data, I'll delete the mentioned 'GarageX' variables. The same logic applies to 'BsmtX' variables.
#
# Regarding 'MasVnrArea' and 'MasVnrType', we can consider that these variables are not essential. Furthermore, they have a strong correlation with 'YearBuilt' and 'OverallQual' which are already considered. Thus, delete them.
#
# Finally, in 'Electrical' with just one missing observation, we'll delete this and keep the variable.
#
# In summary, to handle missing data, we'll delete all the variables with missing data, except the variable 'Electrical'. In 'Electrical' we'll just delete the observation with missing data.

#dealing with missing data
df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)
df_train.isnull().sum().max()

# # Outliers
#
# Outliers can affect our models and can be a valuable source of information, providing us insights about specific behaviours.
#
# Below will be quick analysis through the stadard deviation of 'SalePrice' and a set of scatter plots.

# 여기 어렵다 / 웬만해서 설명 그대로 씀

# ## Univariate analysis

# The primary concern here is to establish a threshold that defines an observation as an outlier. To do so, we'll standardize the data. In this context, data standardization means converting data values to have mean of 0 and a standard deviation of 1.

# https://deepinsight.tistory.com/165

# argsort-작은값부터 순서대로 데이터의 위치를 반환

#standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

# * Low range values are similar and not too far from 0.
# * High range values are far from 0 and the 7.smth values are really out of range.
#
# For now, we'll not consider any of these values as an outlier but we should be careful with those two 7.smth values.

# ## Bivariate analysis (이변량분석)

#bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

# * The two values with bigger 'GrLivArea' can be defined as a outliers and delete them
# * The two observations in the top of the pot are those 7.smth observations that we said we should be careful about. Although they look like special cases, they seem to be following jthe trend. So, we keep them.

#deleting points
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

#bivariate analysis saleprice/grlivarea
var = 'Total'
