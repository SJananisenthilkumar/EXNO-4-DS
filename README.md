# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
**Feature Scaling**
```
import pandas as pd
from scipy import stats
import numpy as np
```
```
import pandas as pd
df=pd.read_csv("/content/bmi.csv")
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/376dd65a-8a46-4e5a-aa06-89db3bf431f6)
```
df.head()
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/33268cf2-cc37-41c2-bad9-cbd2601a933c)
```
import numpy as np
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/ff3d3a86-1323-4af3-995f-dd5681da7c86)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/d1399553-bb11-495a-99e5-622ff96c20b1)
```
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/8172543a-3361-49da-97be-84430ddb9f61)
```
from sklearn.preprocessing import Normalizer
Scaler=Normalizer
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/68ce3578-0f70-48f2-b9ea-1a678840c015)
```
df=pd.read_csv("/content/bmi.csv")
```
```
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/1aedf04d-036c-4544-a2d5-bb05ed2663cd)
**Feature Selection**
```
import pandas as pd
import numpy as np
import seaborn as sns
```
```
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
```
```
data = pd.read_csv("/content/income(1) (1).csv")
data
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/f0b8cd52-62b8-416d-b69d-c8fe069d136d)
```
data.isnull().sum()
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/bf3e8492-5293-468a-9e2c-4f800399cf12)
```
missing = data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/ffd89db6-899b-436e-9a73-4380d04f0798)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/0c6f42de-7403-42ad-941c-b16d5cc00663)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/51a5df28-e2cd-418c-839e-c67d782ca592)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/8f759201-555b-45d9-ba1a-d9fd3afd2cbe)
```
data2
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/ec3cb4e9-473d-412a-ba94-a7ff55948e77)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/8e8a730c-9985-411a-8b74-4858c49b4d0c)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/2ef2c222-a03c-4e7a-809b-2b287ef9e77e)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/43ef60b4-3795-422e-873a-f97e823a5763)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/41ac20e1-b8ca-46d4-baeb-0626dbfe8201)
```
x = new_data[features].values
print(x)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/750457a2-014f-47bb-baa3-4b5a779b1de2)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state = 0)
KNN_classifier = KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x, train_y)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/d674e254-bc50-4f83-a3f4-6e1b8a9579a6)
```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/9bca659d-a62c-4692-a416-b1dd938d4109)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/3febda2d-94ca-4140-a118-453ac440632a)
```
print( 'Misclassified samples: %d' % (test_y != prediction).sum())
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/655cd5ee-0a22-4869-aced-fd8aa6f4503a)
```
data.shape
```
![image](https://github.com/SJananisenthilkumar/EXNO-4-DS/assets/144871139/4232876f-f136-4cc7-945c-9d5237958a8e)


# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
