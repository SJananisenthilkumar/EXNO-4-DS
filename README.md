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
# RESULT:
       # INCLUDE YOUR RESULT HERE
