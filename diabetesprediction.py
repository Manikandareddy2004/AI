#Artificial Intelligence
#importing the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
#importing the dataset
df=pd.read_csv('diabetes.csv')
#Exploratory data analysis
print(df.head())
print(df.shape)
print(df.info)
print(df.describe)
#dividing dataset into X and Y
X=df.drop('Outcome',axis=1)
Y=df['Outcome']
print(X)
print(Y)
#checking for null values
print(df.isnull().sum())
df.dropna(inplace=True)
print(df.shape)
#spliting dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
#creating RandomForest model
rf_model=RandomForestClassifier(n_estimators=100,random_state=42)
rf_model.fit(X_train,y_train)
y_pred=rf_model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
