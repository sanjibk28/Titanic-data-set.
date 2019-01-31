#Titanic data set trained to find survival of a passenger using Random forest

import pandas as pd

#TRAIN FILE WORK

df = pd.read_csv("train.csv")
df.drop(["Name","Cabin","Ticket"], axis=1,inplace=True)
df.index=df.PassengerId
df.drop(["PassengerId"], axis=1,inplace=True)
df.Age.fillna(value=29.7,inplace=True)#replacing nan values with the mean value of 29.7

#TEST FILE WORK

df1=pd.read_csv("test.csv")
df1.drop(["Name","Cabin","Ticket"], axis=1,inplace=True)
df1.index=df1.PassengerId
df1.drop(["PassengerId"], axis=1,inplace=True)
df1.Age.fillna(value=30.27,inplace=True)



x_train=pd.get_dummies(df.loc[:,"Pclass":"Embarked"])
y_train=df.loc[:,"Survived"]
x_test=pd.get_dummies(df1.loc[:,:]) 
x_test.Fare.fillna(value=x_test.Fare.mean(),inplace=True)#replacing nan values with the mean value of 29.7

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred1=rfc.predict(x_test)
print(y_pred1)



result=pd.DataFrame(data=y_pred1,columns= ["Survived"])
result.index=df1.index

result.to_csv("final.csv")



