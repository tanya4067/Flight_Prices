import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_excel(r'C:\Users\tanya\Downloads\flight ticket data science\Data_Train.xlsx')
#print(df.head())
df.dropna(inplace=True)

def change_into_datetime(col):
    df[col]=pd.to_datetime(df[col])

for i in ['Date_of_Journey','Dep_Time','Arrival_Time']:
    change_into_datetime(i)

df['day']=df['Date_of_Journey'].dt.day
df['month']=df['Date_of_Journey'].dt.month
df.drop('Date_of_Journey',axis=1,inplace=True)

def extract_hour(df,col):
    df[col+'_hour']=df[col].dt.hour
def extract_minutes(df,col):
    df[col+'_minute']=df[col].dt.minute   
def drop_column(df,col):
    df.drop(col,axis=1,inplace=True)
    
extract_hour(df,'Dep_Time')
extract_hour(df,'Dep_Time')
drop_column(df,'Dep_Time')

extract_hour(df,'Arrival_Time')
extract_hour(df,'Arrival_Time')
drop_column(df,'Arrival_Time')


duration=list(df['Duration'])

for i in range(len(duration)):
    if(len(duration[i].split(' '))==2):
        pass
    else:
        if 'h' in duration[i]:
            duration[i]=duration[i]+' 0m'
        else:
            duration[i]='0h '+duration[i]
r=[]
for i in duration:
    i=i.split(' ')
    hour=int(i[0][0:-1])
    minutes=int(i[1][0:-1])
    t=(hour*60)+minutes
    r.append(t)
df['Duration']= r


#plt.figure(figsize=(15,5))
#sns.boxplot(x="Total_Stops",y="Price",data=df.sort_values('Price',ascending=False))


stops=list(df['Total_Stops'])
for i in range(len(stops)):
    if(stops[i]=='non-stop'):
        stops[i]=0
    else:
        t=stops[i]
        stops[i]=t[0]
df['Total_Stops']=(stops)
df['Total_Stops']=df['Total_Stops'].astype(int)
#print(df['Total_Stops'])

#print(df['Airline'].value_counts())
#plt.figure(figsize=(15,5))
#sns.boxplot(x="Airline",y="Price",data=df.sort_values('Price',ascending=False))


#One - Hot -Encoding

Airline=pd.get_dummies(df['Airline'],drop_first=True)
#print(Airline.head())

#print(df['Source'].value_counts())
#plt.figure(figsize=(15,5))
#sns.boxplot(x="Source",y="Price",data=df.sort_values('Price',ascending=False))
Source=pd.get_dummies(df['Source'],drop_first=True)
#print(Source)


#print(df['Destination'].value_counts())
#plt.figure(figsize=(15,5))
#sns.boxplot(x="Destination",y="Price",data=df.sort_values('Price',ascending=False))
Destination=pd.get_dummies(df['Destination'],drop_first=True)
#print(Destination)



df['Route 1']=(df['Route'].str.split('*').str[0])
df['Route 2']=(df['Route'].str.split('*').str[1])
df['Route 3']=(df['Route'].str.split('*').str[2]).astype(str)
df['Route 4']=(df['Route'].str.split('*').str[3]).astype(str)
df['Route 5']=(df['Route'].str.split('*').str[4]).astype(str)
drop_column(df,'Route')




from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

for i in ['Route 1','Route 2','Route 3','Route 4','Route 5']:
    df[i]=encoder.fit_transform(df[i])

drop_column(df,'Additional_Info')

df=pd.concat([Airline,Source,Destination,df],axis=1)

drop_column(df,'Source')
drop_column(df,'Destination')
drop_column(df,'Airline')
drop_column(df,'Trujet')

def plot(df,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)

y=df['Price']
drop_column(df,'Price')
X=df
print('*************',X.columns)





df1=pd.read_excel(r'C:\Users\tanya\Downloads\flight ticket data science\Test_set.xlsx')
#print(df.head())
df1.dropna(inplace=True)

def change_into_datetime(col):
    df1[col]=pd.to_datetime(df1[col])

for i in ['Date_of_Journey','Dep_Time','Arrival_Time']:
    change_into_datetime(i)

df1['day']=df1['Date_of_Journey'].dt.day
df1['month']=df1['Date_of_Journey'].dt.month
df1.drop('Date_of_Journey',axis=1,inplace=True)

def extract_hour(df1,col):
    df1[col+'_hour']=df1[col].dt.hour
def extract_minutes(df,col):
    df1[col+'_minute']=df1[col].dt.minute   
def drop_column(df,col):
    df1.drop(col,axis=1,inplace=True)
    
extract_hour(df1,'Dep_Time')
extract_hour(df1,'Dep_Time')
drop_column(df1,'Dep_Time')

extract_hour(df1,'Arrival_Time')
extract_hour(df1,'Arrival_Time')
drop_column(df1,'Arrival_Time')


duration=list(df1['Duration'])

for i in range(len(duration)):
    if(len(duration[i].split(' '))==2):
        pass
    else:
        if 'h' in duration[i]:
            duration[i]=duration[i]+' 0m'
        else:
            duration[i]='0h '+duration[i]
r=[]
for i in duration:
    i=i.split(' ')
    hour=int(i[0][0:-1])
    minutes=int(i[1][0:-1])
    t=(hour*60)+minutes
    r.append(t)
df1['Duration']= r


#plt.figure(figsize=(15,5))
#sns.boxplot(x="Total_Stops",y="Price",data=df.sort_values('Price',ascending=False))


stops=list(df1['Total_Stops'])
for i in range(len(stops)):
    if(stops[i]=='non-stop'):
        stops[i]=0
    else:
        t=stops[i]
        stops[i]=t[0]
df1['Total_Stops']=(stops)
df1['Total_Stops']=df1['Total_Stops'].astype(int)
#print(df['Total_Stops'])

#print(df['Airline'].value_counts())
#plt.figure(figsize=(15,5))
#sns.boxplot(x="Airline",y="Price",data=df.sort_values('Price',ascending=False))


#One - Hot -Encoding

Airline=pd.get_dummies(df1['Airline'],drop_first=True)
#print(Airline.head())

#print(df['Source'].value_counts())
#plt.figure(figsize=(15,5))
#sns.boxplot(x="Source",y="Price",data=df.sort_values('Price',ascending=False))
Source=pd.get_dummies(df1['Source'],drop_first=True)
#print(Source)


#print(df['Destination'].value_counts())
#plt.figure(figsize=(15,5))
#sns.boxplot(x="Destination",y="Price",data=df.sort_values('Price',ascending=False))
Destination=pd.get_dummies(df1['Destination'],drop_first=True)
#print(Destination)



df1['Route 1']=(df1['Route'].str.split('*').str[0])
df1['Route 2']=(df1['Route'].str.split('*').str[1])
df1['Route 3']=(df1['Route'].str.split('*').str[2]).astype(str)
df1['Route 4']=(df1['Route'].str.split('*').str[3]).astype(str)
df1['Route 5']=(df1['Route'].str.split('*').str[4]).astype(str)
drop_column(df1,'Route')




from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()

for i in ['Route 1','Route 2','Route 3','Route 4','Route 5']:
    df1[i]=encoder.fit_transform(df1[i])

drop_column(df1,'Additional_Info')

df1=pd.concat([Airline,Source,Destination,df1],axis=1)

drop_column(df1,'Source')
drop_column(df1,'Destination')
drop_column(df1,'Airline')

def plot(df1,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(df1[col],ax=ax1)
    sns.boxplot(df1[col],ax=ax2)
X1=df1








from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor 

mutual_info_classif(X,y)
imp=pd.DataFrame(mutual_info_classif(X,y),index=X.columns)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
def predict(ml_model):
    model=ml_model.fit(X_train,y_train)
    #print('Trainig score:{}',format(model.score(X_train,y_train)))
    predictions=model.predict(X_test)
    final_predictions=model.predict(X1)
    print('**************************************')
    #print('final_predictions:',final_predictions)
    print('**************************************')
    
    print(list(final_predictions))
    
    
    
    
    
    
    
    
    
    
    
    #print('Predictions array:',predictions)
    #print('Length of predictions array:',len(predictions))
    r2_score=metrics.r2_score(y_test,predictions)
    #print('r2 score is ',r2_score)
    mae=metrics.mean_absolute_error(y_test,predictions)
    #print('Mae:',mae)
    rmse=metrics.mean_absolute_error(y_test,predictions)
    #print('rmse:',rmse)
    sns.distplot(y_test-predictions)


#HERE USE WHICHEVER MODEL REQUIRED SIMPLY IMPORT AND CHECK 
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

print('Random Forest Regressor')
predict(RandomForestRegressor()) 








#print('Linear Regression')
#predict(LinearRegression())
#print('KNN Regressor')
#predict(KNeighborsRegressor())
#print('Decesion Tree Classifier')
#predict(DecisionTreeRegressor())


#FROM THIS WE FOUND THAT RANDOM FOREST REGRESSOR IS MOST ACCURATE , SO WE WILL USE THIS
# Now perform hypertuning of the model
#Using cross-validation
























    
    