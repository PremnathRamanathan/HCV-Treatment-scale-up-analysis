import sys
import pandas as pd
import matplotlib.pyplot as plt

print('python version' + sys.version)
print('pandas version' + pd.__version__)

names = ['Prem', 'John', 'Sue', 'Mary']
births = [167, 965, 456, 665]

BabyDataSet = list(zip(names, births))
print('\n items in the list')
for i in range(len(BabyDataSet)):
    print(BabyDataSet.__getitem__(i))

Location= r'C:\Users\PETM_PC1\Desktop\dummy.csv'

#write table to csv
df = pd.DataFrame(data=BabyDataSet, columns=['Names', 'Births'])
print('\n Data table \n' )
print(df)
df.to_csv(Location, index=False, header=False)

#read csv
df = pd.read_csv(Location,names=['Names','Births'])
print(df)

#Verify the write
flag = df.empty
if flag == False:
    print('write to csv successful')
elif flag == True:
    print('write to csv failed')

import os
os.remove(Location)

#data types of the columns
print(df.dtypes)
#method 1
Sorted =df.sort_values(['Births'],ascending=False)
print('\n')
print(Sorted.head(1))
#method 2
Sort = df['Births'].max()
print('\n')
print(Sort)

# to plot a graph
graph = df['Births'].plot(figsize=(15,5));
Sorted.head(10)

MaxValue = df['Births'].max()
MaxName = df['Names'][df['Births']==df['Births'].max()].values
text = str(MaxValue)+"-"+MaxName
plt.annotate(text,xy=(1,MaxValue),xytext=(8,0),xycoords=('axes fraction','data'), textcoords=('offset points'))
print(graph)

print("The most popular name")
print(Sorted.head(1))

print(df['Names'].unique())

print(df['Births'].plot().bar(1 , MaxValue))
