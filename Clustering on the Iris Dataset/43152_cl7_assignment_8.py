# -*- coding: utf-8 -*-

#This notebook was developed as a Machine Learning lab assignment during my undergraduate degree.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import datasets

iris_data = datasets.load_iris()

iris_df = pd.DataFrame(data=np.c_[iris_data['data'],iris_data['target']],columns=iris_data['feature_names'] + ['target'])
iris_df

iris_df.shape

iris_df.info()

iris_df.target = iris_df.target.astype(int)

iris_df.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Target']
iris_df.columns

X = iris_df.iloc[:,:-1].values
#X

#Implementing elbow method for finding optimum value of n_clusters

inertia_scores = []

for k in range(1,9):
  model = KMeans(n_clusters=k,max_iter=400,random_state=42)
  model.fit(X)
  inertia_scores.append(model.inertia_)            

#inertia_scores

plt.plot(range(1,9),inertia_scores)
plt.xlabel('n_clusters')
plt.ylabel('Inertia')
plt.title('Elbow method for finding optimum "n_clusters"')
plt.show()

#Implementing KMeans, n_clusters=3

model = KMeans(n_clusters=3,max_iter=400,random_state=42)
model.fit(X)
ypred = model.predict(X)

ypred

#Visualization

plt.scatter(X[ypred==0,0],X[ypred==0,1],s=60,c='blue',label='Flower 1')
plt.scatter(X[ypred==1,0],X[ypred==1,1],s=60,c='red',label='Flower 2')
plt.scatter(X[ypred==2,0],X[ypred==2,1],s=60,c='green',label='Flower 3')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering Results')

plt.legend()

plt.scatter(X[ypred==0,2],X[ypred==0,3],s=60,c='blue',label='Flower 1')
plt.scatter(X[ypred==1,2],X[ypred==1,3],s=60,c='red',label='Flower 2')
plt.scatter(X[ypred==2,2],X[ypred==2,3],s=60,c='green',label='Flower 3')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('K-Means Clustering Results')

plt.legend()

#Agglomerative clustering

model = AgglomerativeClustering(n_clusters=3)
ypred = model.fit_predict(X)

ypred

#Visualization

plt.scatter(X[ypred==0,0],X[ypred==0,1],s=60,c='blue',label='Flower 1')
plt.scatter(X[ypred==1,0],X[ypred==1,1],s=60,c='red',label='Flower 2')
plt.scatter(X[ypred==2,0],X[ypred==2,1],s=60,c='green',label='Flower 3')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Agglomerative Clustering Results')
plt.legend()

plt.scatter(X[ypred==0,2],X[ypred==0,3],s=60,c='blue',label='Flower 1')
plt.scatter(X[ypred==1,2],X[ypred==1,3],s=60,c='red',label='Flower 2')
plt.scatter(X[ypred==2,2],X[ypred==2,3],s=60,c='green',label='Flower 3')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Agglomerative Clustering Results')
plt.legend()

