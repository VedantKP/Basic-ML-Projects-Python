# -*- coding: utf-8 -*-

#This notebook was developed as a Machine Learning lab assignment during my undergraduate degree.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split,KFold,StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import plot_confusion_matrix,roc_curve,auc
from sklearn.model_selection import cross_val_score,GridSearchCV,RandomizedSearchCV
from google.colab import files
import io

mushroom_data = files.upload()

#train_df = pd.read_csv('train.csv')
df = pd.read_csv(io.BytesIO(mushroom_data['mushrooms.csv']))

df.head()

df.shape

df.info()

"""**Attribute Information**:

1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
4. bruises?: bruises=t,no=f
5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
6. gill-attachment: attached=a,descending=d,free=f,notched=n
7. gill-spacing: close=c,crowded=w,distant=d
8. gill-size: broad=b,narrow=n
9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
10. stalk-shape: enlarging=e,tapering=t
11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
16. veil-type: partial=p,universal=u
17. veil-color: brown=n,orange=o,white=w,yellow=y
18. ring-number: none=n,one=o,two=t
19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
"""

df['class'].unique() #target variable

encoder = LabelEncoder()
for col in df.columns:
  df[col] = encoder.fit_transform(df[col])

df.head()

df['class'].unique()

sns.countplot(x=df['class'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of classes')
plt.show()

sns.heatmap(df.corr(),cmap='YlOrRd')

df['veil-type'].unique()

#Dropping veil-type
df.drop(columns=['veil-type'],inplace=True)

corr_threshold = 0.2

corr = pd.DataFrame(df.corr()['class'])
corr['abs'] = np.abs(corr['class'])
corr = corr.sort_values(by='abs',ascending=False).drop('abs',axis=1).dropna().reset_index()
corr = corr.rename(columns={'index':'feature','class':'corr'}).loc[1:]

low_corr_features = list(corr[np.abs(corr['corr'])<=corr_threshold]['feature'])
reduced_df = df.drop(low_corr_features,axis=1)
reduced_df.shape

reduced_df

X = reduced_df.drop(columns=['class'])
y = reduced_df['class']

X.shape

scaler = StandardScaler()
X = scaler.fit_transform(X)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)
print('X_train.shape: {}\nX_test.shape: {}\ny_train.shape: {}\ny_test.shape: {}'.format(
    X_train.shape,X_test.shape,y_train.shape,y_test.shape
))

#Model
model = SVC(C=1.0,kernel='rbf')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Results
print('Train Accuracy: {}'.format(model.score(X_train,y_train)))
print('Test Accuracy: {}'.format(accuracy_score(y_test,y_pred)))

print(classification_report(y_test,y_pred))

kfold = KFold(n_splits=7,shuffle=True,random_state=42)

kfold_score = cross_val_score(model,X,y,cv=kfold)
print('7-Fold cross validation results =>\n\nAll scores: {}\nMean: {}\nStd. Deviation: {}'.format(
    kfold_score,np.mean(kfold_score),np.std(kfold_score)
))

print(confusion_matrix(y_test,y_pred))

plot_confusion_matrix(model,X_test,y_test)
plt.show()

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

plt.figure(figsize=(5,5))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

C = [0.01,0.1,1,10]
kernel = ['rbf','linear','poly']
all_results = pd.DataFrame(data={'C':{},'kernel':{},'Train Accuracy':{},'Test Accuracy':{}})
all_results

for c in C:
  for k in kernel:
    svm = SVC(C=c,kernel=k)
    svm.fit(X_train,y_train)
    y_pred = svm.predict(X_test)
    train_acc = svm.score(X_train,y_train)
    test_acc = accuracy_score(y_test,y_pred)
    data_df = pd.DataFrame(data={'C':[c],'kernel':[k],'Train Accuracy':[train_acc],'Test Accuracy':[test_acc]})
    all_results = all_results.append(data_df,ignore_index=True)

all_results.sort_values(by='Test Accuracy',ascending=False)