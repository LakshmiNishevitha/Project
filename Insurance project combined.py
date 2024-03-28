#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data=pd.read_csv("C:\\Users\\laksh\\Downloads\\archive\\TravelInsurancePrediction.csv")
data.head()


# In[4]:


data.drop(columns=["Unnamed: 0"], inplace=True)


# In[5]:


data.isnull().sum()


# In[6]:


data.duplicated().sum()


# In[7]:


data.drop_duplicates(inplace=True)


# In[8]:


data['TravelInsurance'].value_counts()


# In[9]:


data["TravelInsurance"] = data["TravelInsurance"].map({0: "Not Purchased", 1: "Purchased"})


# In[10]:


data['TravelInsurance'].value_counts()


# In[11]:


plt.figure(figsize=(10, 6))
plt.hist([data[data['TravelInsurance'] =='Purchased']['Age'], 
          data[data['TravelInsurance'] =='Not Purchased']['Age']],
          bins=11,color=['blue', 'orange'], label=['Purchased', 'Not Purchased'])
plt.title('Factors Affecting Purchase of Travel Insurance: Age')
plt.xlabel('Age')
plt.ylabel('count')
plt.legend()
plt.show()


# In[14]:


plt.figure(figsize=(10, 6))
plt.hist([data[data['TravelInsurance'] == 'Not Purchased']['AnnualIncome'], 
          data[data['TravelInsurance'] == 'Purchased']['AnnualIncome']],
          bins=11, color=['orange', 'blue'], label=['Not Purchased', 'Purchased'])
plt.title('Factors Affecting Purchase of Travel Insurance: Income')
plt.xlabel('Annual Income')
plt.ylabel('count')
plt.legend()
plt.show()


# In[15]:


plt.figure(figsize=(10, 6))
plt.hist([data[data['TravelInsurance'] == 'Not Purchased']['EverTravelledAbroad'], 
          data[data['TravelInsurance'] == 'Purchased']['EverTravelledAbroad']],
         bins=11, color=['orange', 'blue'], label=['Not Purchased', 'Purchased'])
plt.title('Factors Affecting Purchase of Travel Insurance:Ever Travelled Abroad')
plt.xlabel('EverTravelledAbroad')
plt.ylabel('count')
plt.legend()
plt.show()


# In[16]:


plt.figure(figsize=(10, 6))
plt.hist([data[data['TravelInsurance'] == 'Not Purchased']['Employment Type'], 
          data[data['TravelInsurance'] == 'Purchased']['Employment Type']],
          bins=11, color=['orange', 'blue'], label=['Not Purchased', 'Purchased'])
plt.title('Factors Affecting Purchase of Travel Insurance: Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('count')
plt.legend()
plt.show()


# In[17]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2


# In[18]:


X = data.drop('TravelInsurance', axis=1)
y = data['TravelInsurance']


# In[19]:


# Identify categorical columns for one-hot encoding
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()


# In[20]:


X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)


# In[21]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# # Feature Selection

# In[22]:


X_train_selected = SelectKBest(chi2, k=5).fit_transform(X_train, y_train)
X_test_selected = SelectKBest(chi2, k=5).fit_transform(X_test, y_test)


# In[23]:


dt = DecisionTreeClassifier()
gd = GradientBoostingClassifier()
lr = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC()
knn = KNeighborsClassifier()
nb = GaussianNB()


# In[24]:


# Defining k-fold cross-validation
from sklearn.model_selection import KFold,cross_val_score
kf = KFold(n_splits=5)


# In[25]:


# Decision Tree Model
dt.fit(X_train_selected, y_train)
y_pred_dt = dt.predict(X_test_selected)
print("Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_dt))


# In[26]:


dt_scores = cross_val_score(dt, X_encoded, y, scoring='accuracy', cv=kf)
print(dt_scores.mean())


# In[27]:


# Gradient Boosting Model
gd.fit(X_train_selected, y_train)
y_pred_gd = gd.predict(X_test_selected)
print("Classification Report (Gradient Boosting):")
print(classification_report(y_test, y_pred_gd))


# In[28]:


gd_scores = cross_val_score(gd, X_encoded, y, scoring='accuracy', cv=kf)
print(gd_scores.mean())


# In[29]:


# Logistic Regression Model
lr.fit(X_train_selected, y_train)
y_pred_lr = lr.predict(X_test_selected)
print("Classification Report (Logistic Regression):")
print(classification_report(y_test, y_pred_lr))


# In[30]:


lr_scores = cross_val_score(lr, X_encoded, y, scoring='accuracy', cv=kf)
print( lr_scores.mean())


# In[31]:


#  Random Forest Model
rf.fit(X_train_selected, y_train)
y_pred_rf = rf.predict(X_test_selected)
print("Classification Report (Random Forest):")
print(classification_report(y_test, y_pred_rf))


# In[32]:


rf_scores = cross_val_score(rf, X_encoded, y,scoring='accuracy', cv=kf)
print(rf_scores.mean())


# In[33]:


#  SVM Model
svm.fit(X_train_selected, y_train)
y_pred_svm = svm.predict(X_test_selected)
print("Classification Report (Support Vector Machine):")
print(classification_report(y_test, y_pred_svm))


# In[34]:


svm_scores = cross_val_score(svm, X_encoded, y, cv=kf)
print(svm_scores.mean())


# In[35]:


#  KNeighborsClassifier
knn.fit(X_train_selected, y_train)
y_pred_knn = knn.predict(X_test_selected)
print("Classification Report (KNeighborsClassifier):")
print(classification_report(y_test, y_pred_knn))


# In[36]:


knn_scores = cross_val_score(knn, X_encoded, y, cv=kf)
print(knn_scores.mean())


# In[37]:


# GaussianNB
nb.fit(X_train_selected, y_train)
y_pred_nb = nb.predict(X_test_selected)
print("Classification Report (GaussianNB):")
print(classification_report(y_test, y_pred_nb))


# In[38]:


nb_scores = cross_val_score(nb, X_encoded, y, cv=kf)
print(nb_scores.mean())


# In[ ]:


'''Gradient Boosting algorithm (gd)-- mean accuracy of 76.70%
'''


# In[ ]:


# Fine-Tune Hyperparameters for Gradient Boosting


# In[39]:


#parameter grid for Gradient Boosting

para = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
};


# In[40]:


grid_search_gb = GridSearchCV(gd, para, cv=kf, scoring='accuracy')
grid_search_gb.fit(X_encoded, y)


# In[41]:


grid_search_gb.best_params_


# In[42]:


final_gb_model = GradientBoostingClassifier(
    learning_rate=0.01,
    max_depth=3,
    n_estimators=100
)


# In[43]:


final_gb_model.fit(X_encoded,y)


# In[44]:


y_pred_final_gb = final_gb_model.predict(X_test)
print(classification_report(y_test, y_pred_final_gb))


# In[ ]:




