
# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
df_train = pd.read_csv("toy_trainX.csv", header=None)
df_test = pd.read_csv("toy_testX.csv", header=None)
df_ltrain = pd.read_csv("toy_trainY.csv", header=None)
df_ltest = pd.read_csv("toy_testY.csv", header=None)
print(df_test.head())
traind = df_train.values
labeld = df_ltrain[0].values
testd = df_test.values
labeltest = df_ltest[0].values

x = np.arange(1,4, 0.01)

model = LogisticRegression(max_iter=100)
model = model.fit(traind, labeld)
print(model.score(traind, labeld))
print(model.score(testd, labeltest))


print(model.coef_[0])
print(model.intercept_)

a = model.coef_[0][0]
b = model.coef_[0][1]
c = model.intercept_[0]

colors = np.chararray(380)

plt.scatter(traind[:,0], traind[:,1], c=labeld)

x = np.arange(-5,5, 0.01)
y = -1.0*c/b - a*x/b

plt.plot(x, y, "r")

plt.show()

plt.scatter(testd[:,0], testd[:,1], c=labeltest)

x = np.arange(-5, 5, 0.01)
y = -1.0*c/b - a*x/b

plt.plot(x, y, "r")

plt.show()


# In[ ]:




# In[ ]:




# In[44]:




# In[4]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



