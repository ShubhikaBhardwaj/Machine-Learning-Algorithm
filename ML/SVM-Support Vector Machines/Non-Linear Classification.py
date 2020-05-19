#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from  sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[27]:


X,Y=make_circles(n_samples=500,noise=0.02)


# In[28]:


print(X.shape,Y.shape)


# In[29]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[30]:


#Convert [x1,x2]  to [x1,x2,x3]  where X3=x1**2 +x2**2


# In[31]:


def phi(X):
    """Non Linear Transformation"""
    x1=X[:,0]
    x2=X[:,1]
    x3=x1**2 +x2**2
    
    x_=np.zeros((X.shape[0],3))
    print(x_.shape)
    
    x_[:,:-1]=X
    x_[:,-1]=x3
    
    return x_

    


# In[32]:


x_=phi(X)


# In[33]:


print(X[:3,:])


# In[34]:


print(x_[:3,:])


# In[55]:


def plot3d(X,show=True):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')
    x1=X[:,0]
    x2=X[:,1]
    x3=X[:,2]
    
    ax.scatter(x1,x2,x3,zdir='z',s=20,c=Y,depthshade=True)
    
    if(show==True):
        plt.show()
    return ax
    
    


# In[56]:


plot3d(x_)


# ## Logistic Classifier 

# In[57]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[58]:


lr=LogisticRegression()


# In[59]:


acc=cross_val_score(lr,X,Y,cv=5).mean()
print("Accuracy X(2D) is %.4f" %(acc*100))


# In[60]:


#Not a good classifier for 2D case


# ## Logistic Classifier on Higher Dimension Space

# In[61]:


acc =cross_val_score(lr,x_,Y,cv=5).mean()
print("Accuracy X(2D) is %.4f" %(acc*100))


# ## Visualise the Decision Surface

# In[62]:


lr.fit(x_,Y)


# In[63]:


wts=lr.coef_
print(wts)


# In[64]:


bias=lr.intercept_  #gives bias
print(bias)


# In[65]:


xx,yy=np.meshgrid(range(-2,2),range(-2,2))
print(xx)
print(yy)


# In[66]:


z=-(wts[0,0]*xx+wts[0,1]*yy+bias)/wts[0,2]
print(z)


# In[71]:


ax=plot3d(x_,False)
ax.plot_surface(xx,yy,z,alpha=0.5)
plt.show()


# In[ ]:





# In[ ]:




