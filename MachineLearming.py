#!/usr/bin/env python
# coding: utf-8

# # Modelos de predicción

# In[1]:


import pandas as pd
import numpy as np
#Gráficas
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
#Para evitar alertas
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv("data_preprocessed.csv")
df.iloc[np.random.randint(0,df.shape[0],10),:]


# In[3]:


df.info()


# In[4]:


fig,ax=plt.subplots(1,2,figsize=(20,7))
ax[0].hist(df.price,bins=30)
ax[0].set_title('Sin tranformación logarítmica',fontsize=14,fontweight="bold")
ax[1].hist(np.log(df.price),bins=30)
ax[1].set_title('Con tranformación logarítmica', fontsize=14 ,fontweight="bold")
plt.suptitle("Price",fontsize=16,color="red",fontweight="bold")
plt.show()


# Haciendo la partición de los datos:

# In[5]:


from sklearn.model_selection import train_test_split,cross_validate
train , test = train_test_split(df,test_size=0.2,random_state=141) 
print(f"train:\t{len(train)}\ntest:\t{len(test)}")


# Estableciendo la columna objetivo en `train` y `test`:

# In[6]:


for dfp in ["train","test"]:
    exec("{0}_x={0}.drop(columns='price')".format(dfp))
    exec("{0}_y={0}.price".format(dfp))


# In[7]:


#títulos of gráficos 
curv_val="Curva de validación"
curv_val_trans="Curva de validación con target transformada"


# In[8]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold
main_kfold = KFold(10, shuffle=True, random_state=45)
from sklearn.compose import TransformedTargetRegressor


# ### Regresión linear simple

# In[9]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import Pipeline\nfrom sklearn.preprocessing import RobustScaler\nlr_pipe = Pipeline([('scale', RobustScaler()),('lr', LinearRegression(n_jobs=-1))])\n\nscoring = {'nmae': 'neg_mean_absolute_error',\n           'nmse': 'neg_mean_squared_error',\n           'r2': 'r2'}\n\nlr_scores = cross_validate(lr_pipe, train_x, train_y,\n                           scoring=scoring, cv=main_kfold,\n                           return_train_score=True, return_estimator=True, n_jobs=-1)\nlr_pipe.fit(train_x, train_y)")


# In[10]:


print('Train:', -lr_scores['train_nmae'].mean().round(2))


# In[11]:


predict_y=lr_pipe.predict(test_x)
error_rl=mean_absolute_error(predict_y,test_y).round(2)
print("Test:",error_rl)


# In[12]:


plt.figure(figsize=(10,10))
plt.title("Regresión lineal simple",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
#plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()  


# Con tranformación logarítmica:

# In[13]:


get_ipython().run_cell_magic('time', '', 'lr_trans = TransformedTargetRegressor(lr_pipe,func=np.log,inverse_func=np.exp)\nlr_trans_scores = cross_validate(lr_trans, train_x, train_y,\n                                 scoring=scoring, cv=main_kfold,\n                                 return_train_score=True, return_estimator=True, n_jobs=-1)\n#Ajuste del modelo \nlr_trans.fit(train_x, train_y)')


# In[14]:


print('Train:', -lr_trans_scores['train_nmae'].mean().round(2))


# In[15]:


predict_y=lr_trans.predict(test_x)
predict_y[predict_y==np.inf]=predict_y[predict_y!=np.inf].max()
print("Test:",mean_absolute_error(predict_y,test_y).round(2))


# In[16]:


plt.figure(figsize=(10,10))
plt.title("Regresión lineal simple con transformación logarítmica",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()  


# ### Lasso

# In[17]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import Lasso\nclf = Lasso()\nlr_pipe = Pipeline([('scale', RobustScaler()),('lr', clf)])\n\nscoring = {'nmae': 'neg_mean_absolute_error',\n           'nmse': 'neg_mean_squared_error',\n           'r2': 'r2'}\n\nlr_scores = cross_validate(lr_pipe, train_x, train_y,\n                           scoring=scoring, cv=main_kfold,\n                           return_train_score=True, return_estimator=True, n_jobs=-1)\nlr_pipe.fit(train_x, train_y)")


# In[18]:


print('Train:', -lr_scores['train_nmae'].mean().round(2))


# In[19]:


predict_y=lr_pipe.predict(test_x)
error_lasso=mean_absolute_error(predict_y,test_y).round(2)
print("Test:",error_lasso)


# In[20]:


plt.figure(figsize=(10,10))
plt.title("Lasso",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
#plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show() 


# Con transformación logarítmica:

# In[21]:


get_ipython().run_cell_magic('time', '', 'lr_trans = TransformedTargetRegressor(lr_pipe,func=np.log,inverse_func=np.exp)\nlr_trans_scores = cross_validate(lr_trans, train_x, train_y,\n                                 scoring=scoring, cv=main_kfold,\n                                 return_train_score=True, return_estimator=True, n_jobs=-1)\n#Ajuste del modelo \nlr_trans.fit(train_x, train_y)')


# In[22]:


print('Train:', -lr_trans_scores['train_nmae'].mean().round(2))


# In[23]:


predict_y=lr_trans.predict(test_x)
predict_y[predict_y==np.inf]=predict_y[predict_y!=np.inf].max()
print("Test:",mean_absolute_error(predict_y,test_y).round(2))


# In[24]:


plt.figure(figsize=(10,10))
plt.title("Lasso con transformación logarítmica",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()  


# ### Ridge

# In[25]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import Ridge\nclf = Ridge()\nlr_pipe = Pipeline([('scale', RobustScaler()),('lr', clf)])\n\nscoring = {'nmae': 'neg_mean_absolute_error',\n           'nmse': 'neg_mean_squared_error',\n           'r2': 'r2'}\n\nlr_scores = cross_validate(lr_pipe, train_x, train_y,\n                           scoring=scoring, cv=main_kfold,\n                           return_train_score=True, return_estimator=True, n_jobs=-1)\nlr_pipe.fit(train_x, train_y)")


# In[26]:


print('Train:', -lr_scores['train_nmae'].mean().round(2))


# In[27]:


predict_y=lr_pipe.predict(test_x)
error_ridge=mean_absolute_error(predict_y,test_y).round(2)
print("Test:",error_ridge)


# In[28]:


plt.figure(figsize=(10,10))
plt.title("Ridge",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
#plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show() 


# Con tranformación logarítmica:

# In[29]:


get_ipython().run_cell_magic('time', '', 'lr_trans = TransformedTargetRegressor(lr_pipe,func=np.log,inverse_func=np.exp)\nlr_trans_scores = cross_validate(lr_trans, train_x, train_y,\n                                 scoring=scoring, cv=main_kfold,\n                                 return_train_score=True, return_estimator=True, n_jobs=-1)\n#Ajuste del modelo \nlr_trans.fit(train_x, train_y)')


# In[30]:


print('Train:', -lr_trans_scores['train_nmae'].mean().round(2))


# In[31]:


predict_y=lr_trans.predict(test_x)
predict_y[predict_y==np.inf]=predict_y[predict_y!=np.inf].max()
print("Test:",mean_absolute_error(predict_y,test_y).round(2))


# In[32]:


plt.figure(figsize=(10,10))
plt.title("Ridge con tranformación logarítmica",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()


# ### ElasticNet

# In[33]:


get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import ElasticNet\nclf = ElasticNet()\nlr_pipe = Pipeline([('scale', RobustScaler()),('lr', clf)])\n\nscoring = {'nmae': 'neg_mean_absolute_error',\n           'nmse': 'neg_mean_squared_error',\n           'r2': 'r2'}\n\nlr_scores = cross_validate(lr_pipe, train_x, train_y,\n                           scoring=scoring, cv=main_kfold,\n                           return_train_score=True, return_estimator=True, n_jobs=-1)\nlr_pipe.fit(train_x, train_y)")


# In[34]:


print('Train:', -lr_scores['train_nmae'].mean().round(2))


# In[35]:


predict_y=lr_pipe.predict(test_x)
error_elasticnet=mean_absolute_error(predict_y,test_y).round(2)
print("Test:",error_elasticnet)


# In[36]:


plt.figure(figsize=(10,10))
plt.title("ElasticNet",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
#plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show() 


# Con transformación logarítmica

# In[37]:


get_ipython().run_cell_magic('time', '', 'lr_trans = TransformedTargetRegressor(lr_pipe,func=np.log,inverse_func=np.exp)\nlr_trans_scores = cross_validate(lr_trans, train_x, train_y,\n                                 scoring=scoring, cv=main_kfold,\n                                 return_train_score=True, return_estimator=True, n_jobs=-1)\n#Ajuste del modelo \nlr_trans.fit(train_x, train_y)')


# In[38]:


print('Train:', -lr_trans_scores['train_nmae'].mean().round(2))


# In[39]:


predict_y=lr_trans.predict(test_x)
predict_y[predict_y==np.inf]=predict_y[predict_y!=np.inf].max()
print("Test:",mean_absolute_error(predict_y,test_y).round(2))


# In[40]:


plt.figure(figsize=(10,10))
plt.title("ElasticNet con tranformación logarítmica",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()


# ### K-Nearest Neighboor

# In[41]:


get_ipython().run_cell_magic('time', '', 'n_neighbors= np.arange(3,25,2)\nfrom sklearn.neighbors import KNeighborsRegressor\n\n#Creando un modelo\nknn = KNeighborsRegressor(p=1,n_jobs=-1)\n#Creando un diccionario donde queremos evaluar n_neighbors\nparam_grid = {"n_neighbors": n_neighbors}\n#Usando gridsearch para evaluar todos los valores de n_neighbors\nknn_gscv = GridSearchCV(knn, param_grid, cv=main_kfold,scoring=\'neg_mean_absolute_error\',n_jobs=-1)\n#Ajustando el modelo \nknn_gscv.fit(train_x, train_y)')


# In[42]:


plt.figure(figsize=(20,7))
plt.plot(n_neighbors, -knn_gscv.cv_results_['mean_test_score']) 
plt.title(curv_val+": KNN",fontsize=18,color='red',fontweight="bold")
plt.xlabel('n of neighbors',fontsize=16)
plt.xticks(np.arange(0,25,step=5))
plt.ylabel('mean absolute error',fontsize=16)
plt.grid(ls='--',lw=0.7) 
plt.show()    


# In[43]:


predict_y=knn_gscv.predict(test_x)
error_knn=mean_absolute_error(predict_y,test_y).round(2)
print("Test:",error_knn)


# In[44]:


plt.figure(figsize=(10,10))
plt.title("KNN",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price estimado',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()  


# Con tranformación logarítmica

# In[45]:


get_ipython().run_cell_magic('time', '', "model_pipe = Pipeline([('model', TransformedTargetRegressor(knn,func=np.log,inverse_func=np.exp))])\nparams_grid_trans={'model__regressor__n_neighbors':n_neighbors}\nknn_gscv_trans = GridSearchCV(model_pipe, params_grid_trans, cv=main_kfold,scoring='neg_mean_absolute_error',n_jobs=-1)\nknn_gscv_trans.fit(train_x, train_y)")


# In[46]:


plt.figure(figsize=(20,7))
plt.plot(n_neighbors, -knn_gscv_trans.cv_results_['mean_test_score']) 
plt.title(curv_val_trans +": KNN",fontsize=18,color='red',fontweight="bold")
plt.xlabel('n of neighbors',fontsize=16)
plt.xticks(np.arange(0,25,step=5))
plt.ylabel('mean absolute error',fontsize=16)
plt.grid(ls='--',lw=0.7) 
plt.show()    


# In[47]:


predict_y=knn_gscv_trans.predict(test_x)
print("Test:",mean_absolute_error(test_y,predict_y).round(2))


# In[48]:


plt.figure(figsize=(10,10))
plt.title("KNN con tranformación logarítmica",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()    


# ### Random Forest

# In[49]:


get_ipython().run_cell_magic('time', '', 'N=np.array([15,50,100,250,300,350])\nfrom sklearn.ensemble import RandomForestRegressor \n\n#Creando un modelo\nrf = RandomForestRegressor(n_jobs=-1)\n#Creando un diccionario de todos los valores que queremos evaluar para n_neighbors\nparam_grid = {"n_estimators": N}\n\n#Usando gridsearch para evaluar todos los valores para n_neighbors\nrf_gscv = GridSearchCV(rf, param_grid, cv=main_kfold,scoring=\'neg_mean_absolute_error\',n_jobs=-1)\n#Ajustando el modelo\nrf_gscv.fit(train_x, train_y)')


# In[50]:


plt.figure(figsize=(20,7))
plt.plot(N, -rf_gscv.cv_results_['mean_test_score']) 
plt.title(curv_val + ": RF",fontsize=18,color='red',fontweight="bold")
plt.xlabel('n of estimators',fontsize=16)
plt.ylabel('mean absolute error',fontsize=16)
plt.show() 


# In[51]:


predict_y=rf_gscv.predict(test_x)
error_rf=mean_absolute_error(predict_y,test_y).round(2)
print("Test:",error_rf)


# In[52]:


plt.figure(figsize=(10,10))
plt.title("Random Forest",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()


# Con transformación logarítmica:

# In[53]:


get_ipython().run_cell_magic('time', '', "model_pipe = Pipeline([('model', TransformedTargetRegressor(rf,func=np.log,inverse_func=np.exp))])\nparams_grid_trans={'model__regressor__n_estimators':N}\nrf_gscv_trans = GridSearchCV(model_pipe, params_grid_trans, cv=main_kfold,scoring='neg_mean_absolute_error',n_jobs=-1)\nrf_gscv_trans.fit(train_x, train_y)")


# In[54]:


plt.figure(figsize=(20,7))
plt.plot(N, -rf_gscv_trans.cv_results_['mean_test_score']) 
plt.title(curv_val_trans +": RF",fontsize=18,color='red',fontweight="bold")
plt.xlabel('n of estimators',fontsize=16)
plt.ylabel('mean absolute error',fontsize=16)
plt.show() 


# In[55]:


predict_y=rf_gscv_trans.predict(test_x)
print("Test:",mean_absolute_error(test_y,predict_y).round(2))


# In[56]:


plt.figure(figsize=(10,10))
plt.title("Random Forest con tranformación logarítmica",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()  


# ### Gradient Boosting

# In[57]:


get_ipython().run_cell_magic('time', '', "N=np.array([15,50,100,300,500,800])\nfrom sklearn.ensemble import GradientBoostingRegressor \n#Creando el modelo\ngb = GradientBoostingRegressor()\nparam_grid = {'n_estimators': N}\n#Usando gridsearch para evaluar todos los valores\ngb_gscv = GridSearchCV(gb, param_grid, cv=main_kfold,scoring='neg_mean_absolute_error',n_jobs=-1)\n#Ajustando el modelo\ngb_gscv.fit(train_x, train_y)")


# In[58]:


plt.figure(figsize=(20,7))
plt.plot(N, -gb_gscv.cv_results_['mean_test_score']) 
plt.title(curv_val +": GB",fontsize=18,color='red',fontweight="bold")
plt.xlabel('n of estimators',fontsize=16)
plt.xticks(N)
plt.ylabel('mean absolute error',fontsize=16)
plt.grid(ls='--',lw=0.7) 
plt.show()   


# In[59]:


predict_y=gb_gscv.predict(test_x)
print("Test:",mean_absolute_error(test_y,predict_y).round(2))


# In[60]:


plt.figure(figsize=(10,10))
plt.title("Gradient Bossting",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price predecida',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()    


# Con tranformación logarítmica:

# In[61]:


get_ipython().run_cell_magic('time', '', "model_pipe = Pipeline([('model', TransformedTargetRegressor(gb,func=np.log,inverse_func=np.exp))])\nparams_grid_trans={'model__regressor__n_estimators':N}\ngb_gscv_trans = GridSearchCV(model_pipe, params_grid_trans, cv=main_kfold,scoring='neg_mean_absolute_error',n_jobs=-1)\ngb_gscv_trans.fit(train_x, train_y)")


# In[62]:


plt.figure(figsize=(20,7))
plt.plot(N, -gb_gscv_trans.cv_results_['mean_test_score']) 
plt.title(curv_val_trans +": GB",fontsize=18,color='red',fontweight="bold")
plt.xlabel('n of estimators',fontsize=16)
plt.xticks(N)
plt.ylabel('mean absolute error',fontsize=16)
plt.grid(ls='--',lw=0.7) 
plt.show()    


# In[63]:


predict_y=gb_gscv_trans.predict(test_x)
error_gb=mean_absolute_error(test_y,predict_y).round(2)
print("Test:",error_gb)


# In[64]:


plt.figure(figsize=(10,10))
plt.title("Gradient Boosting con transformación logarítmica",fontsize=16,color='red',fontweight="bold")
plt.scatter(test_y, predict_y,color="steelblue") 
plt.plot(range(test_y.max()),range(test_y.max()),color="orange")
plt.axis('scaled')
plt.xlabel('Price real',fontsize=16)
plt.ylabel('Price estimada',fontsize=16)
plt.grid(ls='--',lw=0.7)    
plt.show()  


# - Es el único método que ofrece una mejor calidad con la transformación logarítmica.
# - El único handicap es el tiempo de cálculo para volumenes más grandes.
# ### Resumen

# In[65]:


#Customizando el texto
class color:
   DARKCYAN = '\033[36m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


# In[66]:


metodos=["rl","lasso","elasticnet","knn","rf","gb"]
sentencia_tl=["No"]*5 + ["Sí"]
print(color.DARKCYAN + color.UNDERLINE + color.BOLD+"RESUMEN:" + color.END)
print(color.BOLD + "Métodos".ljust(14),"Error medio (en euros)".ljust(23)+ "¿Ayuda la transformación logarítmica?"+ color.END)
for met,bol in zip(metodos,sentencia_tl):
    print(met.upper().ljust(14),str(eval(f"error_{met}")).ljust(22),bol)

