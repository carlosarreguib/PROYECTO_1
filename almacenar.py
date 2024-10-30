import pandas as pd
import numpy as np
df=pd.read_csv("Barcelona_Fotocasa_Housingprices.csv")
#Eliminando la primera columna
df.drop("Unnamed: 0",inplace=True,axis=1)
#Reemplazando los valores nulos con None para que mysql lo entienda
df = df.where(pd.notnull(df), None)
#Almacenando en una base de datos de mysql
import pymysql
connection = pymysql.connect( host="127.0.0.1",user='root', charset= 'utf8mb4', db= "viviendas_bcn", cursorclass=pymysql.cursors.DictCursor )
cursor = connection.cursor()
sql = '''INSERT INTO `viviendas_bcn`.`kaggle` (`price`,`rooms`, `bathroom`, `lift`,
`terrace`,`square_meters`,`real_state`, `neighborhood`,`square_meters_price`) VALUES
(%s,%s,%s,%s,%s,%s,%s,%s,%s)'''
for i in range(df.shape[0]):
    cursor.execute(sql,(tuple(df.iloc[i])))
    connection.commit()
cursor.close()
connection.close()
