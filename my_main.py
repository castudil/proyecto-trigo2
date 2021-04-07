# -*- coding: utf-8 -*-
import time
import pandas
import numpy as np
from sklearn.preprocessing import StandardScaler

import scipy.cluster.hierarchy as hcluster

import my_boruta
import my_lasso
import my_kbest
import my_ga


# leer csv
datos = pandas.read_csv("data-total.csv", header=0 ,delimiter=";", encoding='ISO-8859-1')

# Variable a predecir
target = "Chl"

# filtramos los datos con las siguientes condiciones
# Año = 2016
# Fenologia != antesis
# condición != secano
# Genotipo = "QUP 2569-2009"

# filtro por año (2014)------------------------------------
filtro1_2014 = datos[datos["ANIO"] == 2014]
filtro2_2014 = datos[datos["ANIO"] == 2014]

filtro1_2014 = filtro1_2014[filtro1_2014["CONDICION"] != "SECANO"]
filtro2_2014 = filtro2_2014[filtro2_2014["CONDICION"] == "SECANO"]

df_chl_control_2014 = filtro1_2014.loc[ : , target]
df_firma_control_2014 = filtro1_2014.loc[ : , "350":"2500"]
cols = list(df_firma_control_2014.columns.values) # recuperamos los nombres de columnas

# Estandarizar control 2014
df_firma_control_2014 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2014)) 
df_firma_control_2014.columns = cols

df_chl_secano_2014 = filtro2_2014.loc[ : , target]
df_firma_secano_2014 = filtro2_2014.loc[ : , "350":"2500"]

# Estandarizar secano 2014
df_firma_secano_2014 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2014)) 
df_firma_secano_2014.columns = cols

# Unir columna a predecir con predictores
control_2014 = pandas.concat([df_chl_control_2014.reset_index(drop=True), df_firma_control_2014], axis = 1)
secano_2014 = pandas.concat([df_chl_secano_2014.reset_index(drop=True), df_firma_secano_2014], axis = 1)

# eliminar NAs
control_2014.dropna(inplace = True)
secano_2014.dropna(inplace = True)

# Reasignamos estas variables, pero ahora se eliminaron los NAs y se estandarizó
firma_control_2014 = control_2014.loc[ : , "350":"2500"]
firma_secano_2014 = secano_2014.loc[ : , "350":"2500"]


# filtro por año (2015) -----------------------------------
filtro1_2015 = datos[datos["ANIO"] == 2015]
filtro2_2015 = datos[datos["ANIO"] == 2015]

filtro1_2015 = filtro1_2015[filtro1_2015["CONDICION"] != "SECANO"]
filtro2_2015 = filtro2_2015[filtro2_2015["CONDICION"] == "SECANO"]

df_chl_control_2015 = filtro1_2015.loc[ : , target]
df_firma_control_2015 = filtro1_2015.loc[ : , "350":"2500"]
cols = list(df_firma_control_2015.columns.values) # recuperamos los nombres de columnas

# Estandarizar control 2015
df_firma_control_2015 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2015)) 
df_firma_control_2015.columns = cols

df_chl_secano_2015 = filtro2_2015.loc[ : , target]
df_firma_secano_2015 = filtro2_2015.loc[ : , "350":"2500"]

# Estandarizar secano 2015
df_firma_secano_2015 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2015)) 
df_firma_secano_2015.columns = cols

# Unir columna a predecir con predictores
control_2015 = pandas.concat([df_chl_control_2015.reset_index(drop=True), df_firma_control_2015], axis = 1)
secano_2015 = pandas.concat([df_chl_secano_2015.reset_index(drop=True), df_firma_secano_2015], axis = 1)

# eliminar NAs
control_2015.dropna(inplace = True)
secano_2015.dropna(inplace = True)

# Reasignamos estas variables, pero ahora se eliminaron los NAs y se estandarizó
firma_control_2015 = control_2015.loc[ : , "350":"2500"]
firma_secano_2015 = secano_2015.loc[ : , "350":"2500"]


# filtro por año (2016) -----------------------------------------------
filtro1_2016 = datos[datos["ANIO"] == 2016]
filtro2_2016 = datos[datos["ANIO"] == 2016]

filtro1_2016 = filtro1_2016[filtro1_2016["CONDICION"] != "SECANO"]
filtro2_2016 = filtro2_2016[filtro2_2016["CONDICION"] == "SECANO"]


df_chl_control_2016 = filtro1_2016.loc[ : , target]
df_firma_control_2016 = filtro1_2016.loc[ : , "350":"2499"]
cols = list(df_firma_control_2016.columns.values) # recuperamos los nombres de columnas

# Estandarizar control 2016
df_firma_control_2016 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2016)) 
df_firma_control_2016.columns = cols

df_chl_secano_2016 = filtro2_2016.loc[ : , target]
df_firma_secano_2016 = filtro2_2016.loc[ : , "350":"2499"]

# Estandarizar secano 2016
df_firma_secano_2016 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2016)) 
df_firma_secano_2016.columns = cols

# Unir columna a predecir con predictores
control_2016 = pandas.concat([df_chl_control_2016.reset_index(drop=True), df_firma_control_2016], axis = 1)
secano_2016 = pandas.concat([df_chl_secano_2016.reset_index(drop=True), df_firma_secano_2016], axis = 1)

# eliminar NAs
control_2016.dropna(inplace = True)
secano_2016.dropna(inplace = True)

# Reasignamos estas variables, pero ahora se eliminaron los NAs y se estandarizó
firma_control_2016 = control_2016.loc[ : , "350":"2499"]
firma_secano_2016 = secano_2016.loc[ : , "350":"2499"]


# filtro por año 2017 -----------------------------------------------
filtro1_2017 = datos[datos["ANIO"] == 2017]
filtro2_2017 = datos[datos["ANIO"] == 2017]

filtro1_2017 = filtro1_2017[filtro1_2017["CONDICION"] != "SECANO"]
filtro2_2017 = filtro2_2017[filtro2_2017["CONDICION"] == "SECANO"]

df_chl_control_2017 = filtro1_2017.loc[ : , target]
df_firma_control_2017 = filtro1_2017.loc[ : , "350":"2500"]
cols = list(df_firma_control_2017.columns.values) # recuperamos los nombres de columnas

# Estandarizar control 2017
df_firma_control_2017 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2017)) 
df_firma_control_2017.columns = cols

df_chl_secano_2017 = filtro2_2017.loc[ : , target]
df_firma_secano_2017 = filtro2_2017.loc[ : , "350":"2500"]

# Estandarizar secano 2017
df_firma_secano_2017 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2017)) 
df_firma_secano_2017.columns = cols

# Unir columna a predecir con predictores
control_2017 = pandas.concat([df_chl_control_2017.reset_index(drop=True), df_firma_control_2017], axis = 1)
secano_2017 = pandas.concat([df_chl_secano_2017.reset_index(drop=True), df_firma_secano_2017], axis = 1)

# eliminar NAs
control_2017.dropna(inplace = True)
secano_2017.dropna(inplace = True)

# Reasignamos estas variables, pero ahora se eliminaron los NAs y se estandarizó
firma_control_2017 = control_2017.loc[ : , "350":"2500"]
firma_secano_2017 = secano_2017.loc[ : , "350":"2500"]



def rangos_clustering(data):
    # se revisa si el parámetro viene vacío
    if len(data) == 0:
        print("No hay datos.")
        return
    
    if len(data) == 1:
        return data
    
    ndata = [[d, d] for d in data]
    new_data = np.array(ndata)
    
    thresh = (11.0/100.0) * (max(data) - min(data))
    
    clusters = hcluster.fclusterdata(new_data, thresh, criterion = "distance")
    tot_clusters = max(clusters)
    
    clustered_index = []
    for i in range(tot_clusters):
        clustered_index.append([])
    
    for i in range(len(clusters)):
        clustered_index[clusters[i] - 1].append(i)
        
    rngs = []
    for x in clustered_index:
        clustered_index_x = [data[y] for y in x]
        rngs.append((min(clustered_index_x), max(clustered_index_x)))
    
    return sorted(rngs)


def string_to_int(lista):
    for i in range(len(lista)):
        lista[i] = int(lista[i])
    return lista

# Inicio del programa ########################################################

print("Seleccione el algoritmo de selección de atributos que desea ejecutar: ")
print("1:\tBoruta.")
print("2:\tLasso.")
print("3:\tSelectK-Best (Mutual Information).")
print("4:\tSelectK-Best (Correlation).")
print("5:\tGenetic Algorithm.")
print("6:\tTodos los anteriores.")
print("7:\tSalir.")

op = input("Introduzca opción: ")

print("Variable objetivo:", target)

while 1:
    if op == '1':
        start = time.perf_counter()
        #boruta()
        elegidos = my_boruta.my_boruta_init(target, firma_control_2014, control_2014)
        print(rangos_clustering(elegidos))
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
    
    elif op == '2':
        start = time.perf_counter()
        #lasso()
        elegidos = my_lasso.my_lasso_init(target, firma_control_2014, control_2014, cols)
        print(rangos_clustering(elegidos))
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
        
    elif op == '3':
        start = time.perf_counter()
        elegidos = my_kbest.kbest_mi(target, firma_control_2014, control_2014)
        print(rangos_clustering(elegidos))
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
    
    elif op == '4':
        start = time.perf_counter()
        elegidos = my_kbest.kbest_corr(target, firma_control_2014, control_2014)
        print(rangos_clustering(elegidos))
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
        
    elif op == '5':
        start = time.perf_counter()
        #ga()
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
        
    elif op == '6':
        start = time.perf_counter()
        #boruta()
        #lasso()
        #kbest_mi()
        #kbest_corr()
        end = time.perf_counter()
        print(f'Tiempo de ejecución: {end - start:0.2f} segundos.')
        
    elif op == '7':
        print("Fin del programa.")
        break
        
    else:
        print("Opción no válida, intente nuevamente.")
    
    print()
    print("Seleccione el algoritmo de selección de atributos que desea ejecutar: ")
    print("1:\tBoruta.")
    print("2:\tLasso.")
    print("3:\tSelectK-Best (Mutual Information).")
    print("4:\tSelectK-Best (Correlation).")
    print("5:\tGenetic Algorithm.")
    print("6:\tTodos los anteriores.")
    print("7:\tSalir.")
    op = input("Introduzca opción: ")
















