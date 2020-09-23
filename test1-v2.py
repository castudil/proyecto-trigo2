# -*- coding: utf-8 -*-
# from sklearn.ensemble import RandomForestRegressor
import pandas
# from boruta import BorutaPy
import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV


# leer csv
datos = pandas.read_csv("data.csv", header=0 ,delimiter=";", encoding='ISO-8859-1')

# filtramos los datos con las siguientes condiciones
# Año = 2016
# Fenologia != antesis
# condición != secano
# Genotipo = "QUP 2569-2009"

# filtro por año (2014) 
filtro1_2014 = datos[datos["ANIO"] == 2014]
filtro2_2014 = datos[datos["ANIO"] == 2014]

filtro1_2014 = filtro1_2014[filtro1_2014["CONDICION"] != "SECANO"]
filtro2_2014 = filtro2_2014[filtro2_2014["CONDICION"] == "SECANO"]

df_chl_control_2014 = filtro1_2014.loc[ : , "Chl"]
df_firma_control_2014 = filtro1_2014.loc[ : , "350":"2500"]
cols = list(df_firma_control_2014.columns.values) # recuperamos los nombres de columnas
# Estandarizar
df_firma_control_2014 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2014)) 
df_firma_control_2014.columns = cols

df_chl_secano_2014 = filtro2_2014.loc[ : , "Chl"]
df_firma_secano_2014 = filtro2_2014.loc[ : , "350":"2500"]
# Estandarizar
df_firma_secano_2014 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2014)) 
df_firma_secano_2014.columns = cols



# filtro por año (2015)
filtro1_2015 = datos[datos["ANIO"] == 2015]
filtro2_2015 = datos[datos["ANIO"] == 2015]

filtro1_2015 = filtro1_2015[filtro1_2015["CONDICION"] != "SECANO"]
filtro2_2015 = filtro2_2015[filtro2_2015["CONDICION"] == "SECANO"]

df_chl_control_2015 = filtro1_2015.loc[ : , "Chl"]
df_firma_control_2015 = filtro1_2015.loc[ : , "350":"2500"]
cols = list(df_firma_control_2015.columns.values) # recuperamos los nombres de columnas
# Estandarizar
df_firma_control_2015 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2015)) 
df_firma_control_2015.columns = cols

df_chl_secano_2015 = filtro2_2015.loc[ : , "Chl"]
df_firma_secano_2015 = filtro2_2015.loc[ : , "350":"2500"]
# Estandarizar
df_firma_secano_2015 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2015)) 
df_firma_secano_2015.columns = cols

# filtro por año (2016)
filtro1_2016 = datos[datos["ANIO"] == 2016]
filtro2_2016 = datos[datos["ANIO"] == 2016]

filtro1_2016 = filtro1_2016[filtro1_2016["CONDICION"] != "SECANO"]
filtro2_2016 = filtro2_2016[filtro2_2016["CONDICION"] == "SECANO"]


df_chl_control_2016 = filtro1_2016.loc[ : , "Chl"]
df_firma_control_2016 = filtro1_2016.loc[ : , "350":"2499"]
cols = list(df_firma_control_2016.columns.values) # recuperamos los nombres de columnas
# Estandarizar
df_firma_control_2016 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2016)) 
df_firma_control_2016.columns = cols

df_chl_secano_2016 = filtro2_2016.loc[ : , "Chl"]
df_firma_secano_2016 = filtro2_2016.loc[ : , "350":"2499"]
# Estandarizar
df_firma_secano_2016 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2016)) 
df_firma_secano_2016.columns = cols


# filtro por año 2017
filtro1_2017 = datos[datos["ANIO"] == 2017]
filtro2_2017 = datos[datos["ANIO"] == 2017]

filtro1_2017 = filtro1_2017[filtro1_2017["CONDICION"] != "SECANO"]
filtro2_2017 = filtro2_2017[filtro2_2017["CONDICION"] == "SECANO"]

df_chl_control_2017 = filtro1_2017.loc[ : , "Chl"]
df_firma_control_2017 = filtro1_2017.loc[ : , "350":"2500"]
cols = list(df_firma_control_2017.columns.values) # recuperamos los nombres de columnas
# Estandarizar
df_firma_control_2017 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_control_2017)) 
df_firma_control_2017.columns = cols

df_chl_secano_2017 = filtro2_2017.loc[ : , "Chl"]
df_firma_secano_2017 = filtro2_2017.loc[ : , "350":"2500"]
# Estandarizar
df_firma_secano_2017 = pandas.DataFrame(StandardScaler().fit_transform(df_firma_secano_2017)) 
df_firma_secano_2017.columns = cols



# ### BORUTA grupo control 2014 #######################
# forest1 = RandomForestRegressor(
#     n_jobs = -1, 
#     max_depth = 5
# )
# boruta1 = BorutaPy(
#     estimator = forest1, 
#     n_estimators = 'auto',
#     max_iter = 100 # number of trials to perform
# )

# # fit Boruta (it accepts np.array, not pd.DataFrame)
# boruta1.fit(np.array(df_firma_control_2014), np.array(df_chl_control_2014))

# # print results grupo control
# green_area_control = df_firma_control_2014.columns[boruta1.support_].to_list()
# blue_area_control = df_firma_control_2014.columns[boruta1.support_weak_].to_list()

# print('Atributos importantes (grupo control 2014):', green_area_control)
# print('Atributos tentativos (grupo control 2014):', blue_area_control)

# print("")

# ### BORUTA grupo secano 2014 ##########################
# forest2 = RandomForestRegressor(
#     n_jobs= -1,
#     max_depth= 5
# )
# boruta2 = BorutaPy(
#     estimator = forest2,
#     n_estimators = 'auto',
#     max_iter = 100
# )

# boruta2.fit(np.array(df_firma_secano_2014), np.array(df_chl_secano_2014))

# # print results grupo control
# green_area_secano = df_firma_secano_2014.columns[boruta2.support_].to_list()
# blue_area_secano = df_firma_secano_2014.columns[boruta2.support_weak_].to_list()

# print('Atributos importantes (grupo secano 2014):', green_area_secano)
# print('Atributos tentativos (grupo secano 2014):', blue_area_secano)

# print("")

# ### BORUTA grupo control 2015 #######################
# forest1 = RandomForestRegressor(
#     n_jobs = -1, 
#     max_depth = 5
# )
# boruta1 = BorutaPy(
#     estimator = forest1, 
#     n_estimators = 'auto',
#     max_iter = 100 # number of trials to perform
# )

# # fit Boruta (it accepts np.array, not pd.DataFrame)
# boruta1.fit(np.array(df_firma_control_2015), np.array(df_chl_control_2015))

# # print results grupo control
# green_area_control = df_firma_control_2015.columns[boruta1.support_].to_list()
# blue_area_control = df_firma_control_2015.columns[boruta1.support_weak_].to_list()

# print('Atributos importantes (grupo control 2015):', green_area_control)
# print('Atributos tentativos (grupo control 2015):', blue_area_control)

# print("")

# ### BORUTA grupo secano 2015 ##########################
# forest2 = RandomForestRegressor(
#     n_jobs= -1,
#     max_depth= 5
# )
# boruta2 = BorutaPy(
#     estimator = forest2,
#     n_estimators = 'auto',
#     max_iter = 100
# )

# boruta2.fit(np.array(df_firma_secano_2015), np.array(df_chl_secano_2015))

# # print results grupo control
# green_area_secano = df_firma_secano_2015.columns[boruta2.support_].to_list()
# blue_area_secano = df_firma_secano_2015.columns[boruta2.support_weak_].to_list()

# print('Atributos importantes (grupo secano 2015):', green_area_secano)
# print('Atributos tentativos (grupo secano 2015):', blue_area_secano)

# print("")


# ### BORUTA grupo control 2016 #######################
# forest1 = RandomForestRegressor(
#     n_jobs = -1, 
#     max_depth = 5
# )
# boruta1 = BorutaPy(
#     estimator = forest1, 
#     n_estimators = 'auto',
#     max_iter = 100 # number of trials to perform
# )

# # fit Boruta (it accepts np.array, not pd.DataFrame)
# boruta1.fit(np.array(df_firma_control_2016), np.array(df_chl_control_2016))

# # print results grupo control
# green_area_control = df_firma_control_2016.columns[boruta1.support_].to_list()
# blue_area_control = df_firma_control_2016.columns[boruta1.support_weak_].to_list()

# print('Atributos importantes (grupo control 2016):', green_area_control)
# print('Atributos tentativos (grupo control 2016):', blue_area_control)

# print("")

# ### BORUTA grupo secano 2016 ##########################
# forest2 = RandomForestRegressor(
#     n_jobs= -1,
#     max_depth= 5
# )
# boruta2 = BorutaPy(
#     estimator = forest2,
#     n_estimators = 'auto',
#     max_iter = 100
# )

# boruta2.fit(np.array(df_firma_secano_2016), np.array(df_chl_secano_2016))

# # print results grupo control
# green_area_secano = df_firma_secano_2016.columns[boruta2.support_].to_list()
# blue_area_secano = df_firma_secano_2016.columns[boruta2.support_weak_].to_list()

# print('Atributos importantes (grupo secano 2016):', green_area_secano)
# print('Atributos tentativos (grupo secano 2016):', blue_area_secano)

# print("")


# ### BORUTA grupo control 2017 #######################
# forest1 = RandomForestRegressor(
#     n_jobs = -1, 
#     max_depth = 5
# )
# boruta1 = BorutaPy(
#     estimator = forest1, 
#     n_estimators = 'auto',
#     max_iter = 100 # number of trials to perform
# )

# # fit Boruta (it accepts np.array, not pd.DataFrame)
# boruta1.fit(np.array(df_firma_control_2017), np.array(df_chl_control_2017))

# # print results grupo control
# green_area_control = df_firma_control_2017.columns[boruta1.support_].to_list()
# blue_area_control = df_firma_control_2017.columns[boruta1.support_weak_].to_list()

# print('Atributos importantes (grupo control 2017):', green_area_control)
# print('Atributos tentativos (grupo control 2017):', blue_area_control)

# print("")

# ### BORUTA grupo secano 2017 ##########################
# forest2 = RandomForestRegressor(
#     n_jobs= -1,
#     max_depth= 5
# )
# boruta2 = BorutaPy(
#     estimator = forest2,
#     n_estimators = 'auto',
#     max_iter = 100
# )

# boruta2.fit(np.array(df_firma_secano_2017), np.array(df_chl_secano_2017))

# # print results grupo control
# green_area_secano = df_firma_secano_2017.columns[boruta2.support_].to_list()
# blue_area_secano = df_firma_secano_2017.columns[boruta2.support_weak_].to_list()

# print('Atributos importantes (grupo secano 2017):', green_area_secano)
# print('Atributos tentativos (grupo secano 2017):', blue_area_secano)

# print("")


# =============================================================================
# print(df_chl_seca.shape)
# print(df_chl.head(15))
# 
# print(df_firma.shape)
# 
# print(filtro1.shape)
# print(filtro1.head())
# 
# print(list(datos.columns.values))
# =============================================================================


# ### PCA grupo control ###################################
# pca_control = PCA(.9) # PCA con 90% de varianza

# pca_control.fit_transform(df_firma_control)

# print("Grupo control -------")
# print("Componentes que entregan el 90% de varianza explicada: ")
# print(pca_control.explained_variance_ratio_)
# print("")
# print("Lista de componentes principales: ")
# print(pca_control.components_)
# print("")
# print("dimensiones de la lista de componentes: ")
# print(np.array(pca_control.components_).shape)
# print("")

# ### PCA grupo secano ####################################
# pca_secano = PCA(.9)

# pca_secano.fit_transform(df_firma_secano)

# print("Grupo secano -------")
# print("Componentes que entregan el 90% de varianza explicada: ")
# print(pca_secano.explained_variance_ratio_)
# print("")
# print("Lista de componentes principales: ")
# print(pca_secano.components_)
# print("")
# print("dimensiones de la lista de componentes: ")
# print(np.array(pca_secano.components_).shape)
# print("")


# ### LASSO feature selection ####################
# # Grupo control 2014
# lasso_control = LassoCV(max_iter = 6000).fit(df_firma_control_2014, df_chl_control_2014)
# importancia_c = np.abs(lasso_control.coef_)
# #print(importancia_c)

# # Buscaremos los 30 atributos más importantes
# # para estudiar su comportamiento
# # Fijamos el humbral sobre el atributo nro 31
# attr_31 = importancia_c.argsort()[-31]
# humbral = importancia_c[attr_31] + 0.01

# attrs_c = (-importancia_c).argsort()[:30]
# cols_aux = np.array(cols)[attrs_c]

# print("Atributos seleccionados (control 2014): {}".format(cols_aux))
# print("")
# # nota: luego de correr por primera vez el programa,
# # cantidad máxima tentativa de atributos: 37

# # Grupo secano 2014
# lasso_secano = LassoCV(max_iter = 5000).fit(df_firma_secano_2014, df_chl_secano_2014)
# importancia_s = np.abs(lasso_secano.coef_)
# #print(importancia_s)

# attr_31 = importancia_s.argsort()[-31]
# humbral = importancia_s[attr_31] + 0.01

# attrs_s = (-importancia_s).argsort()[:30]
# cols_aux = np.array(cols)[attrs_s]

# print("Atributos seleccionados (secano 2014): {}".format(cols_aux))
# print("")


# Grupo control 2015
lasso_control = LassoCV(max_iter = 6000).fit(df_firma_control_2015, df_chl_control_2015)
importancia_c = np.abs(lasso_control.coef_)
#print(importancia_c)

# Buscaremos los 30 atributos más importantes
# para estudiar su comportamiento
# Fijamos el humbral sobre el atributo nro 31
attr_31 = importancia_c.argsort()[-31]
humbral = importancia_c[attr_31] + 0.01

attrs_c = (-importancia_c).argsort()[:30]
cols_aux = np.array(cols)[attrs_c]

print("Atributos seleccionados (control 2015): {}".format(cols_aux))
print("")
# nota: luego de correr por primera vez el programa,
# cantidad máxima tentativa de atributos: 37

# Grupo secano 2015
lasso_secano = LassoCV(max_iter = 6000).fit(df_firma_secano_2015, df_chl_secano_2015)
importancia_s = np.abs(lasso_secano.coef_)
#print(importancia_s)

attr_31 = importancia_s.argsort()[-31]
humbral = importancia_s[attr_31] + 0.01

attrs_s = (-importancia_s).argsort()[:30]
cols_aux = np.array(cols)[attrs_s]

print("Atributos seleccionados (secano 2015): {}".format(cols_aux))
print("")


# Grupo control 2016
lasso_control = LassoCV(max_iter = 5000).fit(df_firma_control_2016, df_chl_control_2016)
importancia_c = np.abs(lasso_control.coef_)
#print(importancia_c)

# Buscaremos los 30 atributos más importantes
# para estudiar su comportamiento
# Fijamos el humbral sobre el atributo nro 31
attr_31 = importancia_c.argsort()[-31]
humbral = importancia_c[attr_31] + 0.01

attrs_c = (-importancia_c).argsort()[:30]
cols_aux = np.array(cols)[attrs_c]

print("Atributos seleccionados (control 2016): {}".format(cols_aux))
print("")
# nota: luego de correr por primera vez el programa,
# cantidad máxima tentativa de atributos: 37

# Grupo secano 2016
lasso_secano = LassoCV(max_iter = 6000).fit(df_firma_secano_2016, df_chl_secano_2016)
importancia_s = np.abs(lasso_secano.coef_)
#print(importancia_s)

attr_31 = importancia_s.argsort()[-31]
humbral = importancia_s[attr_31] + 0.01

attrs_s = (-importancia_s).argsort()[:30]
cols_aux = np.array(cols)[attrs_s]

print("Atributos seleccionados (secano 2016): {}".format(cols_aux))
print("")


# # Grupo control 2017
# lasso_control = LassoCV(max_iter = 6000).fit(df_firma_control_2017, df_chl_control_2017)
# importancia_c = np.abs(lasso_control.coef_)
# print(importancia_c)

# # Buscaremos los 30 atributos más importantes
# # para estudiar su comportamiento
# # Fijamos el humbral sobre el atributo nro 31
# attr_31 = importancia_c.argsort()[-31]
# humbral = importancia_c[attr_31] + 0.01

# attrs_c = (-importancia_c).argsort()[:30]
# cols_aux = np.array(cols)[attrs_c]

# print("Atributos seleccionados (control 2017): {}".format(cols_aux))
# print("")
# # nota: luego de correr por primera vez el programa,
# # cantidad máxima tentativa de atributos: 37

# # Grupo secano 2014
# lasso_secano = LassoCV(max_iter = 5000).fit(df_firma_secano_2017, df_chl_secano_2017)
# importancia_s = np.abs(lasso_secano.coef_)
# print(importancia_s)

# attr_31 = importancia_s.argsort()[-31]
# humbral = importancia_s[attr_31] + 0.01

# attrs_s = (-importancia_s).argsort()[:30]
# cols_aux = np.array(cols)[attrs_s]

# print("Atributos seleccionados (secano 2017): {}".format(cols_aux))
# print("")

















