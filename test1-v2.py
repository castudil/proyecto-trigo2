# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestRegressor
import math
import time
import pandas
from boruta import BorutaPy
import numpy as np
from numpy import mean, std
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model
from feature_selection_ga import FeatureSelectionGA
# from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV
# from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import RepeatedKFold
# from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
# from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster

import my_boruta


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





# Creamos una función fitness personalizada
class CustomFitnessFunctionClass:
    def __init__(self,n_total_features,n_splits = 5, alpha=0.01, *args,**kwargs):
        """
            Parameters
            -----------
            n_total_features :int
            	Total number of features N_t.
            n_splits :int, default = 5
                Number of splits for cv
            alpha :float, default = 0.01
                Tradeoff between the classifier performance P and size of 
                feature subset N_f with respect to the total number of features
                N_t.
            
            verbose: 0 or 1
        """
        self.n_splits = n_splits
        self.alpha = alpha
        self.n_total_features = n_total_features

    def calculate_fitness(self,model,x,y):
        alpha = self.alpha
        total_features = self.n_total_features

        cv_set = np.repeat(-1.,x.shape[0])
        skf = RepeatedKFold(n_splits = self.n_splits)
        for train_index,test_index in skf.split(x,y):
            x_train,x_test = x[train_index],x[test_index]
            y_train,y_test = y[train_index],y[test_index]
            if x_train.shape[0] != y_train.shape[0]:
                raise Exception()
            model.fit(x_train,y_train)
            predicted_y = model.predict(x_test)
            cv_set[test_index] = predicted_y
        
        # P = accuracy_score(y, cv_set)
        P = r2_score(y, cv_set)
        fitness = (alpha*(1.0 - P) + (1.0 - alpha)*(1.0 - (x.shape[1])/total_features))
        return fitness


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


#### PCA grupo control ###################################
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


#### LASSO feature selection ####################
def lasso():
    print("Ejecutando LASSO...")
    max = 150
    # Grupo control 2014
    lasso_control = LassoCV(max_iter = 10000).fit(firma_control_2014, control_2014.loc[ : , target])
    importancia_c = np.abs(lasso_control.coef_)
    # for i in range(len(importancia_c)):
    #     if importancia_c[i] > 0:
    #         print(str(i) + " - " + str(importancia_c[i]))
    # print(importancia_c[:max]) # primeros 100
    # print(importancia_c[::-1][:max]) # ultimos 100
    # print(importancia_c.argsort()[::-1][:max]) # primeros 100 indices ordenados desc
    
    # Buscaremos los 30 atributos más importantes
    # para estudiar su comportamiento
    # Fijamos el humbral sobre el atributo nro 31
    # attr_31 = importancia_c.argsort()[-31]
    # humbral = importancia_c[attr_n] + 0.01
    
    # Este bucle busca en la lista de valores de importancia ordenadas
    # de mayor a menor el índice del primer valor en 0 para mostrar 
    # esa cantidad o un máximo 150 (max) elementos.
    for i in range(len(importancia_c)):
        if importancia_c[importancia_c.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
    
    attrs_c = importancia_c.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_c]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (control 2014): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    
    # graficamos la importancia de cada atributo
    # plt.bar(height=importancia_c, x=np.array(cols))
    # plt.title("LASSO - Grupo control 2014 (" + target + ")")
    # plt.xlabel("Longitud de onda")
    # plt.ylabel("Importancia")
    # plt.show()
    
    # nota: luego de correr por primera vez el programa,
    # cantidad máxima tentativa de atributos: 37
    
    # Grupo secano 2014
    lasso_secano = LassoCV(max_iter = 20000).fit(firma_secano_2014, secano_2014.loc[ : , target])
    importancia_s = np.abs(lasso_secano.coef_)
    #print(importancia_s)
    
    # attr_31 = importancia_s.argsort()[-31]
    # humbral = importancia_s[attr_31] + 0.01
    
    for i in range(len(importancia_s)):
        if importancia_s[importancia_s.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
    
    attrs_s = importancia_s.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_s]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (secano 2014): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    
    
    # Grupo control 2015
    lasso_control = LassoCV(max_iter = 6000).fit(firma_control_2015, control_2015.loc[ : , target])
    importancia_c = np.abs(lasso_control.coef_)
    # print("grupo control 2015")
    # print(importancia_c.argsort()[::-1][:100])
    # print((-importancia_c).argsort()[:100])
    
    # Buscaremos los 30 atributos más importantes
    # para estudiar su comportamiento
    # Fijamos el humbral sobre el atributo nro 31
    # attr_31 = importancia_c.argsort()[-31]
    # humbral = importancia_c[attr_31] + 0.01
    
    for i in range(len(importancia_c)):
        if importancia_c[importancia_c.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
    
    attrs_c = importancia_c.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_c]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (control 2015): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    # nota: luego de correr por primera vez el programa,
    # cantidad máxima tentativa de atributos: 37
    
    # Grupo secano 2015
    lasso_secano = LassoCV(max_iter = 26000).fit(firma_secano_2015, secano_2015.loc[ : , target])
    importancia_s = np.abs(lasso_secano.coef_)
    #print(importancia_s)
    
    # attr_31 = importancia_s.argsort()[-31]
    # humbral = importancia_s[attr_31] + 0.01

    for i in range(len(importancia_s)):
        if importancia_s[importancia_s.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
    
    attrs_s = importancia_s.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_s]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (secano 2015): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    
    
    # Grupo control 2016
    lasso_control = LassoCV(max_iter = 10000).fit(firma_control_2016, control_2016.loc[ : , target])
    importancia_c = np.abs(lasso_control.coef_)
    #print(importancia_c)
    
    # Buscaremos los 30 atributos más importantes
    # para estudiar su comportamiento
    # Fijamos el humbral sobre el atributo nro 31
    # attr_31 = importancia_c.argsort()[-31]
    # humbral = importancia_c[attr_31] + 0.01
    
    for i in range(len(importancia_c)):
        if importancia_c[importancia_c.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
    
    attrs_c = importancia_c.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_c]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (control 2016): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    # nota: luego de correr por primera vez el programa,
    # cantidad máxima tentativa de atributos: 37
    
    # Grupo secano 2016
    lasso_secano = LassoCV(max_iter = 10000).fit(firma_secano_2016, secano_2016.loc[ : , target])
    importancia_s = np.abs(lasso_secano.coef_)
    #print(importancia_s)
    
    # attr_31 = importancia_s.argsort()[-31]
    # humbral = importancia_s[attr_31] + 0.01
    
    
    for i in range(len(importancia_s)):
        if importancia_s[importancia_s.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
    
    attrs_s = importancia_s.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_s]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (secano 2016): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    
    
    # Grupo control 2017
    lasso_control = LassoCV(max_iter = 10000).fit(firma_control_2017, control_2017.loc[ : , target])
    importancia_c = np.abs(lasso_control.coef_)
    # print(importancia_c)
    
    # Buscaremos los 30 atributos más importantes
    # para estudiar su comportamiento
    # Fijamos el humbral sobre el atributo nro 31
    # attr_31 = importancia_c.argsort()[-31]
    #humbral = importancia_c[attr_31] + 0.01
    
    for i in range(len(importancia_c)):
        if importancia_c[importancia_c.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
    
    attrs_c = importancia_c.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_c]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (control 2017): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    # nota: luego de correr por primera vez el programa,
    # cantidad máxima tentativa de atributos: 37
    
    # Grupo secano 2017
    lasso_secano = LassoCV(max_iter = 10000).fit(firma_secano_2017, secano_2017.loc[ : , target])
    importancia_s = np.abs(lasso_secano.coef_)
    # print(importancia_s)
    
    # attr_31 = importancia_s.argsort()[-31]
    # humbral = importancia_s[attr_31] + 0.01
    
    
    for i in range(len(importancia_s)):
        if importancia_s[importancia_s.argsort()[::-1][i]] == 0 or i >= max:
            attr_n = i
            # print("indice del primer valor 0: " + str(attr_n))
            break
        
    attrs_s = importancia_s.argsort()[::-1][:attr_n]
    cols_aux = np.array(cols)[attrs_s]
    
    # convertir str a int
    cols_aux = cols_aux.astype(int)
        
    # ordenamos en orden ascendente
    cols_aux = np.sort(cols_aux)
    
    print("Atributos seleccionados (secano 2017): {}".format(cols_aux))
    print("Rangos: ", rangos_clustering(cols_aux))
    print("")
    return # fin lasso --------------------------------------------------------


def rec_feat_elim(): # no funcionando
    print("Ejecutando Recursive Feature Elimination...")
    # Grupo control 2014
    # create pipeline
    rfe = RFECV(estimator=DecisionTreeRegressor())
    model = DecisionTreeRegressor()
    pipeline = Pipeline(steps=[('s',rfe),('m',model)])
    # evaluate model
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(pipeline, firma_control_2014, control_2014.loc[:, target], scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f), cantidad: %d' % (mean(n_scores), std(n_scores), n_scores))

    
    
    # Grupo control 2014 ######################
    # rfecv = RFECV(estimator = DecisionTreeRegressor)
    # model = DecisionTreeRegressor()
    # pipeline = Pipeline(steps = [('s', rfecv), ('m', model)])
    # # evaluar el modelo
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # n_scores = cross_val_score(pipeline, firma_control_2014, control_2014.loc[ : , target], scoring='accuracy', cv = cv, n_jobs=-1, error_score='raise')
    # print('Nro óptimo de atributos: {}'.format(rfecv.n_features_))
    
    # grupo control 2014
    # nof_list = np.arange(1, 2150)
    # high_score = 0
    
    # nof = 0
    # score_list = []
    # for n in range(len(nof_list)):
    #     x_train, x_test, y_train, y_test = train_test_split(
    #         firma_control_2014,
    #         control_2014.loc[:, target],
    #         test_size = 0.3,
    #         random_state = 0
    #         )
    #     model = LinearRegression()
    #     rfe = RFE(model, nof_list[n])
    #     x_train_rfe = rfe.fit_transform(x_train, y_train)
    #     x_test_rfe = rfe.transform(x_test)
    #     model.fit(x_train_rfe, y_train)
    #     score = model.score(x_test_rfe, y_test)
    #     score_list.append(score)
    #     if score > high_score:
    #         high_score = score
    #         nof = nof_list[n]
            
    # print("Número óptimo de atributos: %d" %nof)
    # print("Puntaje con %d atributos: %f" % (nof, high_score))
    return

def kbest_corr():
    print("Ejecutando SelectK-Best (Correlation)...")
    # my_k = math.ceil(2150*0.02)
    my_k = 'all'
    # grupo control 2014 #####################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2014,
                                                        control_2014.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2014: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2014 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    
    # grupo secano 2014 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2014,
                                                        secano_2014.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2014: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2014 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    
    # grupo control 2015 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2015,
                                                        control_2015.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2015: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2015 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    
    # grupo secano 2015 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2015,
                                                        secano_2015.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2015: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2015 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    
    # grupo control 2016 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2016,
                                                        control_2016.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2016: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2016 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    
    # grupo secano 2016 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2016,
                                                        secano_2016.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2016: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2016 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    
    # grupo control 2017 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2017,
                                                        control_2017.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2017: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2017 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    
    # grupo secano 2017 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2017,
                                                        secano_2017.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=f_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2017: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2017 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Correlación")
    plt.show()
    return # fin k-best-------------------------------------------------------

def kbest_mi():
    print("Ejecutando SelectK-Best (Mutual Information)...")
    my_k = math.ceil(2150*0.015)
    # my_k = 'all'
    # grupo control 2014 #####################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2014,
                                                        control_2014.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2014: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    # print("tipo datos rangos:", type(elegidos)) # tipo de variable
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2014 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    # grupo secano 2014 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2014,
                                                        secano_2014.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2014: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2014 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    # grupo control 2015 #####################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2015,
                                                        control_2015.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2015: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2015 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    # grupo secano 2015 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2015,
                                                        secano_2015.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2015: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2015 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    # grupo control 2016 #####################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2016,
                                                        control_2016.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2016: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2016 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    # grupo secano 2016 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2016,
                                                        secano_2016.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2016: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2016 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    # grupo control 2017 #####################
    x_train, x_test, y_train, y_test = train_test_split(firma_control_2017,
                                                        control_2017.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo control 2017: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo control 2017 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    # grupo secano 2017 ######################
    x_train, x_test, y_train, y_test = train_test_split(firma_secano_2017,
                                                        secano_2017.loc[:, target],
                                                        test_size=0.33,
                                                        random_state=1)
    
    f_selector = SelectKBest(score_func=mutual_info_regression, k=my_k)
    f_selector.fit(x_train, y_train)
    train_fs = f_selector.transform(x_train)
    test_fs = f_selector.transform(x_test)
    
    # elegidos = f_selector.get_support(True)
    
    elegidos = calcular_75_sup(f_selector)
    
    for i in range(len(elegidos)):
        elegidos[i] = elegidos[i] + 350
        
    print("Grupo secano 2017: ", elegidos)
    print("Rangos: ", rangos_clustering(elegidos))
    print("")
    
    # plot
    plt.bar([i + 350 for i in range(len(f_selector.scores_))], f_selector.scores_)
    plt.title("KBest - Grupo secano 2017 (" + target + ")")
    plt.xlabel("Longitud de onda")
    plt.ylabel("Importancia por Información Mutua")
    plt.show()
    
    return

def ga():
    print("Ejecutando Genetic Algorithm...")
    # Grupo control 2014 ###################################
    x = firma_control_2014
    y = control_2014.loc[:, target]
    model = linear_model.LinearRegression()
    ff = CustomFitnessFunctionClass(n_total_features=x.shape[1], 
                                    n_splits=3,
                                    alpha=0.05)
    fsga = FeatureSelectionGA(model, x, y, ff_obj = ff)
    pop = fsga.generate(5000)
    print(pop)
    
    return

def calcular_75_sup(lista_importancia):
    #lista_importancia.describe()
    mediana = (np.max(lista_importancia.scores_)) / 2
    print("max: " + str(np.max(lista_importancia.scores_)))
    print("mediana: " + str(mediana))
    val_75_sup = mediana + (mediana * 0.5)
    print("75%: " + str(val_75_sup))
    lista_75_sup = []
    for i in range(len(lista_importancia.scores_)):
        if lista_importancia.scores_[i] >= val_75_sup:
            lista_75_sup.append(i)
    return lista_75_sup

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
        lasso()
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
        
    elif op == '3':
        start = time.perf_counter()
        kbest_mi()
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
    
    elif op == '4':
        start = time.perf_counter()
        kbest_corr()
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
        
    elif op == '5':
        start = time.perf_counter()
        ga()
        end = time.perf_counter()
        print(f"Tiempo de ejecución: {end - start:0.2f} segundos.")
        
    elif op == '6':
        start = time.perf_counter()
        #boruta()
        lasso()
        kbest_mi()
        kbest_corr()
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
















