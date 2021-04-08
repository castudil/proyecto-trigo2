# Proyecto de titulación de tesis

### Resumen
En este proyecto se utilizan algoritmos de selección de atributos para optimizar la lectura de firma espectral de plantas de trigo para alguna variable objetivo **target**

procesamiento de los datos, como se obtuvieron. donde estan los datos originales.

### Los datos

filas columnas, descirpcion. origen, plantaciones, predios ubicaciones, tiempos. como se tomaron los datos.

Los datos se encuentran en el archivo **data-total.csv**. Cuenta con mediciones de variables fisiológicas y firma espectral de plantas de trigo en distintas etapas de madurez. 
La firma espectral en este caso, registra la reflectancia de la hoja entre las logitudes de onda 350-2500nm medidas durante 4 años (2014-2017).

### Algoritmos de seleccion de atributos

Se han implementado 4 algoritmos de selección de atributos, divididos en distintos archivos. El nombre de cada archivo corresponde al nombre del algortimo. A continuación se listan los módulos y los algoritmos que contiene cada uno:

1. boruta.py: Algoritmo Boruta. Agregar que es lo aque lo hace distinto a los otros. referecnias a parper o website o libros.
2. lasso.py: Algoritmo Least absolute shrinkage and selection operator (LASSO). LO MISMO, a lo mejor un link a youtube.
3. my_kbest.py: Algoritmo Select K-Best, utilizando dos funciones de score, LO MISMO
  * Mutual information: Función kbest_mi().
  * Correlation: Función kbest_mi().
4. ga.py: Algoritmo genético para selección de atributos. LO MISMO, PENDIENTE.
