# Deteccion de rango de atributos relevantes en datos hiperespectrales

### Resumen
En este proyecto se utilizan algoritmos de selección de atributos para optimizar la lectura de firma espectral de plantas de trigo para alguna variable objetivo.


### Los datos
Los datos se encuentran en el archivo **data-total.csv**. Cuenta con mediciones de variables fisiológicas y firma espectral de plantas de trigo en distintas etapas de madurez. 
La firma espectral en este caso, registra la reflectancia de la hoja entre las logitudes de onda 350-2500nm medidas durante 4 años (2014-2017).

### Los algoritmos
Se han implementado 4 algoritmos de selección de atributos, divididos en módulos. A continuación se listan los módulos y los algoritmos que contiene cada uno:
* my_boruta.py: Algoritmo Boruta.
* my_lasso.py: Algoritmo Least absolute shrinkage and selection operator (LASSO).
* my_kbest.py: Algoritmo Select K-Best, utilizando dos funciones de score,
  * Mutual information: Función kbest_mi().
  * Correlation: Función kbest_mi().
* my_ga.py: Algoritmo genético para selección de atributos.
