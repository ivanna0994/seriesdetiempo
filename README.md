# Detección de Patrones Climáticos a través del Análisis de Series Temporales de Temperatura utilizando modelos [opción escogida]

El conjunto de datos Jena Climate es una serie temporal meteorológica registrada en la estación meteorológica del Instituto Max Planck de Bioquímica en Jena, Alemania.
Este se encuentra compuesto por 14 variables diferentes (como la temperatura del aire, la presión atmosférica, la humedad, la dirección del viento, entre otras) que fueron registradas cada 10 minutos durante varios años.

Este conjunto de datos abarca información desde el 1 de enero de 2009 hasta el 31 de diciembre de 2016

````python
#Cargamos librerias
import pandas as pd
import numpy as np
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import shapiro
from IPython import get_ipython
from IPython.display import display`
````
## Leyendo el archivo y cargando nuestros datos 
```python
# Cargamos datos
file_path = '/content/jena_climate_2009_2016.csv'
df = pd.read_csv(file_path)
print(df.info())
```
Para verificar sí existen o no datos faltantes, se procede a realizar la consulta a nuestra base de datos:

```
print(df.isnull().sum())
```
![Null data](https://github.com/ivanna0994/seriesdetiempo/blob/main/Null%20datos.png?raw=true "Null data")

Encontrándose así que no existen datos nulos. 

El dataset contiene datos climáticos con registros cada 10 minutos. La variable "Date Time" está en formato de texto y debe convertirse a formato de fecha y hora. La variable de interés "T (degC)" representa la temperatura en grados Celsius. Nuestro análisis se centrará en las variables, día, mes y año y hora en que fue tomada la temperatura, el cual se encuentra en el conjunto de datos bajo el nombre de 'Date Time' así como también la variable temperatura que contiene los valores de esta, y en el conjunto de datos se encuentra como T(degC).

```python
# Convertir la columna "Date Time" a formato datetime
df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")

# Verificar el rango de fechas en el dataset
date_range = df["Date Time"].min(), df["Date Time"].max()
```

Ya hecha la transformación de la variable Date time, se realiza una breve descripción estadística de la variable a estudiar, que en este caso es Temperatura medida en grados celsius.

```python
# Descripción estadística de la variable T (degC)
temperature_stats = df['T (degC)'].describe()
print(temperature_stats)
```

![stats temperature](https://github.com/ivanna0994/seriesdetiempo/blob/main/temperature%20stats.png?raw=true "stats temperature")

Como se mencionó anteriormente, para la variable Temperatura en grados celsius, se tienen 420551 observaciones, con una media de 9.45° y una desviación estándar de 8.42°. Dentro de estas observaciones el valor mínimo que toma esta variable es de -23.01° mientras que su valor máximo es de 37.28°. Es importante señalar que el cincuenta por ciento de las temperaturas registradas se encuentra por encima de 9.42°.

Procedemos a construir un nuevo dataset con las variables de interés mencionadas anteriormente.

```python
data=df[['Date Time','T (degC)']]
print('Column names:', data.columns)
print('Numero de filas y columnas: ', data.shape)
```
Procedemos a construir un nuevo dataset con las variables de interés mencionadas anteriormente.


```python
data=df[['Date Time','T (degC)']]
data
```
![data head ](https://github.com/ivanna0994/seriesdetiempo/blob/main/data%20head.png?raw=true "data head ")

# Graficamos la Distribución de la Temperatura en un histograma
```python
import matplotlib.pyplot as plt

# Histograma de la temperatura
plt.figure(figsize=(8, 5))
sns.histplot(df["T (degC)"], bins=50, kde=True, color="blue")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Frecuencia")
plt.title("Distribución de la Temperatura")
plt.show()
```
![Distribución de la temperatura](https://github.com/ivanna0994/seriesdetiempo/blob/main/distribuci%C3%B3n%20de%20la%20temperatura.png?raw=true "Distribución de la temperatura")

La forma de la distribución es aproximadamente normal, lo que sugiere que la temperatura varía de manera continua alrededor de un rango típico (5°C a 15°C).
Sin embargo, la presencia de valores negativos y extremos indica que hay variaciones estacionales o eventos climáticos que afectan la temperatura a lo largo del tiempo.

Realizaremos una prueba de Shapiro-Wilks para evaluar si la distribución de la temperatura sigue una distribución normal.

```python
from scipy.stats import shapiro
print('Shapiro-Wilk Test:')
stat, p = shapiro(data['T (degC)'])
print('Statistics=%.6f, p=%e' % (stat, p))

alpha = 0.05
if p > alpha:
 print('Sample looks Normal (fail to reject H0)')
else:
 print('Sample does not look Normal (reject H0)')
```
![Prueba Shapiro](https://github.com/ivanna0994/seriesdetiempo/blob/main/shapiro-wilks.png?raw=true "Prueba Shapiro")

La hipótesis nula del test establece que los datos siguen una distribución normal.
Dado que el p-valor es extremadamente pequeño (< 0.05), rechazamos la hipótesis nula.
Esto indica que la distribución de la temperatura no es normal, lo que sugiere que tiene sesgo, colas largas o alguna otra forma de desviación de la normalidad.
La distribución sugiere una clara variabilidad de temperatura con una tendencia central entre 5°C y 15°C.
El patrón cíclico probablemente indica estacionalidad anual, lo cual debe confirmarse con un análisis más profundo.
Los eventos extremos pueden influir en los modelos de predicción, por lo que deben tratarse adecuadamente (remoción o ajuste con modelos robustos).
Para hacer un análisis más detallado, se recomienda aplicar métodos de descomposición y autocorrelación para visualizar la dinámica temporal.
