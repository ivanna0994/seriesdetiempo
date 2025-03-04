# Predicción de la Temperatura del Aire: Comparación y Optimización de Modelos de Series de Tiempo y Aprendizaje Automático

Pregunta de investigación:
¿Cuál es la estrategia más precisa y eficiente para predecir la temperatura del aire: modelos de series de tiempo que capturan autocorrelaciones o modelos de regresión basados en variables climáticas? Además, ¿es posible desarrollar un modelo mejorado que optimice la precisión de las predicciones mediante técnicas de series de tiempo y aprendizaje automático?
 
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
Los eventos extremos pueden influir en los modelos de predicción, por lo que deben tratarse adecuadamente.
Para hacer un análisis más detallado procedemos a aplicar métodos de descomposición y autocorrelación para visualizar la dinámica temporal.

## Descomposición de la Serie de tiempo de la Temperatura

![Descomposición serie de tiempo temperatura](https://github.com/ivanna0994/seriesdetiempo/blob/main/download.png?raw=true "Descomposición serie de tiempo temperatura")

### Serie Original (Gráfico 1 - Negro)
Se observa un patrón repetitivo con fluctuaciones estacionales.
La temperatura presenta altibajos anuales, lo que sugiere una clara estacionalidad relacionada con las estaciones del año. Hay períodos de temperaturas altas y bajas que parecen mantenerse en ciclos regulares.

### Tendencia (Gráfico 2 - Azul)

La temperatura muestra una ligera tendencia al alza a partir de 2013, lo que podría indicar un posible calentamiento a largo plazo. Entre 2009 y 2012, la tendencia parece más estable o ligeramente decreciente. Esto puede estar influenciado por cambios climáticos o patrones locales de variabilidad.

### Estacionalidad (Gráfico 3 - Verde)

Se observan ciclos claramente repetitivos cada año, lo que confirma la estacionalidad en la serie. La temperatura sigue un patrón estacional anual, con picos en meses cálidos y valles en meses fríos. Este comportamiento es característico del clima en regiones con estaciones bien definidas.

### Componente Aleatoria (Ruido) (Gráfico 4 - Rojo)

Representa las variaciones no explicadas por la tendencia ni la estacionalidad.
Hay fluctuaciones más pronunciadas en ciertos períodos, lo que puede estar relacionado con eventos climáticos extremos, olas de calor o frío. Este ruido puede afectar la precisión de modelos de predicción si no se maneja adecuadamente.

La serie de temperatura presenta un claro comportamiento estacional con ciclos anuales. Existe una tendencia creciente después de 2013, lo que podría sugerir un fenómeno de calentamiento progresivo. El componente aleatorio muestra fluctuaciones, lo que indica que además de la estacionalidad, hay variaciones impredecibles en los datos.

Continuemos nuestro análisis con un análisis de autocorrelación

## Análisis de autocorrelación

```python
import statsmodels.api as sm

# Graficar la Función de Autocorrelación (ACF)
plt.figure(figsize=(12, 5))
sm.graphics.tsa.plot_acf(df_daily_interpolated, lags=365, alpha=0.05)
plt.title("Función de Autocorrelación (ACF) de la Temperatura")
plt.show()

# Graficar la Función de Autocorrelación Parcial (PACF)
plt.figure(figsize=(12, 5))
sm.graphics.tsa.plot_pacf(df_daily_interpolated, lags=40, alpha=0.05)
plt.title("Función de Autocorrelación Parcial (PACF) de la Temperatura")
plt.show()
```
![Función de autocorrelaicón de la temperatura](https://github.com/ivanna0994/seriesdetiempo/blob/main/funci%C3%B3n%20autocorrelaci%C3%B3n.png?raw=true "Función de autocorrelaicón de la temperatura")

![Función autocorrelación parcial](https://github.com/ivanna0994/seriesdetiempo/blob/main/Funci%C3%B3n%20autocorrelaci%C3%B3n%20parcial.png?raw=true "Función autocorrelación parcial")

## Función de Autocorrelación (ACF)

Se observa un patrón fuertemente periódico, con picos de correlación en torno a 365 días.
Esto confirma la presencia de estacionalidad anual, lo cual es esperable en una serie de temperatura. También se nota una tendencia decreciente, lo que indica que los valores pasados influyen en los futuros, pero esta influencia disminuye con el tiempo.

## Función de Autocorrelación Parcial (PACF)

Se observa un pico fuerte en el primer lag, lo que indica que la temperatura de un día está fuertemente influenciada por la del día anterior.
También hay picos en los retrasos correspondientes a múltiples de 365 días, lo que confirma la estacionalidad anual.

La serie tiene una fuerte componente estacional con un ciclo anual (365 días), los valores pasados influyen en los futuros, especialmente en los primeros lags.


## Prueba de Estacionariedad Dickey-Fuller Aumentado (ADF)

Se realiza esta prueba para determinar si la serie necesita diferenciación

```python
from statsmodels.tsa.stattools import adfuller

# Aplicar la prueba de Dickey-Fuller Aumentada (ADF)
adf_test = adfuller(df_daily_interpolated.dropna())

# Extraer los resultados
adf_statistic = adf_test[0]
p_value = adf_test[1]
critical_values = adf_test[4]

# Mostrar los resultados
adf_results = {
    "Estadístico ADF": adf_statistic,
    "Valor p": p_value,
    "Valores Críticos": critical_values,
}

adf_results
```
![Test Dickey-Fuller ](https://github.com/ivanna0994/seriesdetiempo/blob/main/dickey%20fuller.png?raw=true "Test Dickey-Fuller ")

La hipótesis nula (H0) en la prueba de Dickey-Fuller establece que la serie no es estacionaria (tiene tendencia o variaciones dependientes del tiempo).
Como el estadístico ADF (-3.59) es menor que el valor crítico al 5% (-2.86) y el p-value (0.006) es menor que 0.05, rechazamos la hipótesis nula.
Esto indica que la serie es estacionaria a un nivel de significancia del 5%.

#Visualización de la media y varianza en submuestras
Esto con el fin de evaluar si cambian con el tiempo.

