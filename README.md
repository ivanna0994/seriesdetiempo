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

## Visualización de la media y varianza en submuestras
Esto con el fin de evaluar si cambian con el tiempo.

![Media y varianza submuestras](https://github.com/ivanna0994/seriesdetiempo/blob/main/media%20varianza%20submuestras.png?raw=true "Media y varianza submuestras")

Se observa un aumento progresivo en la media de la temperatura a medida que avanzamos en el tiempo. El primer segmento tiene una media más baja, mientras que los segmentos posteriores muestran valores crecientes. Esto indica que la serie presenta tendencia ascendente, lo que sugiere que no es estacionaria en media.

La varianza disminuye en los segmentos posteriores en comparación con el primer segmento. La varianza cambia con el tiempo, esto indica que la serie también podría no ser estacionaria en varianza.

Según el análisis previo, la serie no es estacionaria porque tiene una tendencia creciente en la media y una varianza cambiante

La tendencia en la serie sugiere que debemos aplicar diferenciación, que consiste en restar el valor actual con el valor anterior.

# Serie diferenciada

![Serie Diferenciada](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/serie%20de%20tiempo%20diferenciada.png)


Si la varianza cambia con el tiempo, podemos aplicar una transformación 

Hay picos extremos negativos que podrían indicar outliers, pero no parecen formar patrones persistentes de cambio de varianza. Visualmente, no se detecta heterocedasticidad clara (es decir, no hay un ensanchamiento o estrechamiento progresivo de la amplitud de la serie).

# Transformación de los datos

![Transformación logarítmica](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/Transformaci%C3%B3n.png)


Después de estos ajustes, debemos volver a aplicar la prueba Dickey-Fuller (ADF) para verificar si la serie ya es estacionaria.

# Prueba Dickey - Fuller 
# Estadístico ADF: -6.735583526083349
p-valor: 3.21385243908857e-09
Valores críticos: {'1%': -3.4305057071359926, '5%': -2.8616088190674343, '10%': -2.5668066301638324}

El estadístico ADF (-6.7356) es más bajo que todos los valores críticos (1%, 5%, 10%).
La serie es estacionaria. 
Dado esto podemos: 
Modelar con SARIMA o ARIMA sin diferenciación adicional.
Utilizar la serie directamente en LSTM, como ya hicimos.
Incluir otras variables climáticas (humedad, presión, viento) si quieres extender el modelo a una versión multivariada.

# Matriz de Correlación entre Series Temporales
![Matriz de correlación](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/Matriz%20de%20correlaci%C3%B3n.png)

En la matriz de correlación presentada en la imagen, se puede observar que la variable Tdew (degC) tiene una alta correlación positiva con T (degC), con un valor de 0.90. Esto indica que tanto la temperatura del aire (T) como la temperatura de rocío (Tdew) están fuertemente relacionadas.

También se destaca que T (degC) tiene una correlación positiva significativa con las variables Tpot (K) (0.89), VPmax (mbar) (0.95), y VPact (mbar) (0.87). Estas relaciones sugieren que las variables relacionadas con la humedad y la presión del aire (como VPmax y VPact) están fuertemente influenciadas por la temperatura.

Por último, es relevante mencionar que T (degC) está correlacionada negativamente con la variable rho (g/m³), con un valor de -0.96, lo que indica que, a medida que la temperatura aumenta, la densidad del aire disminuye.

# Gráficos de Dispersión en el Tiempo
Estos gráficos los realizaremos con el objetivo de detectar relaciones no lineales entre la temperatura y otras variables.

![Dispersión](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/Gr%C3%A1ficos%20de%20dispersi%C3%B3n.png)

En los gráficos se puede observar que la variable T (degC) (Temperatura) tiene varias relaciones no lineales con otras variables meteorológicas. A continuación, se analizan algunas de estas relaciones:

Temperatura vs Humedad (%): La relación entre la temperatura y la humedad relativa parece ser no lineal. A medida que la temperatura aumenta, la humedad tiende a disminuir, pero de manera más pronunciada en los valores más bajos de temperatura. Esta curva sugiere que la humedad relativa disminuye rápidamente con el aumento de la temperatura, lo que es común en las condiciones meteorológicas donde el aire más caliente puede contener más vapor de agua.

Temperatura vs Presión (mbar): La relación entre temperatura y presión también parece ser no lineal, con una ligera disminución de la presión conforme la temperatura aumenta, pero la pendiente de la curva varía a lo largo de los valores de temperatura. Esta relación refleja cómo la temperatura afecta a la densidad del aire y, por ende, a la presión atmosférica, aunque la tendencia no es tan estrictamente lineal.

Temperatura vs Velocidad del Viento (wv): En este gráfico, se observa una fuerte dispersión de los valores de velocidad del viento en temperaturas tanto frías como cálidas. La mayoría de los valores se concentran cerca de cero, con algunos valores extremos negativos y positivos en temperaturas bajas. Esto indica que no hay una relación clara y lineal entre la temperatura y la velocidad del viento en los datos observados, posiblemente por la presencia de datos atípicos.

Temperatura vs Presión de Vapor (Vpact, mbar): La relación entre la temperatura y la presión de vapor parece seguir una curva exponencial. A medida que la temperatura aumenta, la presión de vapor aumenta de manera más pronunciada, lo que refleja la mayor capacidad del aire para retener vapor de agua a temperaturas más altas. Esta relación es un ejemplo típico de un comportamiento no lineal, ya que la presión de vapor aumenta de forma acelerada con la temperatura.

En resumen, varias de las relaciones entre la temperatura y otras variables son no lineales, lo que sugiere que los cambios en las variables meteorológicas no siguen una simple regla lineal, sino que presentan un comportamiento más complejo y en algunos casos exponencial, especialmente en lo que respecta a la presión de vapor.

# Análisis de la Transformada de Fourier

Procedemos a realizar este análisis con el fin de detectar frecuencias dominantes en la serie.

![Transformada](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/Transformada%20fourier.png)

Gráfica Transformada de Fourier Después del pico inicial, la magnitud disminuye rápidamente a medida que aumenta la frecuencia. Esto indica que las componentes de alta frecuencia de la señal tienen una magnitud mucho menor, lo que puede sugerir que la señal es relativamente suave y no presenta fluctuaciones rápidas.

Este espectro de frecuencia muestra que la señal tiene una fuerte componente de baja frecuencia, lo que implica que las fluctuaciones importantes de la señal ocurren a una escala temporal más larga (posiblemente ciclos estacionales o de largo plazo). Las componentes de alta frecuencia tienen poca magnitud, lo que indica que las fluctuaciones rápidas son menos significativas.

Espectro de potencia de la temperatura

El gráfico muestra un pico muy pronunciado en una frecuencia específica (cerca de 0.005 ciclos por día), lo que podría indicar que existe una componente de baja frecuencia dominando la variabilidad de la temperatura. Este pico podría estar asociado con patrones cíclicos, como las fluctuaciones diarias o estacionales de la temperatura.

El espectro de potencia de la temperatura muestra que las variaciones de temperatura se concentran principalmente en frecuencias bajas, lo que indica que los cambios más significativos de temperatura ocurren a una escala temporal más larga, como los ciclos diarios o estacionales.

# Detección de picos inusuales en la temperatura

![Picos inusuales](https://raw.githubusercontent.com)/ivanna0994/seriesdetiempo/blob/main/Picos%20inusuales.png "Picos inusuales")

Los picos rojos en la gráfica indican eventos donde la temperatura alcanza valores extremos, como picos inusuales de calor o frío que no coinciden con el patrón cíclico habitual. Es posible que estos picos rojos representen eventos como olas de calor o heladas atípicas. La serie temporal muestra una periodicidad evidente, probablemente debida a variaciones estacionales (calor en verano y frío en invierno), mientras que las anomalías corresponden a eventos que se desvían de esta periodicidad. En resumen, la gráfica muestra cómo la temperatura varía a lo largo del tiempo, con algunos valores atípicos o extremos marcados como anomalías, lo que permite identificar eventos climáticos excepcionales.

# Prueba para detectar outliers

![Outliers](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/blob/main/Outliers.png "Outliers")

Rango central (IQR): La mayoría de los datos de temperatura se encuentran dentro del rango entre -10°C y 30°C, con la mediana cerca de los 10°C. Outliers: Los puntos fuera de los bigotes, ubicados por encima de 30°C y por debajo de -20°C, son considerados outliers. Estos valores son inusuales y podrían indicar fenómenos extremos o errores en los datos. Distribución de los datos: La temperatura tiene una distribución que se centra principalmente alrededor de la mediana (aproximadamente 10°C), con algunas fluctuaciones hacia valores más bajos y más alto.

