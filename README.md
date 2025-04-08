<h1 align="center">🌡️ Predicción de la Temperatura del Aire</h1>
<h3 align="center">Comparación y Optimización de Modelos de Series de Tiempo y Aprendizaje Automático</h3>



---

### 📌 Descripción del Proyecto

Este proyecto analiza diferentes estrategias para predecir la temperatura del aire a partir de datos meteorológicos registrados por la estación del Instituto Max Planck de Bioquímica en Jena, Alemania.  

Se busca responder:  
**¿Cuál es la estrategia más precisa y eficiente para predecir la temperatura del aire: modelos de series de tiempo que capturan autocorrelaciones o modelos de regresión basados en variables climáticas?**

También se explora la posibilidad de desarrollar un modelo mejorado que combine ambos enfoques: **series de tiempo + aprendizaje automático**.

---

### 📊 Conjunto de Datos

- **Nombre:** Jena Climate
- **Periodo:** 2009-01-01 a 2016-12-31
- **Frecuencia:** cada 10 minutos
- **Variables:** temperatura del aire, presión, humedad, dirección del viento, entre otras (14 en total)

---

````python
# 🔧 Manipulación de datos
import pandas as pd
import numpy as np

# 🎨 Visualización
import seaborn as sns

# 📊 Descomposición estacional de series temporales
from statsmodels.tsa.seasonal import seasonal_decompose

# 📈 Modelos estadísticos
import statsmodels.api as sm

# 🧪 Pruebas estadísticas
from scipy.stats import shapiro

# 💻 Herramientas de entorno interactivo (Jupyter)
from IPython import get_ipython
from IPython.display import display

````
## Leyendo el archivo y cargando nuestros datos 
```python
# 📁 Definir la ruta del archivo CSV
file_path = '/content/jena_climate_2009_2016.csv'

# 📄 Cargar los datos en un DataFrame
df = pd.read_csv(file_path)

# 🔍 Mostrar información general del DataFrame
print(df.info())

```
Para verificar sí existen o no datos faltantes, se procede a realizar la consulta a nuestra base de datos:

```
print(df.isnull().sum())
```
![null data](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/nulos.png?raw=true)


Encontrándose así que no existen datos nulos. 

El dataset contiene datos climáticos con registros cada 10 minutos. La variable "Date Time" está en formato de texto y debe convertirse a formato de fecha y hora. La variable de interés "T (degC)" representa la temperatura en grados Celsius. Nuestro análisis se centrará en las variables, día, mes y año y hora en que fue tomada la temperatura, el cual se encuentra en el conjunto de datos bajo el nombre de 'Date Time' así como también la variable temperatura que contiene los valores de esta, y en el conjunto de datos se encuentra como T(degC).

```python
# 🗓️ Convertir la columna "Date Time" a formato datetime
df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")

# 📆 Verificar el rango de fechas en el dataset
date_range = df["Date Time"].min(), df["Date Time"].max()
print(f"Rango temporal del dataset: {date_range[0]} → {date_range[1]}")

```

Ya hecha la transformación de la variable Date time, se realiza una breve descripción estadística de la variable a estudiar, que en este caso es Temperatura medida en grados celsius.

```python
# 📊 Descripción estadística de la variable "T (degC)"
temperature_stats = df["T (degC)"].describe()

# 🖨️ Mostrar medidas como media, desviación estándar, mínimos y cuartiles
print(temperature_stats)

```

![Estadísticas de temperatura](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/estad%C3%ADsticas.png?raw=true)


Como se mencionó anteriormente, para la variable Temperatura en grados celsius, se tienen 420551 observaciones, con una media de 9.45° y una desviación estándar de 8.42°. Dentro de estas observaciones el valor mínimo que toma esta variable es de -23.01° mientras que su valor máximo es de 37.28°. Es importante señalar que el cincuenta por ciento de las temperaturas registradas se encuentra por encima de 9.42°.

Procedemos a construir un nuevo dataset con las variables de interés mencionadas anteriormente.

```python
# 🧾 Crear un nuevo DataFrame solo con la fecha y la temperatura
data = df[["Date Time", "T (degC)"]]

# 🔍 Verificar los nombres de las columnas seleccionadas
print("Column names:", data.columns)

# 📐 Mostrar el número de filas y columnas
print("Número de filas y columnas:", data.shape)

```
Procedemos a construir un nuevo dataset con las variables de interés mencionadas anteriormente.


```python
data=df[['Date Time','T (degC)']]
data
```
![Vista previa de los datos](/figuras/datahead.png)


### 📈 Distribución de la Temperatura
```python
import matplotlib.pyplot as plt

# 📈 Histograma de la temperatura con curva de densidad (KDE)
plt.figure(figsize=(8, 5))
sns.histplot(df["T (degC)"], bins=50, kde=True, color="blue")

# 🏷️ Etiquetas y título
plt.xlabel("Temperatura (°C)")
plt.ylabel("Frecuencia")
plt.title("Distribución de la Temperatura")
plt.show()

```

![Distribución de la temperatura](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/distribución%20de%20la%20temperatura.png) 

La forma de la distribución es aproximadamente normal, lo que sugiere que la temperatura varía de manera continua alrededor de un rango típico (5°C a 15°C).
Sin embargo, la presencia de valores negativos y extremos indica que hay variaciones estacionales o eventos climáticos que afectan la temperatura a lo largo del tiempo.

Realizaremos una prueba de Shapiro-Wilks para evaluar si la distribución de la temperatura sigue una distribución normal.

```python
from scipy.stats import shapiro

print("Shapiro-Wilk Test:")

# 🔍 Aplicar test de normalidad a la temperatura
stat, p = shapiro(data["T (degC)"])

# 📊 Mostrar estadístico y valor p
print("Statistics = %.6f, p = %e" % (stat, p))

# 🧠 Interpretación
alpha = 0.05
if p > alpha:
    print("✅ La muestra parece normal (no se rechaza H0)")
else:
    print("❌ La muestra no parece normal (se rechaza H0)")

```
![Prueba Shapiro](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/shapiro%20test.png?raw=true)

La hipótesis nula del test establece que los datos siguen una distribución normal.
Dado que el p-valor es extremadamente pequeño (< 0.05), rechazamos la hipótesis nula.
Esto indica que la distribución de la temperatura no es normal, lo que sugiere que tiene sesgo, colas largas o alguna otra forma de desviación de la normalidad.
La distribución sugiere una clara variabilidad de temperatura con una tendencia central entre 5°C y 15°C.
El patrón cíclico probablemente indica estacionalidad anual, lo cual debe confirmarse con un análisis más profundo.
Los eventos extremos pueden influir en los modelos de predicción, por lo que deben tratarse adecuadamente.
Para hacer un análisis más detallado procedemos a aplicar métodos de descomposición y autocorrelación para visualizar la dinámica temporal.

## 🔎 Descomposición de la Serie de tiempo de la Temperatura

![Descomposición serie de tiempo temperatura](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/download.png?raw=true "Descomposición serie de tiempo temperatura")

###   Serie Original (Gráfico 1 - Negro)
Se observa un patrón repetitivo con fluctuaciones estacionales.
La temperatura presenta altibajos anuales, lo que sugiere una clara estacionalidad relacionada con las estaciones del año. Hay períodos de temperaturas altas y bajas que parecen mantenerse en ciclos regulares.

###   Tendencia (Gráfico 2 - Azul)

La temperatura muestra una ligera tendencia al alza a partir de 2013, lo que podría indicar un posible calentamiento a largo plazo. Entre 2009 y 2012, la tendencia parece más estable o ligeramente decreciente. Esto puede estar influenciado por cambios climáticos o patrones locales de variabilidad.

###  Estacionalidad (Gráfico 3 - Verde)

Se observan ciclos claramente repetitivos cada año, lo que confirma la estacionalidad en la serie. La temperatura sigue un patrón estacional anual, con picos en meses cálidos y valles en meses fríos. Este comportamiento es característico del clima en regiones con estaciones bien definidas.

###  Componente Aleatoria (Ruido) (Gráfico 4 - Rojo)

Representa las variaciones no explicadas por la tendencia ni la estacionalidad.
Hay fluctuaciones más pronunciadas en ciertos períodos, lo que puede estar relacionado con eventos climáticos extremos, olas de calor o frío. Este ruido puede afectar la precisión de modelos de predicción si no se maneja adecuadamente.

La serie de temperatura presenta un claro comportamiento estacional con ciclos anuales. Existe una tendencia creciente después de 2013, lo que podría sugerir un fenómeno de calentamiento progresivo. El componente aleatorio muestra fluctuaciones, lo que indica que además de la estacionalidad, hay variaciones impredecibles en los datos.

Continuemos nuestro análisis con un análisis de autocorrelación

### 📈 Autocorrelación (ACF y PACF)

```python
import statsmodels.api as sm

# 📊 ACF: Función de Autocorrelación (365 lags ~ 1 año)
plt.figure(figsize=(12, 5))
sm.graphics.tsa.plot_acf(df_daily_interpolated, lags=365, alpha=0.05)
plt.title("Función de Autocorrelación (ACF) de la Temperatura")
plt.xlabel("Rezagos (días)")
plt.ylabel("Autocorrelación")
plt.show()

# 🔍 PACF: Función de Autocorrelación Parcial (primeros 40 rezagos)
plt.figure(figsize=(12, 5))
sm.graphics.tsa.plot_pacf(df_daily_interpolated, lags=40, alpha=0.05)
plt.title("Función de Autocorrelación Parcial (PACF) de la Temperatura")
plt.xlabel("Rezagos (días)")
plt.ylabel("Autocorrelación parcial")
plt.show()

```
![Función de autocorrelaicón de la temperatura](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/funci%C3%B3n%20autocorrelaci%C3%B3n.png?raw=true "Función de autocorrelaicón de la temperatura")

![Función autocorrelación parcial](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Funci%C3%B3n%20autocorrelaci%C3%B3n%20parcial.png?raw=true "Función autocorrelación parcial")

## Función de Autocorrelación (ACF)

Se observa un patrón fuertemente periódico, con picos de correlación en torno a 365 días.
Esto confirma la presencia de estacionalidad anual, lo cual es esperable en una serie de temperatura. También se nota una tendencia decreciente, lo que indica que los valores pasados influyen en los futuros, pero esta influencia disminuye con el tiempo.

## Función de Autocorrelación Parcial (PACF)

Se observa un pico fuerte en el primer lag, lo que indica que la temperatura de un día está fuertemente influenciada por la del día anterior.
También hay picos en los retrasos correspondientes a múltiples de 365 días, lo que confirma la estacionalidad anual.

La serie tiene una fuerte componente estacional con un ciclo anual (365 días), los valores pasados influyen en los futuros, especialmente en los primeros lags.


### 🧪 Test de Estacionariedad (ADF)

Se realiza esta prueba para determinar si la serie necesita diferenciación

```python
from statsmodels.tsa.stattools import adfuller

# 🧪 Aplicar la prueba de Dickey-Fuller Aumentada (ADF)
adf_test = adfuller(df_daily_interpolated.dropna())

# 📋 Extraer los resultados
adf_statistic = adf_test[0]
p_value = adf_test[1]
critical_values = adf_test[4]

# 📊 Mostrar resultados organizados
adf_results = {
    "Estadístico ADF": adf_statistic,
    "Valor p": p_value,
    "Valores Críticos": critical_values,
}

adf_results

```
![Test Dickey-Fuller ](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/dickeyfuller.png?raw=true)

La hipótesis nula (H0) en la prueba de Dickey-Fuller establece que la serie no es estacionaria (tiene tendencia o variaciones dependientes del tiempo).
Como el estadístico ADF (-3.59) es menor que el valor crítico al 5% (-2.86) y el p-value (0.006) es menor que 0.05, rechazamos la hipótesis nula.
Esto indica que la serie es estacionaria a un nivel de significancia del 5%.

## 📊 Visualización de la media y varianza en submuestras
Esto con el fin de evaluar si cambian con el tiempo.

![Media y varianza submuestras](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/media%20varianza%20submuestras.png?raw=true "Media y varianza submuestras")

Se observa un aumento progresivo en la media de la temperatura a medida que avanzamos en el tiempo. El primer segmento tiene una media más baja, mientras que los segmentos posteriores muestran valores crecientes. Esto indica que la serie presenta tendencia ascendente, lo que sugiere que no es estacionaria en media.

La varianza disminuye en los segmentos posteriores en comparación con el primer segmento. La varianza cambia con el tiempo, esto indica que la serie también podría no ser estacionaria en varianza.

Según el análisis previo, la serie no es estacionaria porque tiene una tendencia creciente en la media y una varianza cambiante

La tendencia en la serie sugiere que debemos aplicar diferenciación, que consiste en restar el valor actual con el valor anterior.

## Serie diferenciada

![Serie Diferenciada](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/serie%20de%20tiempo%20diferenciada.png)


Si la varianza cambia con el tiempo, podemos aplicar una transformación 

Hay picos extremos negativos que podrían indicar outliers, pero no parecen formar patrones persistentes de cambio de varianza. Visualmente, no se detecta heterocedasticidad clara (es decir, no hay un ensanchamiento o estrechamiento progresivo de la amplitud de la serie).

## 🔧 Transformación de los datos

![Transformación logarítmica](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/Transformaci%C3%B3n.png)


Después de estos ajustes, debemos volver a aplicar la prueba Dickey-Fuller (ADF) para verificar si la serie ya es estacionaria.

## 🧪 Prueba Dickey - Fuller 
![Dickeynueva](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/nueva%20prueba%20dickey.png?raw=true)

El estadístico ADF (-6.7356) es más bajo que todos los valores críticos (1%, 5%, 10%). La serie es estacionaria. 
Dado esto podemos: 
Modelar con SARIMA o ARIMA sin diferenciación adicional. Utilizar la serie directamente en LSTM, como ya hicimos.
Incluir otras variables climáticas (humedad, presión, viento) si quieres extender el modelo a una versión multivariada.

## 📅 Matriz de Correlación entre Series Temporales
![Matriz de correlación](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/Matriz%20de%20correlaci%C3%B3n.png)

En la matriz de correlación presentada en la imagen, se puede observar que la variable Tdew (degC) tiene una alta correlación positiva con T (degC), con un valor de 0.90. Esto indica que tanto la temperatura del aire (T) como la temperatura de rocío (Tdew) están fuertemente relacionadas.

También se destaca que T (degC) tiene una correlación positiva significativa con las variables Tpot (K) (0.89), VPmax (mbar) (0.95), y VPact (mbar) (0.87). Estas relaciones sugieren que las variables relacionadas con la humedad y la presión del aire (como VPmax y VPact) están fuertemente influenciadas por la temperatura.

Por último, es relevante mencionar que T (degC) está correlacionada negativamente con la variable rho (g/m³), con un valor de -0.96, lo que indica que, a medida que la temperatura aumenta, la densidad del aire disminuye.

## 📊 Gráficos de Dispersión en el Tiempo
Estos gráficos los realizaremos con el objetivo de detectar relaciones no lineales entre la temperatura y otras variables.

![Dispersión](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/Gr%C3%A1ficos%20de%20dispersi%C3%B3n.png)

En los gráficos se puede observar que la variable T (degC) (Temperatura) tiene varias relaciones no lineales con otras variables meteorológicas. A continuación, se analizan algunas de estas relaciones:

Temperatura vs Humedad (%): La relación entre la temperatura y la humedad relativa parece ser no lineal. A medida que la temperatura aumenta, la humedad tiende a disminuir, pero de manera más pronunciada en los valores más bajos de temperatura. Esta curva sugiere que la humedad relativa disminuye rápidamente con el aumento de la temperatura, lo que es común en las condiciones meteorológicas donde el aire más caliente puede contener más vapor de agua.

Temperatura vs Presión (mbar): La relación entre temperatura y presión también parece ser no lineal, con una ligera disminución de la presión conforme la temperatura aumenta, pero la pendiente de la curva varía a lo largo de los valores de temperatura. Esta relación refleja cómo la temperatura afecta a la densidad del aire y, por ende, a la presión atmosférica, aunque la tendencia no es tan estrictamente lineal.

Temperatura vs Velocidad del Viento (wv): En este gráfico, se observa una fuerte dispersión de los valores de velocidad del viento en temperaturas tanto frías como cálidas. La mayoría de los valores se concentran cerca de cero, con algunos valores extremos negativos y positivos en temperaturas bajas. Esto indica que no hay una relación clara y lineal entre la temperatura y la velocidad del viento en los datos observados, posiblemente por la presencia de datos atípicos.

Temperatura vs Presión de Vapor (Vpact, mbar): La relación entre la temperatura y la presión de vapor parece seguir una curva exponencial. A medida que la temperatura aumenta, la presión de vapor aumenta de manera más pronunciada, lo que refleja la mayor capacidad del aire para retener vapor de agua a temperaturas más altas. Esta relación es un ejemplo típico de un comportamiento no lineal, ya que la presión de vapor aumenta de forma acelerada con la temperatura.

En resumen, varias de las relaciones entre la temperatura y otras variables son no lineales, lo que sugiere que los cambios en las variables meteorológicas no siguen una simple regla lineal, sino que presentan un comportamiento más complejo y en algunos casos exponencial, especialmente en lo que respecta a la presión de vapor.

## 📊 Análisis de la Transformada de Fourier

Procedemos a realizar este análisis con el fin de detectar frecuencias dominantes en la serie.

![Transformada](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/Transformada%20fourier.png)

Gráfica Transformada de Fourier Después del pico inicial, la magnitud disminuye rápidamente a medida que aumenta la frecuencia. Esto indica que las componentes de alta frecuencia de la señal tienen una magnitud mucho menor, lo que puede sugerir que la señal es relativamente suave y no presenta fluctuaciones rápidas.

Este espectro de frecuencia muestra que la señal tiene una fuerte componente de baja frecuencia, lo que implica que las fluctuaciones importantes de la señal ocurren a una escala temporal más larga (posiblemente ciclos estacionales o de largo plazo). Las componentes de alta frecuencia tienen poca magnitud, lo que indica que las fluctuaciones rápidas son menos significativas.

Espectro de potencia de la temperatura

El gráfico muestra un pico muy pronunciado en una frecuencia específica (cerca de 0.005 ciclos por día), lo que podría indicar que existe una componente de baja frecuencia dominando la variabilidad de la temperatura. Este pico podría estar asociado con patrones cíclicos, como las fluctuaciones diarias o estacionales de la temperatura.

El espectro de potencia de la temperatura muestra que las variaciones de temperatura se concentran principalmente en frecuencias bajas, lo que indica que los cambios más significativos de temperatura ocurren a una escala temporal más larga, como los ciclos diarios o estacionales.

## 🔎Detección de picos inusuales en la temperatura

![Picos inusuales](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/Picos%20inusuales.png)

Los picos rojos en la gráfica indican eventos donde la temperatura alcanza valores extremos, como picos inusuales de calor o frío que no coinciden con el patrón cíclico habitual. Es posible que estos picos rojos representen eventos como olas de calor o heladas atípicas. La serie temporal muestra una periodicidad evidente, probablemente debida a variaciones estacionales (calor en verano y frío en invierno), mientras que las anomalías corresponden a eventos que se desvían de esta periodicidad. En resumen, la gráfica muestra cómo la temperatura varía a lo largo del tiempo, con algunos valores atípicos o extremos marcados como anomalías, lo que permite identificar eventos climáticos excepcionales.

## 🧪 Prueba para detectar outliers

![Outliers](https://raw.githubusercontent.com/ivanna0994/seriesdetiempo/main/figuras/Outliers.png)

Rango central (IQR): La mayoría de los datos de temperatura se encuentran dentro del rango entre -10°C y 30°C, con la mediana cerca de los 10°C. Outliers: Los puntos fuera de los bigotes, ubicados por encima de 30°C y por debajo de -20°C, son considerados outliers. Estos valores son inusuales y podrían indicar fenómenos extremos o errores en los datos. Distribución de los datos: La temperatura tiene una distribución que se centra principalmente alrededor de la mediana (aproximadamente 10°C), con algunas fluctuaciones hacia valores más bajos y más alto.

## 🔎 Resumen de Hallazgos Claves

Patrones Generales

Tendencia Ascendente: La temperatura media diaria muestra un aumento a lo largo de los años, lo que sugiere un posible cambio climático o variabilidad climática de largo plazo. 
Estacionalidad Anual Fuerte: Se observan ciclos recurrentes con picos en verano y bajas temperaturas en invierno, confirmados por la descomposición de la serie y el análisis de autocorrelación (ACF y PACF). 
Dependencia Temporal: La temperatura de un día está altamente correlacionada con los días anteriores, lo que justifica el uso de modelos como SARIMA o modelos autoregresivos.

Anomalías y Eventos Extremos

Picos inusuales identificados mediante análisis visual y pruebas estadísticas (IQR y Z-score).  Algunos outliers coinciden con eventos climáticos extremos como olas de calor o fríos intensos. Otros valores atípicos pueden deberse a errores de medición, que podrían afectar la precisión de los modelos predictivos.

Análisis de Estacionariedad

La serie original NO era estacionaria, debido a la tendencia ascendente. La diferenciación (d=1) permitió hacerla estacionaria, confirmada por la prueba Dickey-Fuller (ADF). Los ciclos estacionales sugieren que un modelo SARIMA es más adecuado que un ARIMA simple.

## 🧾 Pasos a seguir:

Ajustes para la Modelización

**SARIMA (p,d,q)(P,D,Q,s)** 

- Captura tanto la tendencia como la estacionalidad anual. 
- Se recomienda optimizar los parámetros con técnicas como Grid Search o Auto-SARIMA.

**Modelos más avanzados**

-Transformers Temporales (TFT) o DeepAR podrían mejorar la predicción capturando mejor patrones no lineales. 
-Procesos Gaussianos pueden ser útiles para modelar incertidumbre en predicciones.

**Manejo de Problemas Detectados**

- No Estacionariedad 
- Aplicar diferenciación (ya realizada). 
- Usar modelos con términos estacionales (SARIMA, Prophet, LSTM con ventanas de tiempo).

** Valores Atípicos**

-Eliminar outliers si son errores de medición. 
-Mantener outliers si representan eventos climáticos reales y usarlos para entrenar modelos de predicción de eventos extremos.

**Mejorar la Calidad de Datos**

- Normalizar o estandarizar la temperatura para mejorar la estabilidad del modelo.
- Evaluar otras variables climáticas (humedad, presión, viento) para mejorar la predicción con un enfoque multivariado.

## 🔮 Conclusión Final

 La serie de temperatura es predecible con modelos estacionales debido a su fuerte periodicidad.  La eliminación de outliers y la diferenciación mejoran la precisión de los modelos. Probar con modelos SARIMA, LSTM, o Modelos Transformers para optimizar las predicciones.

## 🧠  Modelo SARIMA
Ejecutaremos el modelo SARIMA(1,1,1)(1,1,0,52)

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 📅 Resampleo semanal de la temperatura promedio (hasta 2013)
df_weekly = df["T (degC)"].resample("W").mean()
df_weekly = df_weekly[:'2013']  # Limitar a datos hasta 2013

# 🧠 Ajuste del modelo SARIMA refinado
modelo_refinado = SARIMAX(
    df_weekly,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 52),  # Estacionalidad anual semanal (sin componente MA estacional)
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# 📋 Mostrar resumen del modelo
print(modelo_refinado.summary())

# 📈 Visualización del ajuste del modelo
df_weekly.plot(label="Observado", figsize=(14, 5))
modelo_refinado.fittedvalues.plot(label="Ajustado")
plt.title("SARIMA Refinado (1,1,1)(1,1,0,52)")
plt.legend()
plt.grid(True)
plt.show()

# 🔮 Pronóstico para las siguientes 52 semanas
forecast = modelo_refinado.forecast(steps=52)
forecast.plot(title="Pronóstico 1 año - Modelo Refinado", figsize=(14, 5))
plt.grid(True)
plt.show()


```
Hemos ejecutado el modelo SARIMA(1,1,1)(1,1,0,52) y ahora sí los resultados son estables y bien condicionados. Vamos a analizarlos:

[![Sarima Refinado](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Sarima%20refinado.png?raw=true "Sarima Refinado")](http://https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Sarima%20refinado.png?raw=true "Sarima Refinado")

Esta gráfica muestra la comparación entre la serie de temperatura observada (línea azul) y la serie ajustada por el modelo SARIMA(1,1,1)(1,1,0,52) (línea naranja), utilizando datos semanales de temperatura promedio en Jena (Alemania) entre 2009 y 2013.

El modelo logra capturar correctamente la estacionalidad anual, con picos en verano y descensos en invierno, como se aprecia en las repeticiones cíclicas. También se observa que el modelo sigue bien la tendencia general de la serie, adaptándose a los cambios interanuales.

[![Pronostico a un año](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Figure_1.png?raw=true "Pronostico a un año")](http://https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Figure_1.png?raw=true "Pronostico a un año")

El modelo logra proyectar un patrón estacional coherente y suavizado, lo que indica un buen ajuste y capacidad predictiva. La curva muestra una transición gradual entre estaciones, sin picos atípicos ni irregularidades, lo que es una señal de estabilidad del modelo.

[![Resultados](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Sarima%20Optimizado.png?raw=true "Resultados")](http://https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Sarima%20Optimizado.png?raw=true "Resultados")

## 🔧 **Estructura:**

```python
SARIMAX(1, 1, 1)x(1, 1, [], 52)

```

Parte no estacional: ARIMA(1,1,1)

Parte estacional: SAR(1), D=1, s=52 (sin MA estacional)

## **Indicadores de Ajuste**

| Métrica | Valor   | Comparación con modelo anterior                                              |
|---------|---------|------------------------------------------------------------------------------|
| AIC     | 847.49  | 🔻 ¡Mejoró! (antes 5335.3, pero en log-likelihood diferente)                 |
| BIC     | 859.67  | Consistente con menor complejidad                                            |
| HQIC    | 852.44  | También bajó                                                                 |

## **Coeficientes del Modelo**

| Parámetro   | Coef  | p-valor | ¿Significativo? | Observación                                 |
|-------------|-------|---------|------------------|----------------------------------------------|
| ar.L1       | 0.433 | 0.000   | ✅ Sí            | Influencia directa del rezago                |
| ma.L1       | -1.000| 0.988   | ❌ No            | No significativo, posible sobreajuste        |
| ar.S.L52    | -0.653| 0.000   | ✅ Sí            | Fuerte estacionalidad anual                  |
| sigma2      | 12.85 | 0.988   | ❌ No            | Alta incertidumbre en la varianza residual   |

## **Diagnosticos de resultados**

| Prueba                 | Resultado | Interpretación                                 |
|------------------------|-----------|------------------------------------------------|
| Ljung-Box (Q)          | 0.13      | ✅ No hay autocorrelación                      |
| Jarque-Bera (JB)       | 0.01      | ❌ Residuos no son normales                    |
| Heterocedasticidad H   | 0.68      | ✅ Varianza residual aceptable                 |
| Kurtosis               | 4.07      | Leve colas pesadas, normal en clima           |

Comentarios: 
Aunque ma.L1 no es significativo, mantenerlo no genera inestabilidad 
El modelo es más estable y estadísticamente más confiable.

## 📊 **Métricas de Predicción**

[![Metricas](https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Metricas%20de%20predicci%C3%B3n.png?raw=true "Metricas")](http://https://github.com/ivanna0994/seriesdetiempo/blob/main/figuras/Metricas%20de%20predicci%C3%B3n.png?raw=true "Metricas")

El RMSE indica un error promedio de 3.44°C en las predicciones semanales. El MAPE de 94% es alto indica que el modelo falla en capturar algunos patrones o los valores son muy cercanos a cero (lo que distorsiona el MAPE)

Posiblemente hay semanas con valores muy pequeños (cerca de 0 °C) que están inflando el MAPE. Aun así, el RMSE es aceptable.

Aunque los residuos no son normales (común en datos climáticos), no hay autocorrelación ni heterocedasticidad significativa.

El modelo captura bien la estructura temporal, especialmente la estacionalidad. El MAPE alto sugiere explorar ajustes, por ejemplo:

1. Normalizar/estandarizar la temperatura
2. Eliminar ma.L1
3. Probar modelos no lineales como Prophet o LSTM



