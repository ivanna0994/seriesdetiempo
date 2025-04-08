import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import shapiro
from IPython import get_ipython
from IPython.display import display
import pyperclip
from io import StringIO

# === Cargamos datos ===
file_path = "C:/Users/10Pearls/Documents/Proyecto final/jena_climate_2009_2016.csv"
df = pd.read_csv(file_path)

# === Mostrar primeras filas en consola ===
import pandas as pd
import pyperclip
from io import StringIO

# Cargar datos
file_path = "C:/Users/10Pearls/Documents/Proyecto final/jena_climate_2009_2016.csv"
df = pd.read_csv(file_path)

# Calcular valores nulos
nulos = df.isnull().sum()
print("🔍 Valores nulos por columna:")
print(nulos)

# Convertir la columna "Date Time" a formato datetime
df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")

# Verificar el rango de fechas en el dataset
date_range = df["Date Time"].min(), df["Date Time"].max()

# === Histograma de la temperatura ===
plt.figure(figsize=(8, 5))
sns.histplot(df["T (degC)"], bins=50, kde=True, color="blue")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Frecuencia")
plt.title("Distribución de la Temperatura")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Estadísticas descriptivas ===
estadisticas = df["T (degC)"].describe()
print("📊 Estadísticas de la Temperatura:")
print(estadisticas)


# Prueba de Shapiro-Wilk
stat, p = shapiro(df['T (degC)'])

# Formato de salida
resultado = StringIO()
resultado.write("🔬 Prueba de Normalidad: Shapiro-Wilk\n")
resultado.write(f"Statistics = {stat:.6f}, p-value = {p:.6e}\n")

alpha = 0.05
if p > alpha:
    conclusion = "✅ La muestra parece normal (no se rechaza H0)"
else:
    conclusion = "❌ La muestra NO parece normal (se rechaza H0)"

resultado.write(conclusion)

# Imprimir en consola
print(resultado.getvalue())

# Seleccionar columnas
data = df[['Date Time', 'T (degC)']]

# Mostrar primeras filas
print("🔍 Primeras filas de data:")
print(data.head())

# Extract Year and Month from 'Date Time'
data['Year'] = data['Date Time'].dt.year
data['Month'] = data['Date Time'].dt.month

# Now you can create the index
data.index = data['Year'].astype(str) + '-' + data['Month'].astype(str).map(lambda x: '0'+x if len(x) == 1 else x)
data.head()

data = data.loc[(~pd.isnull(data['T (degC)'])) & \
                (~pd.isnull(data['Year']))& \
                (~pd.isnull(data['Month']))]

data.sort_values(['Year', 'Month'], inplace=True)

plt.figure(figsize=(5.5, 5.5))
data['T (degC)'].plot(color='b')
plt.title('Temperatura Mensual')
plt.xlabel('Time')
plt.ylabel('C° Temperatura')
plt.xticks(rotation=30);

data.index = pd.to_datetime(data.index, format='%Y-%m')
frecuencia = pd.infer_freq(data.index)
print("Frecuencia: ", frecuencia)


from sklearn.linear_model import LinearRegression
import numpy as np
trend_model = LinearRegression(fit_intercept=True)
trend_model.fit(np.arange(data.shape[0]).reshape((-1,1)), data['T (degC)'])
trend_model = LinearRegression(fit_intercept=True)
trend_model.fit(np.arange(data.shape[0]).reshape((-1,1)), data['T (degC)'])

print('Trend model coefficient={} and intercept={}'.format(trend_model.coef_[0], trend_model.intercept_))

residuals = np.array(data['T (degC)']) - trend_model.predict(np.arange(data.shape[0]).reshape((-1,1)))
plt.figure(figsize=(5.5, 5.5))
pd.Series(data=residuals, index=data.index).plot(color='b')
plt.title('Residuals of trend model for C° temperature')
plt.xlabel('Time')
plt.ylabel('C° Temperature')
plt.xticks(rotation=30);

data['Residuals'] = residuals
month_quarter_map = {1: 'Q1', 2: 'Q1', 3: 'Q1',
                     4: 'Q2', 5: 'Q2', 6: 'Q2',
                     7: 'Q3', 8: 'Q3', 9: 'Q3',
                     10: 'Q4', 11: 'Q4', 12: 'Q4'}
data['Quarter'] = data['Month'].map(lambda m: month_quarter_map.get(m))

seasonal_sub_series_data = data.groupby(by=['Year', 'Quarter'])['Residuals'].aggregate([np.mean, np.std])
seasonal_sub_series_data.columns = ['Quarterly Mean', 'Quarterly Standard Deviation']
seasonal_sub_series_data.reset_index(inplace=True)
seasonal_sub_series_data.index = seasonal_sub_series_data['Year'].astype(str) + '-' + seasonal_sub_series_data['Quarter']
seasonal_sub_series_data.head()

plt.figure(figsize=(5.5, 5.5))
seasonal_sub_series_data['Quarterly Mean'].plot(color='b')
plt.title('Quarterly Mean of Residuals')
plt.xlabel('Time')
plt.ylabel('Temperature C°')
plt.xticks(rotation=30);

plt.figure(figsize=(5.5, 5.5))
seasonal_sub_series_data['Quarterly Standard Deviation'].plot(color='b')
plt.title('Quarterly Quarterly Standard Deviation of Residuals')
plt.xlabel('Time')
plt.ylabel('Temperature C°')
plt.xticks(rotation=30);

plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(data=data[data['Year'] > 2009], y='Residuals', x='Quarter')
g.set_title('Quarterly Boxplots of Residuals')
g.set_xlabel('Time')
g.set_ylabel('Temperature C°');

# Se generan las medias diarias
daily = data['T (degC)'].resample('D')
daily_mean = daily.mean()

monthly = data['T (degC)'].resample('ME')
monthly_mean = monthly.mean()

from statsmodels.tsa.stattools import adfuller
import pyperclip
from io import StringIO

# Aplicar la prueba de Dickey-Fuller Aumentada (ADF)
adf_test = adfuller(daily_mean.dropna())

# Extraer los resultados
adf_statistic = adf_test[0]
p_value = adf_test[1]
critical_values = adf_test[4]

# Formatear resultados
output = StringIO()
output.write("🔍 Prueba de Estacionariedad: Dickey-Fuller Aumentada (ADF)\n")
output.write(f"Estadístico ADF: {adf_statistic:.4f}\n")
output.write(f"Valor p: {p_value:.6f}\n")
output.write("Valores Críticos:\n")
for clave, valor in critical_values.items():
    output.write(f"   {clave}: {valor:.4f}\n")

# Imprimir en consola
print(output.getvalue())

import numpy as np

# Resampleo para ver tendencias diarias y mensuales
# Set 'Date Time' column as index before resampling
df_daily = df.set_index("Date Time")["T (degC)"].resample("D").mean()
df_monthly = df.set_index("Date Time")["T (degC)"].resample("M").mean()

# Gráfico de tendencias diarias
plt.figure(figsize=(12, 5))
plt.plot(df_daily, label="Temperatura diaria (°C)", alpha=0.7)
plt.xlabel("Fecha")
plt.ylabel("Temperatura (°C)")
plt.title("Tendencia diaria de la temperatura")
plt.legend()
plt.grid()

# Mostrar el gráfico
plt.show()

# Gráfico de tendencias mensuales
plt.figure(figsize=(12, 5))
plt.plot(df_monthly, label="Temperatura mensual (°C)", color="red", alpha=0.8)
plt.xlabel("Fecha")
plt.ylabel("Temperatura (°C)")
plt.title("Tendencia mensual de la temperatura")
plt.legend()
plt.grid()

# Mostrar el gráfico
plt.show()

# Interpolación de valores faltantes en la serie diaria
df_daily_interpolated = df_daily.interpolate()

# Aplicar descomposición estacional
decomposition = seasonal_decompose(df_daily_interpolated, model="additive", period=365)

# Graficar la descomposición
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(df_daily_interpolated, label="Serie Original", alpha=0.7)
plt.legend()

plt.subplot(412)
plt.plot(decomposition.trend, label="Tendencia", alpha=0.7)
plt.legend()

plt.subplot(413)
plt.plot(decomposition.seasonal, label="Estacionalidad", alpha=0.7)
plt.legend()

plt.subplot(414)
plt.plot(decomposition.resid, label="Residuo", alpha=0.7)
plt.legend()

plt.tight_layout()
plt.show()

# Autocorrelación y autocorrelación parcial
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Gráfico de la función de autocorrelación (ACF)
sm.graphics.tsa.plot_acf(df_daily_interpolated, lags=50, ax=axes[0])
axes[0].set_title("Autocorrelación (ACF)")

# Gráfico de la función de autocorrelación parcial (PACF)
sm.graphics.tsa.plot_pacf(df_daily_interpolated, lags=50, ax=axes[1])
axes[1].set_title("Autocorrelación Parcial (PACF)")

plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "C:/Users/10Pearls/Documents/Proyecto final/jena_climate_2009_2016.csv"
df = pd.read_csv(file_path)
df = pd.read_csv(file_path, parse_dates=['Date Time'], index_col='Date Time')

# Seleccionar la variable de temperatura
serie = df['T (degC)']

# Definir número de segmentos (submuestras)
num_segm = 4
longitud = len(serie) // num_segm

# Calcular medias y varianzas en cada submuestra
medias = []
varianzas = []
segmentos = []

for i in range(num_segm):
    inicio = i * longitud
    fin = (i + 1) * longitud if i < num_segm - 1 else len(serie)

    submuestra = serie.iloc[inicio:fin]
    segmentos.append(f"Segmento {i+1}")

    medias.append(np.mean(submuestra))
    varianzas.append(np.var(submuestra))

# Graficar la media y varianza en submuestras
fig, ax = plt.subplots(2, 1, figsize=(12, 8))

ax[0].bar(segmentos, medias, color='royalblue')
ax[0].set_title("Media de la Temperatura en Submuestras")
ax[0].set_ylabel("Media (°C)")

ax[1].bar(segmentos, varianzas, color='darkorange')
ax[1].set_title("Varianza de la Temperatura en Submuestras")
ax[1].set_ylabel("Varianza")

plt.tight_layout()
plt.show()

df['Temp_diff'] = df['T (degC)'].diff()  # Primera diferencia
df['Temp_diff'].dropna().plot(title="Serie Diferenciada")
plt.show()

df['Temp_log'] = np.log(df['T (degC)'])
df['Temp_log'].plot(title="Transformación Logarítmica")
plt.show()


# Reducir muestra: cada 10 registros
df_sample = df['T (degC)'][::10].dropna()

# Prueba ADF
adf_result_sample = adfuller(df_sample)

# Formatear resultados
output = StringIO()
output.write("🔍 Prueba ADF sobre muestra reducida (cada 10 registros)\n")
output.write(f"Estadístico ADF: {adf_result_sample[0]:.4f}\n")
output.write(f"p-valor: {adf_result_sample[1]:.6f}\n")
output.write("Valores críticos:\n")
for nivel, valor in adf_result_sample[4].items():
    output.write(f"   {nivel}: {valor:.4f}\n")

# Imprimir en consola
print(output.getvalue())

# Copiar al portapapeles
pyperclip.copy(output.getvalue())
print("📋 Resultados copiados al portapapeles.")