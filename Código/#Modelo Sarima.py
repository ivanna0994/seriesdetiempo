import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Cargar datos
df = pd.read_csv("C:/Users/10Pearls/Documents/Proyecto final/jena_climate_2009_2016.csv")  # ← Ajusta la ruta aquí
df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
df.set_index('Date Time', inplace=True)

# Resampleo semanal y recorte
df_weekly = df['T (degC)'].resample('W').mean()
df_weekly = df_weekly[:'2013']

# Modelo refinado sin ma.S.L52
modelo_refinado = SARIMAX(
    df_weekly,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 52),  # ← sin ma estacional
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

# Mostrar resumen
print(modelo_refinado.summary())

# Graficar ajuste
df_weekly.plot(label='Observado', figsize=(14, 5))
modelo_refinado.fittedvalues.plot(label='Ajustado')
plt.title("SARIMA Refinado (1,1,1)(1,1,0,52)")
plt.legend()
plt.grid(True)
plt.show()

# Pronóstico 52 semanas
forecast = modelo_refinado.forecast(steps=52)
forecast.plot(title="Pronóstico 1 año - Modelo Refinado", figsize=(14, 5))
plt.grid(True)
plt.show()

