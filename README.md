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
!([Null datos.png](https://github.com/ivanna0994/seriesdetiempo/blob/main/Null%20datos.png))
