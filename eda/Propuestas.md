# Variables ERA5-Land para enriquecimiento del dataset CONAF

El dataset de CONAF (2002–2020) provee información operacional de cada incendio registrado, cuyo valor analítico directo es limitado. La excepción son las coordenadas de ignición (latitud y longitud), que permiten enriquecer el dataset con información climatológica de alta resolución.

Para ello se utiliza **ERA5-Land**, el producto de reanálisis climático del ECMWF con resolución espacial de ~9 km y cobertura horaria. Dado que ERA5-Land expone aproximadamente 7.700 variables, la selección se fundamenta en lo reportado por dos estudios recientes de predicción de incendios forestales con XAI:

1. Zakari, R. Y., Malik, O. A., & Ong, W.-H. (2025). Spatio-temporal wildfire forecasting in Australia using deep learning and explainable AI. *Modeling Earth Systems and Environment*, 11, 425. https://doi.org/10.1007/s40808-025-02621-7

2. Liao, B., Zhou, T., Liu, Y., Li, M., & Zhang, T. (2025). Tackling the wildfire prediction challenge: An explainable artificial intelligence (XAI) model combining extreme gradient boosting (XGBoost) with SHapley additive exPlanations (SHAP) for enhanced interpretability and accuracy. *Forests*, 16(4), 689. https://doi.org/10.3390/f16040689

---

## Variables seleccionadas

### Temperatura

| Variable | Código ERA5-Land |
|---|---|
| Temperatura del aire a 2 m (máxima diaria) | `2m_temperature` |
| Temperatura del punto de rocío a 2 m | `2m_dewpoint_temperature` |
| Temperatura de la superficie terrestre | `skin_temperature` |
| Temperatura del suelo capa 1 (0–7 cm) | `soil_temperature_level_1` |

### Humedad y precipitación

| Variable | Código ERA5-Land |
|---|---|
| Humedad relativa (derivada, ver abajo) | — |
| Déficit de presión de vapor — VPD (derivada, ver abajo) | — |
| Precipitación total | `total_precipitation` |
| Evapotranspiración potencial | `potential_evaporation` |
| Contenido de agua en suelo capa 1 (0–7 cm) | `volumetric_soil_water_layer_1` |
| Contenido de agua en suelo capa 2 (7–28 cm) | `volumetric_soil_water_layer_2` |

### Viento

| Variable | Código ERA5-Land |
|---|---|
| Componente U del viento a 10 m | `10m_u_component_of_wind` |
| Componente V del viento a 10 m | `10m_v_component_of_wind` |
| Velocidad escalar del viento (derivada, ver abajo) | — |

### Radiación

| Variable | Código ERA5-Land |
|---|---|
| Radiación solar en superficie (SSRD) | `surface_solar_radiation_downwards` |
| Radiación solar neta | `surface_net_solar_radiation` |
| Radiación de onda larga hacia abajo | `surface_thermal_radiation_downwards` |

### Vegetación

| Variable | Código ERA5-Land |
|---|---|
| Índice de área foliar — vegetación alta | `leaf_area_index_high_vegetation` |
| Índice de área foliar — vegetación baja | `leaf_area_index_low_vegetation` |
| Fracción de cobertura vegetal | `fraction_of_vegetation_cover` |

---

## Variables derivadas

Las siguientes variables no se extraen directamente de ERA5-Land sino que se calculan a partir de las anteriores.

### Humedad relativa (HR)

Se deriva de `2m_temperature` (T) y `2m_dewpoint_temperature` (Td), ambas en °C:

```python
import numpy as np

def humedad_relativa(t, td):
    es = 0.6108 * np.exp(17.27 * t  / (t  + 237.3))
    ea = 0.6108 * np.exp(17.27 * td / (td + 237.3))
    return 100 * (ea / es)  # porcentaje
```

### Déficit de presión de vapor (VPD)

El VPD es el predictor individual más potente para discriminar tamaño final de incendio en la literatura (Coffield et al., 2019). Integra temperatura y humedad en un único índice de estrés hídrico atmosférico. Se obtiene como la diferencia entre la presión de vapor de saturación y la presión de vapor real:

```python
def vpd(t, td):
    es = 0.6108 * np.exp(17.27 * t  / (t  + 237.3))
    ea = 0.6108 * np.exp(17.27 * td / (td + 237.3))
    return es - ea  # kPa
```

Se recomienda calcular el **VPD máximo diario** (peak de la tarde), ya que es el momento de mayor peligro de propagación. Para ello, aplicar la fórmula hora a hora sobre datos horarios de ERA5-Land y tomar el máximo diario.

### Velocidad escalar del viento

Se combina a partir de las componentes vectoriales U y V:

```python
def velocidad_viento(u, v):
    return np.sqrt(u**2 + v**2)  # m/s
```

---

## Nota sobre el contexto del análisis

El dataset de CONAF contiene exclusivamente registros de incendios confirmados, por lo que el problema no es de detección de ocurrencia (fire vs. no-fire) sino de **clasificación de severidad dentro del subconjunto de incendios** (incendio ordinario vs. mega-incendio). Este setting desplaza la relevancia de las variables: los predictores de propagación —VPD, viento, precipitación acumulada previa— adquieren mayor peso que los de ignición, que dominan la literatura de detección.
