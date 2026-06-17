# Datos y etiquetado de megaincendios

> **Documento consolidado.** Fusiona dos borradores previos (antes `docs/megafire_labeling_pipeline.html` y `docs/data_section_draft.html`):
> la **Parte I** es la metodología de construcción de las etiquetas (L1/L2, FRP→FLI, pipeline) y la
> **Parte II** es la sección «Data» del paper en prosa IEEE. Hay solape temático (CONAF, ERA5-Land, MODIS/FIRMS, FLI);
> cada parte lo aborda desde su ángulo (referencia interna vs. texto publicable).

---

## Parte I — Pipeline de etiquetado de megaincendios

Construcción del label binario para clasificación XGBoost desde datos
CONAF + ERA5-Land

Proyecto XAI — Mayo 2026 · Documento técnico v1.0

### 1. Contexto y objetivo

#### Pregunta del proyecto

Se requiere construir un modelo de clasificación binaria que prediga, en
el momento de ignición de un incendio forestal en Chile, si éste
alcanzará comportamiento de **Extreme Wildfire Event (EWE)** según el
estándar internacional de Tedim et al. 2018 — categoría 5+ con FLI ≥
10.000 kW/m (label **L2**) — usando exclusivamente las condiciones
meteorológicas, geográficas y vegetativas presentes al inicio del
evento.

**Target objetivo vs proxy secundario.**
El **target objetivo** del proyecto es L2 (FLI ≥ 10.000 kW/m vía
MODIS-FRP). El pipeline L2 ya está implementado con NASA FIRMS Area API;
L1b queda como *proxy secundario*: threshold P99 regional sobre
`superficie_quemada_total_ha`. Este documento describe ambos labels,
pero la jerarquía es: L2 = objetivo científico, L1b = sensitivity
analysis.

El modelo objetivo es **XGBoost**, con dos casos de uso:

1.  **Operacional**: «¿Este fuego que recién partió alcanzará
    comportamiento EWE (cat 5+ de Tedim)?» — predicción ex-ante para
    asignación de recursos de combate y declaración de alerta
    operacional.
2.  **Interpretativo (XAI / SHAP)**: «¿Qué condiciones climáticas
    predisponen a megaincendio?» — análisis de drivers físicos para
    informar política pública.

#### Fuentes de datos

- **CONAF** (Corporación Nacional Forestal): registro oficial de
  incendios forestales en Chile, originalmente 2002-2020 con ~110.000
  eventos georreferenciados, superficie quemada por tipo de vegetación,
  duración y contexto operacional. Fuente: [Datos para Resiliencia /
  itrend](https://datospararesiliencia.cl), DOI
  [10.71578/UXAUN5](https://doi.org/10.71578/UXAUN5). El dataset
  2002-2020 está archivado en `data/archive/conaf_clean.parquet`.
- **ERA5-Land** (Copernicus / ECMWF): reanálisis climático horario a ~9
  km de resolución espacial, variables meteorológicas, de suelo y
  vegetativas. Documentación:
  [ecmwf.int/era5-land](https://www.ecmwf.int/en/era5-land).

**Alcance del análisis.** Este documento
describe el pipeline sobre la corrida base **2012-2018**
(`data/processed/conaf_enriched_2012_2018.parquet` / latest) y el subset
analítico de tres regiones CONAF históricas: **Maule**, **Biobío** y
**Araucanía**. En este período, **Ñuble** no aparece separado en CONAF y
se trata como parte de Biobío histórico.

#### Por qué clasificación binaria y no regresión

Se descartó predecir directamente `superficie_quemada_total_ha`
(regresión) por dos razones:

- La distribución es *power-law* con mediana 0.45 ha y máximo 159.812 ha
  — un MSE estaría dominado por errores en los pocos eventos extremos.
- El usuario final necesita una decisión operacional binaria, no una
  estimación continua de hectáreas.

La pregunta de fondo es *clasificatoria*: ¿este fuego pertenece a la
categoría «manejable» o «catastrófica»? El umbral que separa estas
categorías es la pieza central del diseño y motiva todo este documento.

**Nota.** En adelante, «label», «etiqueta»
y «target» se usan como sinónimos. Las features («predictoras»,
«variables independientes») son lo que el modelo recibe como input.

### 2. Glosario técnico

Conceptos básicos para entender la literatura internacional de
comportamiento del fuego y el pipeline propuesto. Cada término incluye
su fórmula (si aplica), unidades y rol.

FLI — Fire Line Intensity (Intensidad del Frente de Llamas)  
Cantidad de energía liberada por unidad de tiempo y unidad de longitud
del frente de fuego. Es el parámetro más importante en clasificación de
comportamiento extremo.  
**Fórmula de Byram (1959)**: `FLI = H · w · R`  
donde `H` = calor de combustión efectivo (kJ/kg), `w` = carga de
combustible consumida (kg/m²), `R` = rate of spread (m/s).  
**Unidades**: kW/m (kilovatios por metro lineal de frente).  
**Umbral crítico**: ≥10.000 kW/m → el fuego excede la capacidad de
control con tecnología actual (incluso aviones tanqueros pesados).

ROS — Rate of Spread  
Velocidad a la cual avanza el frente del fuego. Unidades típicas: m/min
o m/s. Es función de viento, pendiente, humedad del combustible, y
propiedades del lecho combustible. Modelo de referencia: Rothermel
(1972).

FL — Flame Length  
Longitud de la llama desde su base hasta la punta, medida a lo largo de
su eje. Unidades: metros. Se relaciona con FLI vía Alexander (1982):
`FLI ≈ 259.833 · FL`^(`2.174`). Permite estimar FLI sin medir
directamente ROS y carga.

FRP — Fire Radiative Power  
Potencia electromagnética emitida por un fuego en la banda
media-infrarroja (3.9 μm), medida desde sensores satelitales. Unidades:
MW (megavatios por pixel). Es proporcional a la tasa de consumo de
biomasa y, indirectamente, a FLI (Wooster et al. 2003). **Ventaja
clave**: medible desde el espacio, cobertura global, sin necesidad de
instrumentación in situ.

MODIS — Moderate Resolution Imaging Spectroradiometer  
Sensor multi-banda a bordo de los satélites NASA **Terra** (lanzado
1999, operativo desde 2000) y **Aqua** (lanzado 2002). Resolución
espacial nominal 1 km para detección de fuegos. Pasa sobre Chile ~4
veces al día (2 Terra + 2 Aqua, mañana/tarde y noche). **Productos
relevantes**:

- `MOD14` (Terra) y `MYD14` (Aqua): swath fire pixels — detección
  instantánea de hotspots con FRP.
- `MOD14A1`/`MYD14A1`: composite diario — más fácil de consultar vía
  Google Earth Engine.

**Limitación**: fuegos \<~50 ha no son detectados consistentemente
(sub-pixel mixing), y la cobertura se interrumpe con nubosidad. **Para
estimar FLI se usa el swath L2** (`MOD14`/`MYD14`), no el composite
diario: la conversión de Wooster 2003 necesita el FRP instantáneo de
cada detección, que los composites A1/A2 promedian (ver Sección 7).

PyroCb — Pyrocumulonimbus  
Nube cumulonimbus generada por la columna convectiva de un fuego de muy
alta intensidad. Puede alcanzar la tropopausa (15 km), producir rayos,
downdrafts y spotting masivo. Es una **firma inequívoca de Extreme Fire
Behavior (EFB)**.

Spotting  
Ignición de nuevos focos secundarios cuando el fuego primario lanza
pavesas o brasas transportadas por la columna convectiva o el viento.
Distancia varía desde decenas de metros (cat 2-3 Tedim) hasta \>5 km en
firestorms (cat 7). Para fuegos cat 5+ el spotting domina la
propagación, más que el avance del frente.

EFB — Extreme Fire Behavior  
Conjunto de manifestaciones de comportamiento no-controlable: crown
fire, PyroCb, downbursts, spotting masivo, frente errático. Cualitativo;
categorías 5+ de Tedim lo presentan parcialmente, categoría 7 lo
presenta plenamente.

EWE — Extreme Wildfire Event  
Terminología propuesta por Tedim et al. (2018) para incendios que
superan la **capacidad de control con tecnología actual**. Se diferencia
operacionalmente de *megafire* y *large fire* al no apoyarse en tamaño
sino en comportamiento. Definición cuantitativa: **FLI ≥ 10.000 kW/m +
ROS \> 50 m/min + spotting \> 1 km + comportamiento errático**.

Terminología comparada  
- **Large Fire (LF)**: definido por tamaño, con umbral variable según
  país (100 ha Europa, 1.000 ha Australia, 4.950 ha boreal Ontario).
- **Megafire**: tamaño + impacto. NIFC (US) usa 40.469 ha (100.000
  acres); EU MEGAFIREs y España usan 500 ha. *Sin consenso*.
- **EWE**: comportamiento físico (FLI, ROS, spotting). El paper Tedim
  argumenta que es la terminología más rigurosa al ser independiente de
  la geografía.

### 3. Estándar internacional: Tedim et al. 2018

El paper *«Defining Extreme Wildfire Events: Difficulties, Challenges,
and Impacts»* (Tedim et al. 2018, *Fire* MDPI 1(1):9) es el documento de
referencia internacional más citado para definición operacional de
incendios extremos. Es un Concept Paper que sintetiza literatura
interdisciplinaria y propone una taxonomía de 7 categorías. Disponible
localmente en `references/Tedim2018_DefiningEWE.pdf`.

#### 3.1 Definición operacional de EWE

> «A pyro-convective phenomenon overwhelming capacity of control
> (fireline intensity currently assumed ≥ 10,000 kWm⁻¹; rate of spread
> \>50 m/min; exhibiting spotting distance \>1 km, and erratic and
> unpredictable fire behavior and spread). It represents a heightened
> threat to crews, population, assets, and natural values, and likely
> causes relevant negative socio-economic and environmental impacts.»
> Tedim et al. 2018, p. 10

La clave: **EWE se define por comportamiento del fuego, no por tamaño**.
El umbral fundamental es `FLI ≥ 10.000 kW/m` porque a partir de ese
punto el control con tecnología actual deja de ser viable, incluso con
water bombers pesados.

#### 3.2 Tabla 3: clasificación en 7 categorías

Reproducción de la Tabla 3 del paper. Las categorías 1-4 son *Normal
Fires*; 5-7 son *Extreme Wildfire Events*. La línea punteada marca el
umbral de capacidad operativa de control.

| Cat | FLI (kW/m) | ROS (m/min) | FL (m) | PyroCb | Downdrafts | Spotting Act. | Spotting Dist. (m) | Tipo / Control |
|----|----|----|----|----|----|----|----|----|
| 1 | \<500 | \<5–15 | \<1.5 | Ausente | Ausente | Ausente | 0 | Surface fire — fácil |
| 2 | 500–2.000 | \<15–30 | \<2.5 | Ausente | Ausente | Bajo | \<100 | Surface fire — moderadamente difícil |
| 3 | 2.000–4.000 | \<20–50 | 2.5–3.5 | Ausente | Ausente | Alto | ≥100 | Surface fire, torching posible — muy difícil |
| 4 | 4.000–10.000 | \<50–100 | 3.5–10 | Improbable | Localizado | Prolífico | 500–1.000 | Crown fire posible — extremadamente difícil |
| ▼ Umbral de capacidad de control: a partir de aquí Extreme Wildfire Events (EWE) ▼ |  |  |  |  |  |  |  |  |
| 5 | 10.000–30.000 | \<150–250 | 10–50 | Posible | Presente | Prolífico | \>1.000 | Crown fire — virtualmente imposible |
| 6 | 30.000–100.000 | \<300 | 50–100 | Probable | Presente | Masivo | \>2.000 | Plume-driven — imposible |
| 7 | \>100.000 | \>300 | \>100 | Presente | Presente | Masivo | \>5.000 | Firestorm — imposible |

#### 3.3 Por qué el paper rechaza hectáreas como criterio

En la Sección 3.1 (página 7) el paper es explícito:

> «We do not propose wildfire size as a criterion to define EWE for
> several reasons: (i) it is place-dependent, reflecting landscape
> characteristics, so it is not possible to establish a commonly
> accepted and absolute threshold; (ii) size and severity do not have a
> direct correlation; (iii) size tells us little about losses and
> damages; (iv) size can also be the result of wildland fire use.» Tedim
> et al. 2018, p. 7

El argumento es contundente: un fuego de 10.000 ha en pastizal
patagónico tras varios días de propagación lenta NO es un EWE; un fuego
de 2.000 ha cat 6 con PyroCb sí lo es. El tamaño es un *resultado
correlacionado* con el comportamiento, pero no es el criterio
definitorio.

#### 3.4 Las tres vías para estimar FLI mencionadas en el paper

En la Sección 3.3 (p. 8) el paper enumera tres rutas para estimar FLI:

1.  **Directo (Byram 1959)**: medir o estimar ROS y consumo de
    combustible, aplicar `FLI = H · w · R`. Requiere modelo de
    comportamiento (Rothermel, BehavePlus, FlamMap) más datos in situ.
2.  **Desde Flame Length (Alexander 1982)**:
    `FLI ≈ 259.833 · FL`^(`2.174`). Operacionalmente conveniente
    porque la FL se estima visualmente en terreno.
3.  **Desde FRP (Wooster et al. 2003)**: convertir potencia radiativa
    MODIS a intensidad lineal del frente. Cita textual: *«FLI estimates
    from FRP can be obtained for ongoing monitored wildfires by remote
    sensing imagery»*.

Para datos históricos (post-evento) sin instrumentación in situ, la
**vía FRP es la única operacionalmente viable** a escala de dataset
completo.

### 4. El problema del label en CONAF

#### 4.1 Caracterización del dataset CONAF (2012-2018; subset Maule–Biobío–Araucanía)

La base CONAF 2012-2018 contiene **42.963 eventos**. El subset analítico
usado para modelado concentra **29.510 eventos** en Maule, Biobío y
Araucanía. El parquet enriquecido con L2 debe quedar con **70
columnas**: 27 CONAF, 19 ERA5 crudas, 11 derivadas, 6 invariantes, 3
metadatos ERA5 y 4 columnas MODIS/L2 (`modis_n_matches`,
`modis_frp_max_mw`, `fli_estimado_kw_m`, `label_l2`).

| Estadístico           | superficie_quemada_total_ha |
|-----------------------|-----------------------------|
| N total subset        | 29.510                      |
| N con superficie \> 0 | 29.501                      |
| Mediana               | 0.30                        |
| P75                   | 1.5                         |
| P90                   | 5.5                         |
| P95                   | 15.4                        |
| P98                   | 60.0                        |
| P99                   | 148.3                       |
| Máximo                | 159.813                     |
| Suma total            | 625.098 ha                  |

La distribución es marcadamente **power-law**: la mayoría de eventos son
pequeños (mediana 0.45 ha), pero una cola larga incluye fuegos
catastróficos como **Las Máquinas** (enero 2017, Maule) que aporta el
máximo de ~160.000 ha. El período 2012-2018 incluye años normales y la
temporada catastrófica 2016-2017; por eso permite analizar el régimen
centro-sur sin que el documento dependa exclusivamente del «incendio del
siglo».

#### 4.2 Por qué no se puede aplicar Tedim directo

El paper Tedim define EWE por FLI, ROS, FL y spotting. Para aplicar el
estándar tal cual necesitaríamos:

- Medición instantánea de FLI durante el incendio → **no existe** en
  CONAF.
- Mediciones de FL en terreno → no registradas.
- Logs de comportamiento de la columna convectiva (PyroCb) → no
  registrados.

CONAF solo registra *outcomes* (superficie final, duración, alerta,
escenario) — todos **ex-post**. Para aplicar Tedim necesitamos *inferir*
el comportamiento del fuego desde proxies, y la única vía
operacionalmente viable a escala completa es FRP desde MODIS.

#### 4.3 Análisis estadístico realizado

El script `scripts/megafire_thresholds.py` aplica cuatro métodos para
proponer umbrales de severidad:

1.  **Percentil**: P95, P98, P99 sobre eventos con superficie \> 0.
2.  **Log-normal**: ajuste Normal sobre log(ha), umbral en exp(μ + k·σ).
3.  **Pareto-80%**: superficie mínima donde el 20% superior acumula 80%
    del área.
4.  **Benchmark**: ¿en qué percentil caen 200, 500 y 1.000 ha (umbrales
    literatura internacional)?

El reporte completo está en `data/processed/megafire_thresholds.md`.

**Caveat sobre el reporte de umbrales.**
Los números de esta sección corresponden al recálculo sobre CONAF
2012-2018 filtrado a Maule, Biobío y Araucanía. Cuando el parquet
enriquecido 2012-2018 quede como `conaf_enriched_latest.parquet`, el
script debe re-ejecutarse para regenerar
`data/processed/megafire_thresholds.md` con la misma ventana.

#### 4.4 P99 regional — resultados

De los cuatro métodos, se adopta **P99 regional** como base del label
L1b. Justificación en la [Sección 5](#estrategia). Tabla con los
umbrales por región:

| Región                | N total | P99 (ha) | Nota                        |
|-----------------------|---------|----------|-----------------------------|
| **SUBSET 3 regiones** | 29.510  | 148.3    | Referencia analítica        |
| Araucanía             | 7.090   | 250.3    | —                           |
| Biobío                | 17.717  | 89.2     | Incluye Ñuble histórico     |
| Maule                 | 4.703   | 204.0    | Incluye Las Máquinas (2017) |

**Nota sobre Ñuble:** no se trata como cuarta región porque CONAF
2012-2018 no la expone separada en la columna `region`; sus eventos
quedan dentro de Biobío.

**Observación.** P99 varía entre las tres
regiones: Biobío 89.2 ha, Maule 204.0 ha y Araucanía 250.3 ha. La
heterogeneidad justifica mantener umbrales regionales para L1b, aunque
L2 sea el target científico principal.

### 5. Estrategia de etiquetado adoptada

El proyecto adopta **dos labels con roles diferenciados**: **L2** (FLI ≥
10.000 kW/m vía MODIS-FRP) como *target objetivo* alineado con el
estándar Tedim et al. 2018, y **L1b** (P99 regional de hectáreas) como
*proxy secundario* para análisis de sensibilidad. Se entrenan dos
modelos XGBoost independientes (uno por label) para comparar sus SHAPs e
identificar drivers robustos versus específicos del label.

#### 5.1 Label L2 — FLI desde MODIS-FRP (target objetivo, alineado con Tedim 2018)

**Definición**: `label_l2 = 1` si `FLI_estimado ≥ 10.000 kW/m`, sino
`0`. FLI se estima desde FRP MODIS (ver pipeline detallado en [Sección
6](#pipeline)).

**Por qué L2 es el target objetivo del
proyecto.**

- **Alineamiento con estándar internacional**: Tedim et al. 2018 es la
  referencia metodológica consolidada para definir *Extreme Wildfire
  Events* (cat 5+). El paper final puede comparar drivers de EWE en
  Chile con los reportados en Australia, China o Mediterráneo europeo.
- **Captura comportamiento físico**: FLI integra contenido calórico,
  masa de combustible y velocidad de propagación. Predecir FLI es
  predecir el comportamiento peligroso, no solo el outcome de
  superficie.
- **Inmune al sesgo de 2016-17**: si Las Máquinas tuvo FLI ~60k kW/m,
  será label=1 sin importar que distorsione los P99 regionales del proxy
  L1b.
- **El paper reporta sobre L2**: SHAP analysis sobre L2 es el resultado
  central; L1b aparece como sensitivity check.

**Limitación**: MODIS detecta consistentemente fuegos \>50 ha. Para
fuegos más pequeños, `label_l2 = 0` por defecto (correcto: si no fue
visible al satélite, no era catastrófico).

**Estado actual**: L2 *implementado* en `src/modis.py`. La propuesta
original (Google Earth Engine + MOD14A1/MYD14A1) se reemplazó por la
**NASA FIRMS Area API** (dataset `MODIS_SP`, swath L2 MOD14/MYD14),
significativamente más simple: HTTP directo con `MAP_KEY` gratuita, sin
SDK pesado. La especificación completa lista para paper está en la
[Sección 7](#implementacion-l2).

#### 5.2 Label L1b — Hectáreas P99 regional (proxy secundario + sensitivity analysis)

**Definición**: `label_l1b = 1` si
`superficie_quemada_total_ha ≥ P99_region`, sino `0`.

L1b cumple un rol secundario: sirve como análisis de sensibilidad
metodológico en el paper — *¿qué cambia si el target es un proxy
estadístico de superficie en lugar del criterio físico de EWE?*

**Justificación P99 regional (threshold del
proxy).**

**Por qué P99 y no P95 o P98:**

- P95 en Biobío sigue bajo umbrales operacionales de megaincendio:
  captura fuego no-trivial, no comportamiento catastrófico.
- P99 sobre 29.510 eventos produce un proxy cercano al 1% por
  construcción y preserva la rareza del fenómeno.
- P99 produce ~1% de imbalance — manejable con `scale_pos_weight` en
  XGBoost.
- Preserva el significado semántico de «mega» = top 1% de severidad por
  región.

**Por qué regional y no global:**

- P99 global del subset = 148.3 ha. Para Biobío el P99 regional es 89.2
  ha; para Maule es 204.0 ha y para Araucanía 250.3 ha. Un umbral único
  borraría diferencias regionales relevantes.
- Un umbral regional respeta el régimen de fuego de cada bioma chileno.
- El modelo aprenderá «¿qué hace que un fuego sea grande *para su
  región*?», más informativo que «¿es grande comparado con todo Chile?».

**Limitaciones de L1b como proxy de EWE.**
Superficie quemada y FLI están correlacionadas pero no son equivalentes.
Un fuego de 200 ha en bosque seco con viento extremo (alto FLI) y un
fuego de 200 ha en pastizal con propagación lenta (bajo FLI) reciben el
mismo label L1b, pero solo el primero es EWE según Tedim. Por eso L1b
funciona como sensitivity analysis, no como sustituto de L2.

#### 5.3 Alcance regional del estudio

No se aplica exclusión por baja muestra dentro del subset analítico:
Maule, Biobío y Araucanía tienen miles de eventos cada una. El filtro
regional es una decisión de alcance del estudio, no una limpieza por
insuficiencia estadística.

**Ñuble** se documenta como caso especial: para 2012-2018 no aparece
separada en la columna `region`, por lo que se mantiene absorbida en
Biobío histórico.

**Justificación.**

- El bbox actual cubre el 100% de los eventos georreferenciados de
  Maule, Biobío y Araucanía en 2012-2018.
- Las tres regiones concentran 29.510 eventos, suficiente para L1b y
  para modelado L2.
- El alcance evita mezclar regímenes de fuego de zonas fuera del foco
  centro-sur.

#### 5.4 Por qué L2 como target y L1b como proxy + sensitivity

La relación entre ambos labels **no es simétrica**: L2 es el objetivo
científico del proyecto, L1b es un instrumento que sirve dos propósitos
auxiliares.

**Rol de L2 (target objetivo)**:

- Es lo que el paper reporta como resultado central.
- Permite comparar drivers de EWE en Chile con literatura internacional
  (Australia, China, Mediterráneo) que usa el mismo criterio Tedim 2018.
- Captura comportamiento físico del fuego, no solo el outcome de
  superficie.

**Rol de L1b (proxy secundario + sensitivity)**:

1.  **Sensitivity analysis en el paper**: comparar SHAPs L1b vs L2
    cuantifica cuánto del resultado depende de la definición del target.
    Drivers consistentes entre ambos son robustos; drivers que solo
    aparecen en uno revelan artefactos del label.
2.  **Matriz de confusión L1b vs L2**: identifica eventos discrepantes
    (fuegos pequeños con FLI alto = pastizal extremo seco; fuegos
    grandes con FLI bajo = quemas que se descontrolaron lento). Insumo
    cualitativo para discusión del paper.

### 6. Pipeline a prueba de tontos

Procedimiento paso a paso para transformar el parquet crudo CONAF en un
dataset con dos labels binarios listo para XGBoost. Cada paso describe:
entrada, proceso, salida, y código ilustrativo en Python.

#### 6.1 Diagrama de flujo general

``` flow-diagram
INPUT: data/processed/conaf_enriched_2012_2018.parquet
       Run base 2012-2018 · 42.963 eventos CONAF
       Subset analítico · 29.510 eventos en Maule, Biobío y Araucanía
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ [1] Filtrar alcance regional del estudio                       │
│     Mantener: Maule, Biobío, Araucanía                         │
│     → 29.510 eventos                                           │
└────────────────────────────────────────────────────────────────┘
    │
    ├═══════════════════════════════════╗  ←── PATH PRINCIPAL (target objetivo)
    ▼                                   ▼
┌──────────────────────┐    ╔═════════════════════════════════════╗
│ [2a] Threshold       │    ║ [2b] L2: pipeline MODIS-FRP         ║
│      P99 regional    │    ║      (Tedim 2018, FLI ≥ 10 kW/m)    ║
│   (solo para L1b)    │    ║                                     ║
│                      │    ║  [2b.1] Filtrar candidatos > 50 ha  ║
│  Para cada región:   │    ║         (~396 en subset)            ║
│  P99 de              │    ║  [2b.2] Query MODIS vía FIRMS API   ║
│  superficie_quemada  │    ║  [2b.3] FRP → FLI (Wooster 2003)    ║
│  → {region: p99_ha}  │    ║  [2b.4] label_l2 = 1 si FLI ≥ 10k   ║
└──────────────────────┘    ║         ~30–80 positivos (estimado) ║
    │                       ╚═════════════════════════════════════╝
    ▼                                   │
┌──────────────────────┐                │
│ [3] L1b (proxy):     │                │
│   label_l1b = 1 si   │                │
│   area ≥ P99_region  │                │
│   ≈1% positivos      │                │
└──────────────────────┘                │
    │                                   │
    └───────────────┬───────────────────┘
                    ▼
┌────────────────────────────────────────────────────────────────┐
│ [4] Cross-check L1b vs L2 (matriz de confusión)                │
│     Insumo para sensitivity analysis del paper                 │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
OUTPUT: data/processed/conaf_labeled.parquet
        Columnas originales + label_l1b (proxy) + label_l2 (target)
        Listo para feature engineering (Sección 7)

LEYENDA:
  ║ ═  Path principal — L2 (target objetivo, paper reporta sobre este)
  │ ─  Path secundario — L1b (proxy secundario + sensitivity analysis)
  
```

#### 6.2 Paso 1 — Filtrar alcance regional

**Entrada**: `conaf_enriched_2012_2018.parquet` o
`conaf_enriched_latest.parquet` actualizado a 2012-2018.  
**Proceso**: mantener filas cuya `region` esté en
`{Maule, Biobío, Araucanía}`.  
**Salida**: DataFrame con 29.510 eventos antes del enriquecimiento
final.

    import pandas as pd

    REGIONES_ESTUDIO = {"Maule", "Biobío", "Araucanía"}

    df = pd.read_parquet("data/processed/conaf_enriched_latest.parquet")
    df = df[df["region"].isin(REGIONES_ESTUDIO)].reset_index(drop=True)
    print(f"Eventos retenidos: {len(df):,}")  # Esperado: 29.510

#### 6.3 Paso 2 — Calcular threshold P99 regional

**Entrada**: DataFrame filtrado.  
**Proceso**: para cada región, computar P99 sobre eventos con
`superficie_quemada_total_ha > 0` (excluye reportes sin superficie
informada).  
**Salida**: `dict {region: p99_ha}`.

    import numpy as np

    def p99_por_region(df: pd.DataFrame) -> dict:
        """Calcula P99 de superficie quemada por región (solo eventos positivos)."""
        thresholds = {}
        for region, grupo in df.groupby("region"):
            positivos = grupo.loc[grupo["superficie_quemada_total_ha"] > 0,
                                  "superficie_quemada_total_ha"]
            thresholds[region] = float(np.percentile(positivos, 99))
        return thresholds

    p99_dict = p99_por_region(df)
    # {'Araucanía': 250.3, 'Biobío': 89.2, 'Maule': 204.0}

#### 6.4 Paso 3 — Label L1b (hectáreas)

**Proceso**: para cada fila, comparar `superficie_quemada_total_ha` con
el threshold de su región. **Crítico**: eventos con superficie 0 o NaN
se etiquetan como 0 (no son megaincendios por construcción).

    def aplicar_l1b(df: pd.DataFrame, thresholds: dict) -> pd.Series:
        """Genera label binario L1b basado en threshold P99 regional."""
        threshold_por_fila = df["region"].map(thresholds)
        return (df["superficie_quemada_total_ha"] >= threshold_por_fila).astype(int)

    df["label_l1b"] = aplicar_l1b(df, p99_dict)
    print(f"Positivos L1b: {df['label_l1b'].sum():,} ({df['label_l1b'].mean()*100:.2f}%)")
    # Esperado: cercano a 1% por construcción

#### 6.5 Paso 4 — Label L2 (MODIS-FRP → FLI)

**Propuesta original (GEE) — superada.**
Esta sección documenta el enfoque inicial vía Google Earth Engine y se
conserva como referencia. La **implementación final usa NASA FIRMS**
(HTTP + MAP_KEY, sin GEE); su especificación formal está en la [Sección
7 (Methods)](#implementacion-l2).

El ETL productivo usa NASA FIRMS Area API y ejecuta el matching sobre
eventos CONAF con coordenadas y timestamp válidos. La guardia de
superficie mínima se aplica al asignar `label_l2`, no como feature de
entrenamiento.

##### 4.1 Ejecutar matching FIRMS

    from src.modis import download_firms_for_conaf, load_firms_csvs, match_modis_to_conaf, label_l2

    firms_paths = download_firms_for_conaf(conaf, bbox=download_bbox)
    modis_df = load_firms_csvs(firms_paths)
    matches = match_modis_to_conaf(enriched, modis_df)
    enriched = label_l2(enriched, matches)

##### 4.2 Query MODIS vía NASA FIRMS

Para cada bloque temporal, consultar `MODIS_SP` vía FIRMS Area API y
matchear detecciones por ventana espacial (5 km de radio alrededor de
`lat/lon`) y temporal (`fecha_hora_inicio` a
`fecha_hora_inicio + duracion` o + 7 días si duración es inválida).

    import ee
    ee.Initialize()

    def query_frp_modis(lat: float, lon: float,
                        start: str, end: str,
                        buffer_km: float = 5.0) -> float | None:
        """Devuelve el FRP máximo (MW) detectado en la ventana espacio-temporal."""
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_km * 1000)
        col = (ee.ImageCollection("MODIS/061/MOD14A1")
                 .merge(ee.ImageCollection("MODIS/061/MYD14A1"))
                 .filterBounds(region)
                 .filterDate(start, end)
                 .select("FirePower"))
        if col.size().getInfo() == 0:
            return None
        max_frp_img = col.max()
        val = max_frp_img.reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=region,
            scale=1000
        ).get("FirePower").getInfo()
        return val  # en MW

##### 4.3 Convertir FRP → FLI (Wooster et al. 2003)

La relación FRP→FLI no es directa; depende del largo del frente de
fuego. Aproximación operacional:

    def frp_to_fli(frp_mw: float, fire_front_length_m: float,
                   radiant_fraction: float = 0.17) -> float:
        """
        Convierte FRP (MW) a FLI (kW/m) usando relación de Wooster (2003).

        radiant_fraction: fracción de la potencia total emitida como radiación
                          (típicamente 0.13–0.20 según vegetación).
        fire_front_length_m: longitud del frente de fuego activo.
                             Estimación gruesa: sqrt(area_quemada_m2) si no se conoce.
        """
        # Potencia total = FRP / radiant_fraction
        total_power_kw = (frp_mw * 1000) / radiant_fraction
        # FLI = potencia por unidad de longitud de frente
        fli_kw_m = total_power_kw / fire_front_length_m
        return fli_kw_m

**Caveat.** La estimación del largo del
frente desde la superficie final es gruesa. Para una primera versión
`fire_front_length_m = sqrt(area_ha * 10000) / 4` (asume forma
aproximadamente cuadrada con frente activo en un lado). Refinamientos
posteriores pueden usar el shapefile del perímetro de incendio si está
disponible.

##### 4.4 Aplicar threshold y asignar label

    THRESHOLD_FLI = 10_000  # kW/m, definición Tedim cat 5

    candidatos["frp_max_mw"] = candidatos.apply(
        lambda r: query_frp_modis(r["latitud"], r["longitud"],
                                  r["fecha_hora_inicio_utc"],
                                  r["fecha_hora_inicio_utc"] + pd.Timedelta(days=7)),
        axis=1
    )
    candidatos["fire_front_m"] = (candidatos["superficie_quemada_total_ha"] * 10_000) ** 0.5 / 4
    candidatos["fli_kw_m"] = candidatos.apply(
        lambda r: frp_to_fli(r["frp_max_mw"], r["fire_front_m"]) if pd.notna(r["frp_max_mw"]) else None,
        axis=1
    )

    # Asignar label_l2 al dataset completo
    df["label_l2"] = 0
    positivos_l2 = candidatos[candidatos["fli_kw_m"] >= THRESHOLD_FLI].index
    df.loc[positivos_l2, "label_l2"] = 1

#### 6.6 Paso 5 — Cross-check L1b vs L2

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(df["label_l1b"], df["label_l2"])
    print("Matriz de confusión L1b (filas) vs L2 (columnas):")
    print(cm)

    # Eventos donde difieren — para revisión manual
    discrepancias = df[df["label_l1b"] != df["label_l2"]].copy()
    discrepancias["tipo"] = discrepancias.apply(
        lambda r: "L1b=1, L2=0 (grande pero no extremo)"
                  if r["label_l1b"] == 1 else "L1b=0, L2=1 (chico pero extremo)",
        axis=1
    )
    discrepancias.to_csv("data/processed/label_discrepancies.csv", index=False)

**Interpretación de discrepancias.**

- **L1b=1, L2=0**: fuego grande en hectáreas pero sin firma extrema en
  MODIS. Probablemente fuego lento que cubrió mucha superficie sin alta
  intensidad (pastizal patagónico, plantación joven).
- **L1b=0, L2=1**: fuego pequeño en hectáreas pero con FLI alto. Puede
  ser un fuego intenso detenido a tiempo, o un falso positivo del
  MODIS-FRP.

#### 6.7 Output final

    df.to_parquet("data/processed/conaf_labeled.parquet")
    print(f"L1b positivos: {df['label_l1b'].sum():,}")
    print(f"L2 positivos:  {df['label_l2'].sum():,}")
    print(f"Ambos = 1:     {((df['label_l1b']==1) & (df['label_l2']==1)).sum():,}")
    print(f"Solo L1b = 1:  {((df['label_l1b']==1) & (df['label_l2']==0)).sum():,}")
    print(f"Solo L2 = 1:   {((df['label_l1b']==0) & (df['label_l2']==1)).sum():,}")

### 7. Implementación L2 — Methods section (paper-ready)

Esta sección documenta la implementación final del label L2 (módulo
`src/modis.py`), redactada como *Methods section* lista para el paper.
Define el producto satelital, la conversión física FRP→FLI, el matching
espacio-temporal con los eventos CONAF, la regla de decisión del label,
sus parámetros, limitaciones y referencias.

#### 7.1 Datos

**Producto MODIS específico**: MODIS Thermal Anomalies/Fire **Level 2
5-Min Swath 1 km** — productos **MOD14** (Terra) y **MYD14** (Aqua),
Collection 6.1. *No* se usan los composites diarios L3 (MOD14A1/MYD14A1)
ni los de 8 días (MOD14A2/MYD14A2): el swath L2 preserva el FRP
instantáneo de cada detección individual, insumo directo de la ecuación
de Wooster (2003). Los composites agregan o promedian la potencia
radiativa y pierden el pico que define la intensidad EWE.

**Acceso**: NASA FIRMS Area API (HTTP, `MAP_KEY` gratuita), dataset
`MODIS_SP` (Standard Processing). FIRMS extrae los fire pixels del swath
L2 y los expone como CSV con columnas
`latitude, longitude, frp, acq_date, acq_time, satellite, instrument, confidence, daynight, brightness, bright_t31, scan, track`.
Esto evita la manipulación de archivos HDF crudos desde LAADS DAAC.

**Sensores y cobertura**:

- Terra: paso aproximado 10:30 LST (órbita descendente).
- Aqua: paso aproximado 13:30 LST (órbita ascendente).
- Combinado: típicamente 2–4 observaciones/día/punto en latitudes
  medias.
- Resolución espacial: 1 km en nadir; degrada a ~10 km en los bordes del
  swath.
- Cobertura temporal de `MODIS_SP`: 2002–presente, con lag de varios
  meses respecto al near-real-time.

**Limitación documentada (Giglio et al.
2016).** El algoritmo de detección sub-reporta fuegos \< ~50 ha por
dilución térmica en píxeles de 1 km²; la cobertura es intermitente por
nubosidad, alto ángulo cenital solar o saturación del detector en fuegos
muy intensos.

#### 7.2 Conversión FRP → FLI (interpretación *peak local*)

Ecuación de Wooster (2003) aplicada al **píxel MODIS más caliente** del
frente. Para el evento \\i\\:

\$\$\text{FLI}\_i\\\[\text{kW m}^{-1}\] = \frac{\text{FRP}\_i \cdot
10^{3} / \eta_r}{L\_{\text{px}}}\$\$

donde \\\text{FRP}\_i = \max\_{j \in \mathcal{M}\_i} \text{FRP}\_j\\ es
la potencia radiativa máxima (MW) entre las detecciones MODIS matcheadas
al evento \\i\\, \\\eta_r = 0.17\\ es la fracción radiante por defecto
(Wooster 2003; rango 0.13–0.20) y \\L\_{\text{px}} = 1000\\ m es la
longitud nominal del píxel MODIS en nadir.

**Por qué *peak local* y no front length desde
el área CONAF.** La superficie quemada CONAF
(`superficie_quemada_total_ha`) es el **área acumulada final** del
incendio (días o semanas), mientras que el FRP es una medida
**instantánea** de un pase satelital. Derivar el largo del frente de un
área circular equivalente daría frentes de 10–45 km en megaincendios y,
al repartir el FRP de un único píxel sobre esa longitud, diluiría la FLI
hasta hacer \\\text{label\\l2}=0\\ incluso para eventos categóricamente
EWE (p. ej. Las Máquinas, 159.812 ha, requeriría \\\text{FRP}\approx
77\\ GW en un píxel para superar el umbral). La interpretación *peak
local* es dimensionalmente coherente —un píxel de 1 km² ↔ su propia
longitud— y define \\\text{label\\l2}=1\\ cuando **alguna parte del
frente** alcanzó intensidad EWE.

**Verificación dimensional.**
\\P\_{\text{total}}\\\[\text{W}\] = \text{FRP}\\\[\text{MW}\]\cdot
10^{6} / \eta_r\\; luego \\\text{FLI}\\\[\text{W m}^{-1}\] =
P\_{\text{total}} / L\_{\text{px}}\\; finalmente
\\\text{FLI}\\\[\text{kW m}^{-1}\] = \text{FLI}\\\[\text{W m}^{-1}\] /
10^{3}\\. Ejemplo: \\\text{FRP}=170\\ MW, \\L\_{\text{px}}=1000\\ m,
\\\eta_r=0.17 \Rightarrow \text{FLI}=1000\\ kW m⁻¹. El umbral EWE de
10.000 kW m⁻¹ se alcanza en \\\text{FRP} \approx 1700\\ MW, de modo que
**\\\text{label\\l2}=1 \iff \text{FRP}\_i \gtrsim 1700\\ MW**.

#### 7.3 Matching espacio-temporal CONAF ↔ MODIS

Para cada evento CONAF \\i\\ con timestamp \\t_i\\ (UTC) y coordenadas
\\(\phi_i, \lambda_i)\\, el conjunto de detecciones MODIS asociadas es:

\$\$\mathcal{M}\_i = \\\\ j : d\_{\text{hav}}(\phi_i, \lambda_i, \phi_j,
\lambda_j) \le r\_{\max} \\\wedge\\ \|t_i - t_j\| \le \Delta t\_{\max}
\\\\\$\$

donde \\d\_{\text{hav}}\\ es la distancia haversine, \\r\_{\max} = 5\\
km y \\\Delta t\_{\max} = 24\\ h. Estos defaults cubren la resolución
nominal MODIS (~1 km nadir, hasta ~10 km en bordes) con margen, y la
cadencia combinada Terra+Aqua (2–4 observaciones/día/ punto en latitudes
medias). La agregación por evento usa \\\text{FRP}\_i = \max\_{j \in
\mathcal{M}\_i} \text{FRP}\_j\\, defendible porque Wooster (2003)
vincula la FLI con la intensidad radiativa instantánea del frente.

#### 7.4 Definición operacional del label L2

\$\$\text{label\\l2}\_i = \mathbb{1}\\\left\[\\\text{FLI}\_i \ge
\text{FLI}\_{\text{EWE}} \\\wedge\\ A_i \ge A\_{\min}\\\right\], \qquad
\text{FLI}\_{\text{EWE}} = 10\\000\\ \text{kW m}^{-1},\\ A\_{\min} =
50\\ \text{ha}\$\$

Los eventos sin match MODIS (\\\mathcal{M}\_i = \varnothing\\) reciben
\\\text{FRP}\_i = \text{NaN} \Rightarrow \text{FLI}\_i = \text{NaN}
\Rightarrow \text{label} = 0\\. Interpretación: la ausencia de detección
satelital del píxel térmico es evidencia (débil pero válida) de que el
frente no superó los umbrales radiativos asociados a EWE. El umbral
10.000 kW m⁻¹ corresponde a Tedim et al. (2018) categoría 5 (límite
inferior EWE; Tabla 3 del paper original, reproducida en la [Sección
3](#tedim)).

**Guardia de coherencia área–FLI
(\\A\_{\min}\\).** El segundo término exige una superficie quemada
mínima de 50 ha (el límite de detección de MODIS). Sin esta guardia, el
radio de matching de 5 km puede atribuir el FRP de un megaincendio
vecino a un evento CONAF pequeño y producir falsos positivos físicamente
incoherentes (un EWE no ocupa \< 50 ha). En cada corrida, reportar
cuántos candidatos positivos descarta esta guardia para auditar falsos
positivos por atribución espacial.

#### 7.5 Tabla de parámetros

| Parámetro | Símbolo | Valor | Fuente / justificación |
|----|----|----|----|
| Radio matching espacial | \\r\_{\max}\\ | 5 km | Resolución MODIS (1 km nadir) + margen off-nadir |
| Ventana matching temporal | \\\Delta t\_{\max}\\ | 24 h | Cobertura combinada Terra+Aqua, robusta a nubosidad parcial |
| Fracción radiante | \\\eta_r\\ | 0.17 | Wooster (2003) default; rango 0.13–0.20 |
| Longitud del píxel (front length) | \\L\_{\text{px}}\\ | 1000 m | Tamaño píxel MODIS nadir (interpretación peak local) |
| Umbral EWE | \\\text{FLI}\_{\text{EWE}}\\ | 10.000 kW m⁻¹ | Tedim et al. (2018) categoría 5+ |
| Superficie mínima (guardia) | \\A\_{\min}\\ | 50 ha | Límite detección MODIS; descarta falsos positivos por atribución espacial |
| Dataset MODIS | — | MODIS_SP C6.1 | Standard Processing, calibración consistente |
| Bbox descarga | CHILE_BBOX | 74°W, 42°S, 70°W, 34°S | 4 regiones del subset enriquecido |

#### 7.6 Limitaciones reconocidas

1.  **Subestimación por nubosidad o pase fuera de ventana**: si Terra y
    Aqua no observan el evento en su fase activa, `label_l2 = 0` (falso
    negativo posible).
2.  **FLI puntual, no integrada sobre el frente**: la interpretación
    peak local mide la intensidad del píxel más caliente, no la del
    frente completo. Marca correctamente si *alguna parte* del frente
    alcanzó intensidad EWE, pero no cuantifica su extensión; un frente
    largo de intensidad moderada uniforme podría no detectarse como EWE.
3.  **\\\eta_r\\ constante**: la vegetación heterogénea (matorral vs
    bosque vs pastizal) modula la fracción radiante real entre 0.13 y
    0.20; sensibilidad explorable en análisis posterior.
4.  **Resolución MODIS limita la atribución**: hotspots de fuegos
    cercanos pueden atribuirse a un evento CONAF distinto dentro del
    radio de 5 km. La guardia de coherencia área–FLI (\\A\_{\min} = 50\\
    ha, §7.4) mitiga el caso más severo —FRP de un megaincendio vecino
    asignado a un fuego pequeño— pero no elimina la ambigüedad entre dos
    fuegos grandes y próximos.
5.  **Cobertura temporal sesgada**: el subset usa Terra+Aqua
    exclusivamente; los sensores geoestacionarios (GOES-16/17)
    capturarían la dinámica intra-día, pero quedan fuera de alcance.

#### 7.7 Referencias

Referencias clave ya listadas en la [Sección de
Referencias](#referencias): Wooster et al. (2003) \[6\] — conversión
FRP↔FLI; Tedim et al. (2018) \[1\] — definición y umbral EWE; NASA LP
DAAC MOD14/MYD14 \[8\] — producto satelital. Referencias adicionales
propias de esta sección: Giglio et al. (2016) \[15\] — algoritmo de
detección Collection 6 y sus limitaciones; Kaufman et al. (1998) \[16\]
— fundamentos del fire monitoring con MODIS y del FRP.

### 8. Features para XGBoost (sin leakage)

El modelo predice en el momento de ignición, por lo que solo puede usar
variables medibles **antes o exactamente al momento de inicio** del
fuego. Toda información generada por el fuego mismo introduce *data
leakage* y debe excluirse.

##### ✓ Permitidas (no leakage)

**ERA5-Land al momento de ignición:**

- `t2m_celsius` — temperatura a 2 m
- `d2m_celsius` — punto de rocío a 2 m
- `relative_humidity` — humedad relativa
- `vpd_hpa` — déficit presión vapor
- `wind_speed` — velocidad viento 10 m
- `wind_direction` — dirección viento
- `swvl1`, `swvl2`, `swvl3`, `swvl4` — humedad suelo 4 capas
- `tp_mm` — precipitación acumulada
- `pev`, `e`, `evavt` — evaporación
- `ssrd` — radiación solar superficie
- `stl1..4_celsius` — temperatura suelo 4 capas

**Estáticas geográficas:**

- `latitud`, `longitud`

**Estacionales (derivar de `fecha_hora_inicio`):**

- `mes` (1–12)
- `hora` (0–23)
- `dia_semana` (0–6)

**Vegetativas (estado del paisaje):**

- `lai_hv`, `lai_lv` — Leaf Area Index alta/baja vegetación
- `cvh`, `cvl` — fracción cobertura alta/baja
- `tvh`, `tvl` — tipo alta/baja vegetación

##### ✗ Prohibidas (data leakage)

**Outcomes del fuego:**

- `duracion_minutos`
  

  Se conoce solo cuando termina el fuego.

  
- `superficie_quemada_total_ha`
  

  Es prácticamente el target — leakage total.

  
- `superficie_quemada_pino_*`, `_eucalipto_`, `_matorral_`, etc.
  

  Mismo problema: outcome del fuego.

  

**Etiquetas operacionales ex-post:**

- `alerta`
  

  La declaración de alerta (amarilla/roja) se decide tras evaluar el
  fuego, no antes.

  
- `escenario`
  

  CONAF codifica el escenario IFor-Vn/IFor-PI/etc. al cerrar el reporte.
  Aunque conceptualmente describe el contexto, en la práctica es
  ex-post.

  

**Metadata identificadora:**

- `nombre`
  

  No es señal predictiva.

  
- `fecha_hora_inicio` directo
  

  Descomponer en mes/hora/día semana sí, pero no usar el timestamp
  absoluto (introduce dependencia temporal espuria).

  

**Metadatos de matching ERA5:**

- `era5_dist_km`, `era5_dt_hours`, `era5_match_quality`
  

  Describen la calidad del cruce espacio-temporal CONAF↔ERA5. Son útiles
  para diagnóstico y filtrado de matches malos, pero no son señales
  físicas del fuego. No usar como features.

  

#### 8.1 Class imbalance

Con L1b a P99 regional sobre 29.510 eventos, el proxy produce un
imbalance cercano a 1%. Con L2, usar los conteos reales de `label_l2`
del parquet enriquecido 2012-2018; por diseño sigue siendo un problema
severamente desbalanceado.

##### Opciones de mitigación en XGBoost

- **`scale_pos_weight`**: factor multiplicador para el gradiente de la
  clase positiva. Setting recomendado:
  `scale_pos_weight = n_neg / n_pos` calculado desde el target final.
- **Focal loss** (custom objective): pondera más los ejemplos difíciles.
  Útil cuando los positivos son heterogéneos.
- **Undersampling de negativos**: muestrear aleatoriamente el 5-10% de
  los negativos. Pierde información pero acelera entrenamiento.
- **SMOTE / synthetic minorities**: *no recomendado* aquí — sintetizar
  megaincendios artificiales viola la naturaleza espacio-temporal del
  problema.

##### Métricas de evaluación

**No usar accuracy** con este imbalance — un modelo que predice todo
cero obtiene 99% accuracy. Usar en su lugar:

- **Precision-Recall AUC**: más sensible al imbalance que ROC AUC.
- **F1 macro**: promedio balanceado entre clases.
- **Recall a precisión fija** (e.g., recall@precision=0.5): pregunta
  operacional: «¿qué fracción de megaincendios reales detectamos si
  aceptamos 50% de falsos positivos?»

#### 8.2 Train/test split

**No usar split aleatorio** como evaluación principal: los incendios
tienen correlación temporal y espacial, y enero-febrero 2017 concentra
eventos extremos.

- **Split temporal**: entrenar con 2012-2017 y validar/testear en 2018.
- **Split espacial complementario**: leave-one-region-out entre Maule,
  Biobío y Araucanía.
- **Validación de estrés**: reportar métricas específicas para
  enero-febrero 2017 por concentración de eventos extremos.

### 9. Referencias

1.  **Tedim, F., Leone, V., Amraoui,
    M., et al. (2018)**. *Defining Extreme Wildfire Events:
    Difficulties, Challenges, and Impacts.* *Fire*, 1(1), 9. MDPI.  
    DOI: [10.3390/fire1010009](https://doi.org/10.3390/fire1010009)  
    Texto completo: [mdpi.com](https://www.mdpi.com/2571-6255/1/1/9) ·
    [USDA mirror](https://research.fs.usda.gov/treesearch/57472) · copia
    local: `references/Tedim2018_DefiningEWE.pdf`
2.  **Byram, G. M. (1959)**.
    *Combustion of Forest Fuels.* In: Davis, K. P. (Ed.), *Forest Fire:
    Control and Use*. McGraw-Hill, New York, pp. 61–89.  
    Define formalmente la fireline intensity (FLI =
    H·w·R).
3.  **Rothermel, R. C. (1972)**. *A
    Mathematical Model for Predicting Fire Spread in Wildland Fuels.*
    USDA Forest Service Research Paper INT-115.  
    [USDA Treesearch](https://www.fs.usda.gov/research/treesearch/32533)
4.  **Alexander, M. E. (1982)**.
    *Calculating and Interpreting Forest Fire Intensities.* *Canadian
    Journal of Botany*, 60(4), 349–357.  
    Establece la relación FLI ≈ 259.833 ·
    FL^(2.174).
5.  **Alexander, M. E., &
    Lanoville, R. A. (1989)**. *Predicting Fire Behavior in the Black
    Spruce-Lichen Woodland Fuel Type of Western and Northern Canada.*
    Forestry Canada Northern Forestry Centre.  
    Origen de la clasificación FLI categorías 1–4
    que Tedim extiende a 7.
6.  **Wooster, M. J., Zhukov, B., &
    Oertel, D. (2003)**. *Fire radiative energy for quantitative study
    of biomass burning.* *Remote Sensing of Environment*, 86(1),
    83–107.  
    DOI:
    [10.1016/S0034-4257(03)00070-1](https://doi.org/10.1016/S0034-4257(03)00070-1)  
    Establece la relación FRP↔consumo de
    biomasa↔FLI.
7.  **Lannom, K. B., Tinkham, W. T.,
    Newingham, B. A., et al. (2014)**. *Defining extreme wildland fires
    using geospatial and ancillary metrics.* *International Journal of
    Wildland Fire*, 23(3), 322–337.  
    DOI: [10.1071/WF13065](https://doi.org/10.1071/WF13065)  
    Propone umbrales basados en percentiles 90/95/99
    de área quemada.
8.  **NASA LP DAAC**. *MODIS
    MOD14/MYD14 Thermal Anomalies and Fire — User Guide.*
    [lpdaac.usgs.gov](https://lpdaac.usgs.gov/products/mod14v061/)  
    Documentación oficial productos MODIS
    Fire.
9.  **Google Earth Engine — MODIS
    Fire Collection.**
    [MOD14A1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD14A1)
    ·
    [MYD14A1](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD14A1)  
    API para consulta programática de FRP.
10. **Copernicus Climate Change
    Service (C3S)**. *ERA5-Land hourly data from 1950 to present.*
    [Climate Data
    Store](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land)
11. **CONAF / itrend — Datos para
    Resiliencia.** *Registro de incendios forestales en Chile.*  
    DOI: [10.71578/UXAUN5](https://doi.org/10.71578/UXAUN5) ·
    [datospararesiliencia.cl](https://datospararesiliencia.cl)
12. **Pedernera, P., et al.
    (2017)**. *Modelos de combustibles para Chile.* CONAF / INIA.  
    Carga combustible por tipo de vegetación nativa
    chilena.
13. **Scott, J. H., & Burgan, R. E.
    (2005)**. *Standard Fire Behavior Fuel Models: A Comprehensive Set
    for Use with Rothermel's Surface Fire Spread Model.* USDA Forest
    Service General Technical Report RMRS-GTR-153.  
    [USDA Treesearch](https://www.fs.usda.gov/research/treesearch/9521)
14. **Jain, P., Coogan, S. C. P.,
    Subramanian, S. G., et al. (2020)**. *A review of machine learning
    applications in wildfire science and management.* *Environmental
    Reviews*, 28(4), 478–505.  
    DOI: [10.1139/er-2020-0019](https://doi.org/10.1139/er-2020-0019)  
    Estado del arte ML en ciencia del fuego, incluye
    análisis de XGBoost.
15. **Giglio, L., Schroeder, W., &
    Justice, C. O. (2016)**. *The Collection 6 MODIS active fire
    detection algorithm and fire products.* *Remote Sensing of
    Environment*, 178, 31–41.  
    DOI:
    [10.1016/j.rse.2016.02.054](https://doi.org/10.1016/j.rse.2016.02.054)  
    Algoritmo de detección C6 y limitaciones
    (sub-reporte de fuegos pequeños).
16. **Kaufman, Y. J., Justice, C.
    O., Flynn, L. P., et al. (1998)**. *Potential global fire monitoring
    from EOS-MODIS.* *Journal of Geophysical Research*, 103(D24),
    32215–32238.  
    DOI: [10.1029/98JD01644](https://doi.org/10.1029/98JD01644)  
    Fundamentos del fire monitoring satelital con
    MODIS y del FRP.

### Apéndice A. Glosario CONAF (códigos de escenario)

El campo `escenario` de CONAF clasifica el contexto físico-geográfico
del incendio. **No usar como feature** (es codificado ex-post — ver
[Sección 7](#features)), pero útil para análisis exploratorio.

| Código | Nombre | Descripción |
|----|----|----|
| `IFor-PI` | Plantaciones | Incendio en plantaciones forestales (pino, eucalipto comercial) |
| `IFor-Vn` | Vegetación natural | Bosque nativo o matorral natural |
| `IFIUr-Fo` | Interfaz urbano-forestal | Fuego en zona Wildland-Urban Interface (WUI) |
| `IFASP` | Área Silvestre Protegida | Parque nacional, reserva, monumento natural |
| `IFCo` | Cordillera | Sobre 1.000 m.s.n.m., alta cordillera |
| `IFCSo` | Conflicto social | Asociado a contexto de conflicto territorial |
| `IFSu` | Subterráneo | Combustión bajo superficie (turbas, raíces) |
| `IFIns` | Insular | En isla (Juan Fernández, Isla de Pascua, Chiloé) |
| `No definido` | — | Sin clasificación registrada |

### Apéndice B. Por qué no usar MODIS-FRP como feature

Aunque MODIS-FRP es central para construir el label L2, **no puede ser
feature del modelo XGBoost**. Razones:

1.  **Leakage temporal directo**. FRP se mide *durante* el fuego activo,
    no en el momento de ignición. Para t=0 (instante de inicio) no hay
    aún ningún píxel MODIS detectado. Incluirlo como feature equivale a
    darle al modelo el resultado.
2.  **Construye el label**. L2 = (FLI derivada de FRP ≥ 10.000). Usar
    FRP como feature simultáneamente como target sería un loop trivial:
    el modelo aprendería `label = FRP ≥ threshold` con 100% accuracy y
    SHAP totalmente dominada por FRP.
3.  **Caso de uso**. El modelo se quiere usar operacionalmente al
    momento de ignición, cuando todavía no hay paso del satélite. MODIS
    pasa cada 6–12 horas; el fuego ya está corriendo cuando llega la
    primera detección.

MODIS-FRP es un **instrumento de etiquetado**, no una feature
predictiva. Mismo principio para `duracion_minutos` y
`superficie_quemada_total_ha`.

### Apéndice C. Snippet completo Google Earth Engine

Código de referencia (Python) para consultar FRP MODIS para una lista de
eventos. Requiere autenticación previa: `earthengine authenticate`.

    import ee
    import pandas as pd
    from tqdm import tqdm

    ee.Initialize()

    def query_modis_frp_batch(events_df: pd.DataFrame,
                              buffer_km: float = 5.0,
                              window_days: int = 7) -> pd.DataFrame:
        """
        Para cada evento en `events_df` (con columnas latitud, longitud,
        fecha_hora_inicio_utc, duracion_minutos), consulta el FRP máximo
        detectado por MOD14A1 + MYD14A1 en una ventana espacio-temporal.

        Returns: events_df con columna adicional 'frp_max_mw'.
        """
        results = []

        for idx, row in tqdm(events_df.iterrows(), total=len(events_df)):
            point = ee.Geometry.Point([row["longitud"], row["latitud"]])
            region = point.buffer(buffer_km * 1000)

            start = pd.to_datetime(row["fecha_hora_inicio_utc"])
            # Ventana = max(duración real, window_days)
            if pd.notna(row["duracion_minutos"]) and row["duracion_minutos"] > 0:
                end = start + pd.Timedelta(minutes=row["duracion_minutos"])
            else:
                end = start + pd.Timedelta(days=window_days)

            try:
                col_terra = ee.ImageCollection("MODIS/061/MOD14A1")
                col_aqua = ee.ImageCollection("MODIS/061/MYD14A1")
                col = (col_terra.merge(col_aqua)
                         .filterBounds(region)
                         .filterDate(start.isoformat(), end.isoformat())
                         .select("FirePower"))

                if col.size().getInfo() == 0:
                    results.append(None)
                    continue

                max_img = col.max()
                val = max_img.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=region,
                    scale=1000,
                    maxPixels=1e6
                ).get("FirePower").getInfo()
                results.append(val)
            except Exception as e:
                print(f"Error en evento {idx}: {e}")
                results.append(None)

        events_df = events_df.copy()
        events_df["frp_max_mw"] = results
        return events_df

    # Uso:
    # candidatos = df[df["superficie_quemada_total_ha"] > 50]
    # resultado = query_modis_frp_batch(candidatos)
    # resultado.to_parquet("data/processed/modis_frp_results.parquet")

**Consideraciones operacionales.**

- Earth Engine tiene cuotas de uso. Para ~2.000 eventos candidatos,
  considera procesar en batches y guardar resultados intermedios.
- Eventos con `fecha_hora_inicio_utc` nulo (~6 casos en el dataset)
  requieren tratamiento manual.
- Para eventos \< 2002, MOD14 (Terra) no está disponible; solo MOD14 hay
  desde febrero 2000. MYD14 (Aqua) desde julio 2002.

Documento generado para el proyecto XAI / clasificación de
megaincendios.  
Última actualización: Mayo 2026 · Versión 1.0 · [PDF Tedim 2018
local](../references/Tedim2018_DefiningEWE.pdf)

---

## Parte II — Sección «Data» del paper (III. Data)

Standalone English draft for the paper data section. Citation keys are
LaTeX-compatible.

This section describes the dataset assembled to study extreme wildfire
behavior in Chile. The dataset integrates three sources at the level of
individual fire events: the CONAF historical wildfire inventory, which
provides the event anchors (ignition location and time); the ERA5-Land
reanalysis, which supplies the meteorological and land-surface
covariates used as ex-ante predictors; and MODIS/FIRMS active-fire
detections, which provide the physically grounded extreme-fire label
following the Extreme Wildfire Event (EWE) paradigm.

Each CONAF event is enriched by spatio-temporal matching to ERA5-Land
and to MODIS/FIRMS, producing one row per event with a strict separation
between ex-ante predictors and post-ignition target information. The
study focuses on the four south-central regions defined in Section
III-A; the empirical figures reported in this section correspond to the
2016-2017 enriched data, comprising 12,381 events in the national
inventory, of which 8,664 fall in the study regions. The intended
training window extends to the 2012-2018 seasons.

### III-A. Study Area and Regional Scope

The study is restricted to the south-central macrozone of Chile, defined
by the geographic bounding box `lat ∈ [−42°, −34°]` and
`lon ∈ [−74°, −70°]`, which covers the regions of Maule, Biobío,
Araucanía, and O'Higgins. This macrozone concentrates the large majority
of the country's wildfire impact: over the 2002-2003 to 2019-2020
seasons it accounts for 72.7% of all recorded fire events (79,886 of
109,947) and 75.6% of the total burned area (1,181,479 of 1,562,870 ha),
as summarized in Table I \cite{mcwethy2018Landscape,ubeda2016Chile}.
O'Higgins and Maule are particularly relevant because they contribute a
far larger share of burned area than of event counts, indicating a
concentration of high-severity fires.

**Administrative note.** Ñuble is a new region created in 2018, when it
was split from the Biobío region. For the 2002-2017 seasons its
territory is recorded under Biobío, so this study uses the Biobío label
to represent that territory consistently across the full time series.

This regional scope also defines the spatial domain over which ERA5-Land
covariates are extracted. CONAF events recorded elsewhere in the
national inventory fall outside this domain, receive no ERA5-Land
covariates, and are flagged as `out_of_coverage` and excluded from
modeling (Section III-D).

| Region                 | Events     | Events % | Burned area (ha) | Burned area % |
|------------------------|------------|----------|------------------|---------------|
| Maule                  | 10,111     | 9.2      | 386,081          | 24.7          |
| Biobío                 | 45,720     | 41.6     | 356,640          | 22.8          |
| Araucanía              | 19,689     | 17.9     | 214,572          | 13.7          |
| O'Higgins              | 4,366      | 4.0      | 224,185          | 14.3          |
| **Four study regions** | **79,886** | **72.7** | **1,181,479**    | **75.6**      |
| Rest of Chile          | 30,061     | 27.3     | 381,391          | 24.4          |
| National total         | 109,947    | 100.0    | 1,562,870        | 100.0         |

    \begin{table}[t]
    \centering
    \caption{Concentration of wildfire activity in the four study regions. CONAF inventory, 2002--2003 to 2019--2020 seasons.}
    \label{tab:study-regions}
    \begin{tabular}{lrrrr}
    \toprule
    Region & Events & Events \% & Burned area (ha) & Burned area \% \\
    \midrule
    Maule      & 10,111 & 9.2  & 386,081   & 24.7 \\
    Biobío     & 45,720 & 41.6 & 356,640   & 22.8 \\
    Araucanía  & 19,689 & 17.9 & 214,572   & 13.7 \\
    O'Higgins  & 4,366  & 4.0  & 224,185   & 14.3 \\
    \midrule
    \textbf{Four study regions} & \textbf{79,886} & \textbf{72.7} & \textbf{1,181,479} & \textbf{75.6} \\
    Rest of Chile               & 30,061 & 27.3 & 381,391   & 24.4 \\
    National total              & 109,947 & 100.0 & 1,562,870 & 100.0 \\
    \bottomrule
    \end{tabular}
    \end{table}

### III-B. CONAF: Historical Fire Event Records

The primary event inventory is the historical wildfire record maintained
by the Chilean National Forestry Corporation (CONAF), distributed
through the Datos para Resiliencia platform \cite{conaf_dataset}. The
local cleaned version used in this project contains 109,947 wildfire
records from the 2002-2003 through 2019-2020 fire seasons. Each row
represents a reported ignition or fire event and includes administrative
location, ignition timing, event metadata, final burned-area attributes,
and point coordinates for the estimated ignition location.

For modeling, the CONAF inventory is treated as an event anchor rather
than as a complete predictor matrix. Latitude, longitude, and ignition
time define the spatio-temporal key used to enrich each event with
meteorological and land-surface covariates. In contrast, final burned
area, operational alert state, duration, and scenario labels are
recorded after the fire has developed and are therefore not valid
ex-ante predictors for a model intended to support resource allocation
at or near ignition.

| \# | Variable | Role | Use in this study |
|----|----|----|----|
| 1 | `region` | metadata | Administrative context; may support stratified analysis. |
| 2 | `provincia` | metadata | Intermediate administrative context. |
| 3 | `comuna` | spatial key | Local administrative spatial context. |
| 4 | `temporada` | temporal key | Fire-season identifier for temporal splits and reporting. |
| 5 | `nombre` | metadata | Event name; retained for traceability, not as a predictor. |
| 6 | `fecha` | temporal key | Ignition date used to build the event timestamp. |
| 7 | `hora_inicio` | temporal key | Ignition hour used to build the event timestamp. |
| 8 | `duracion_minutos` | excluded ex-post variable | Known only after the event ends. |
| 9 | `alerta` | excluded ex-post variable | Operational response state; not available at ignition. |
| 10 | `escenario` | excluded ex-post variable | Operational/event classification; may encode later fire behavior. |
| 11 | `causa` | possible leakage | Cause attribution can be assigned after investigation. |
| 12 | `superficie_quemada_pino_a_ha` | outcome | Final burned area by fuel class. |
| 13 | `superficie_quemada_pino_b_ha` | outcome | Final burned area by fuel class. |
| 14 | `superficie_quemada_pino_c_ha` | outcome | Final burned area by fuel class. |
| 15 | `superficie_quemada_eucalipto_ha` | outcome | Final burned area by fuel class. |
| 16 | `superficie_quemada_otras_plantas_ha` | outcome | Final burned area by fuel class. |
| 17 | `superficie_quemada_arbolado_ha` | outcome | Final burned area by fuel class. |
| 18 | `superficie_quemada_matorral_ha` | outcome | Final burned area by fuel class. |
| 19 | `superficie_quemada_pastizal_ha` | outcome | Final burned area by fuel class. |
| 20 | `superficie_quemada_agricola_ha` | outcome | Final burned area by fuel class. |
| 21 | `superficie_quemada_desechos_ha` | outcome | Final burned area by fuel class. |
| 22 | `superficie_quemada_total_ha` | outcome | Final burned area; used only for descriptive analysis and the L1 proxy. |
| 23 | `latitud` | spatial key | Point coordinate for event enrichment. |
| 24 | `longitud` | spatial key | Point coordinate for event enrichment. |
| 25 | `datum` | metadata | Coordinate reference metadata. |
| 26 | `geometry` | spatial key | Geometric representation of the event point. |
| 27 | `fecha_hora_inicio` | temporal key | Local ignition timestamp in Chilean time. |
| 28 | `fecha_hora_inicio_utc` | temporal key | UTC timestamp used for ERA5-Land and FIRMS matching. |

### III-C. Exploratory Analysis of CONAF Records

The exploratory analysis supports the premise that wildfire impact in
Chile is highly concentrated in the upper tail. In the cleaned CONAF
inventory, 1,240 events account for 80% of the total burned area,
corresponding to approximately 1.13% of all records. The median burned
area is 0.30 ha, whereas the 99th percentile is 154.0 ha and the maximum
recorded event reaches 159,812.58 ha. This asymmetry is consistent with
the self-organized criticality framing used in forest-fire literature
\cite{malamud1998}.

This distributional evidence is used only to motivate the learning
problem. It does not imply that final burned area should be used as an
ex-ante feature. Instead, burned-area statistics define a less faithful
exploratory proxy label and provide sensitivity checks against the
physically grounded MODIS/FIRMS label described below.

**Label framing.** `L1` is exploratory, less faithful, and area-based:
it is a proxy derived from burned-area thresholds. It is useful for
sensitivity analysis and for comparing model behavior against the
heavy-tailed CONAF outcome distribution. `L2` is the main, more
exhaustive, physically grounded label: it uses satellite-observed fire
radiative power converted into fire-line intensity and is the preferred
training target for the paper.

### III-D. ERA5-Land: Meteorological and Land-Surface Covariates

Each CONAF event is enriched with ERA5-Land covariates at the ignition
point and nearest available hourly timestamp. ERA5 provides a physically
consistent global reanalysis suitable for retrospective event
reconstruction \cite{era5}. The project uses variables selected for
fire-weather relevance and for consistency with recent wildfire
prediction work using machine learning and explainable AI
\cite{zakari2025spatio,liao2025tackling}. ERA5-Land is distributed at
approximately 9 km (0.1°) horizontal resolution with hourly temporal
sampling \cite{munozsabater2021}.

The temporal ERA5-Land variables include 2 m air temperature, 2 m
dew-point temperature, 10 m wind components, total precipitation,
downward surface solar radiation, soil temperature at four layers,
volumetric soil water at four layers, potential evaporation, total
evaporation, evaporation from vegetation transpiration, and leaf area
index for high and low vegetation. Static land-surface covariates
include soil type, land-sea mask, high- and low-vegetation cover, and
high- and low-vegetation type.

The enrichment is formulated as a nearest-neighbor extraction problem.
For event *i*, the location `(lat_i, lon_i)` and timestamp `t_i` are
matched to the nearest ERA5-Land grid cell and hour. Match quality is
retained separately through distance and temporal offset fields so that
downstream training can filter or audit events with poor coverage.
ERA5-Land covariates are extracted only over the study-area domain
defined in Section III-A (Maule, Biobío, Araucanía, O'Higgins). Within
that domain almost every event obtains a valid match: in the 2016-2017
subset, 8,650 of 8,664 events (99.8%) are matched (median horizontal
distance 4.0 km, median temporal offset 0.25 h). CONAF events recorded
elsewhere in the national inventory have no ERA5-Land covariates and are
flagged as `out_of_coverage` and excluded from modeling.

#### ERA5-Land Temporal (Hourly) Variables

Table III lists the hourly ERA5-Land variables extracted for every
event. All of them are candidate ex-ante predictors. Soil-related fields
are resolved at the four standard ERA5-Land layers: L1 (0-7 cm), L2
(7-28 cm), L3 (28-100 cm), and L4 (100-289 cm). Precipitation,
radiation, and evaporation fields are reported as hourly accumulations
in their native units.

| \# | Variable | GRIB short name | CDS long name | Physical dimension | Training role |
|----|----|----|----|----|----|
| 1 | `t2m` | 2t | 2m_temperature | K | feature (ex-ante) |
| 2 | `d2m` | 2d | 2m_dewpoint_temperature | K | feature (ex-ante) |
| 3 | `u10` | 10u | 10m_u_component_of_wind | m s⁻¹ | feature (ex-ante) |
| 4 | `v10` | 10v | 10m_v_component_of_wind | m s⁻¹ | feature (ex-ante) |
| 5 | `tp` | tp | total_precipitation | m (hourly accumulation) | feature (ex-ante) |
| 6 | `ssrd` | ssrd | surface_solar_radiation_downwards | J m⁻² (hourly accumulation) | feature (ex-ante) |
| 7-10 | `stl1`-`stl4` | stl1-stl4 | soil_temperature_level_1..4 | K | feature (ex-ante) |
| 11-14 | `swvl1`-`swvl4` | swvl1-swvl4 | volumetric_soil_water_layer_1..4 | m³ m⁻³ | feature (ex-ante) |
| 15 | `pev` | pev | potential_evaporation | m (water equivalent) | feature (ex-ante) |
| 16 | `e` | e | total_evaporation | m (water equivalent) | feature (ex-ante) |
| 17 | `evavt` | evavt | evaporation_from_vegetation_transpiration | m (water equivalent) | feature (ex-ante) |
| 18 | `lai_hv` | lai_hv | leaf_area_index_high_vegetation | m² m⁻² (dimensionless) | feature (ex-ante) |
| 19 | `lai_lv` | lai_lv | leaf_area_index_low_vegetation | m² m⁻² (dimensionless) | feature (ex-ante) |

    \begin{table}[t]
    \centering
    \caption{ERA5-Land temporal (hourly) variables extracted at each event. Soil layers: L1 0--7~cm, L2 7--28~cm, L3 28--100~cm, L4 100--289~cm.}
    \label{tab:era5-temporal}
    \small
    \begin{tabular}{cllll}
    \toprule
    \# & Variable & GRIB & CDS long name & Dimension \\
    \midrule
    1     & \texttt{t2m}            & 2t        & \texttt{2m\_temperature}                          & K \\
    2     & \texttt{d2m}            & 2d        & \texttt{2m\_dewpoint\_temperature}                & K \\
    3     & \texttt{u10}            & 10u       & \texttt{10m\_u\_component\_of\_wind}              & m\,s$^{-1}$ \\
    4     & \texttt{v10}            & 10v       & \texttt{10m\_v\_component\_of\_wind}              & m\,s$^{-1}$ \\
    5     & \texttt{tp}             & tp        & \texttt{total\_precipitation}                     & m (hourly accum.) \\
    6     & \texttt{ssrd}           & ssrd      & \texttt{surface\_solar\_radiation\_downwards}     & J\,m$^{-2}$ (hourly accum.) \\
    7--10 & \texttt{stl1}--\texttt{stl4}   & stl1--stl4   & \texttt{soil\_temperature\_level\_1..4}     & K \\
    11--14& \texttt{swvl1}--\texttt{swvl4} & swvl1--swvl4 & \texttt{volumetric\_soil\_water\_layer\_1..4} & m$^3$\,m$^{-3}$ \\
    15    & \texttt{pev}            & pev       & \texttt{potential\_evaporation}                   & m (water eq.) \\
    16    & \texttt{e}              & e         & \texttt{total\_evaporation}                       & m (water eq.) \\
    17    & \texttt{evavt}          & evavt     & \texttt{evaporation\_from\_vegetation\_transpiration} & m (water eq.) \\
    18    & \texttt{lai\_hv}        & lai\_hv   & \texttt{leaf\_area\_index\_high\_vegetation}      & m$^2$\,m$^{-2}$ \\
    19    & \texttt{lai\_lv}        & lai\_lv   & \texttt{leaf\_area\_index\_low\_vegetation}       & m$^2$\,m$^{-2}$ \\
    \bottomrule
    \end{tabular}
    \end{table}

#### ERA5-Land Static (Invariant) Variables

Table IV lists the time-invariant land-surface variables, joined to each
event by spatial nearest neighbor only. Soil type and vegetation type
are integer-coded categorical fields; their full code-to-class mappings
are given in Section III-D.1. Land-sea mask and vegetation cover are
continuous fractions in `[0, 1]`.

| \# | Variable | GRIB short name | CDS long name | Possible values / dimension | Training role |
|----|----|----|----|----|----|
| 1 | `slt` | slt | soil_type | integer code 1-7 (Table V) | feature (ex-ante) |
| 2 | `lsm` | lsm | land_sea_mask | fraction 0-1 (dimensionless) | feature (ex-ante) |
| 3 | `cvh` | cvh | high_vegetation_cover | fraction 0-1 (dimensionless) | feature (ex-ante) |
| 4 | `cvl` | cvl | low_vegetation_cover | fraction 0-1 (dimensionless) | feature (ex-ante) |
| 5 | `tvh` | tvh | type_of_high_vegetation | integer code, high subset (Table VI) | feature (ex-ante) |
| 6 | `tvl` | tvl | type_of_low_vegetation | integer code, low subset (Table VI) | feature (ex-ante) |

    \begin{table}[t]
    \centering
    \caption{ERA5-Land static (invariant) land-surface variables, joined by spatial nearest neighbor.}
    \label{tab:era5-invariant}
    \small
    \begin{tabular}{cllll}
    \toprule
    \# & Variable & GRIB & CDS long name & Possible values / dimension \\
    \midrule
    1 & \texttt{slt} & slt & \texttt{soil\_type}                 & integer code 1--7 (Table~\ref{tab:soil-type}) \\
    2 & \texttt{lsm} & lsm & \texttt{land\_sea\_mask}            & fraction 0--1 (dimensionless) \\
    3 & \texttt{cvh} & cvh & \texttt{high\_vegetation\_cover}    & fraction 0--1 (dimensionless) \\
    4 & \texttt{cvl} & cvl & \texttt{low\_vegetation\_cover}     & fraction 0--1 (dimensionless) \\
    5 & \texttt{tvh} & tvh & \texttt{type\_of\_high\_vegetation} & integer code (Table~\ref{tab:veg-type}) \\
    6 & \texttt{tvl} & tvl & \texttt{type\_of\_low\_vegetation}  & integer code (Table~\ref{tab:veg-type}) \\
    \bottomrule
    \end{tabular}
    \end{table}

### III-D.1. Static Land-Surface Classification Codes

The soil-type and vegetation-type fields are categorical integer codes
inherited from the ECMWF land-surface scheme (H-TESSEL) and are stored
without remapping in the pipeline. Tables IV and V give the standard
code-to-class definitions used to interpret `slt`, `tvl`, and `tvh`
\cite{munozsabater2021,balsamo2009,vandenhurk2000,dickinson1993}. In the
Chilean subset the observed ranges are `slt` in 0-4, `tvh` in 0-19, and
`tvl` in 0-17, where the code 0 denotes ocean or undefined surface.

#### Soil Type (slt)

| Code | Texture class     |
|------|-------------------|
| 0    | Ocean / undefined |
| 1    | Coarse            |
| 2    | Medium            |
| 3    | Medium fine       |
| 4    | Fine              |
| 5    | Very fine         |
| 6    | Organic           |
| 7    | Tropical organic  |

    \begin{table}[t]
    \centering
    \caption{ECMWF H-TESSEL soil-texture classes for the \texttt{slt} field \citep{balsamo2009,munozsabater2021}.}
    \label{tab:soil-type}
    \begin{tabular}{cl}
    \toprule
    Code & Texture class \\
    \midrule
    0 & Ocean / undefined \\
    1 & Coarse \\
    2 & Medium \\
    3 & Medium fine \\
    4 & Fine \\
    5 & Very fine \\
    6 & Organic \\
    7 & Tropical organic \\
    \bottomrule
    \end{tabular}
    \end{table}

#### Vegetation Type (tvl, tvh)

| Code | Vegetation type            | Stratum |
|------|----------------------------|---------|
| 1    | Crops, mixed farming       | low     |
| 2    | Short grass                | low     |
| 3    | Evergreen needleleaf trees | high    |
| 4    | Deciduous needleleaf trees | high    |
| 5    | Deciduous broadleaf trees  | high    |
| 6    | Evergreen broadleaf trees  | high    |
| 7    | Tall grass                 | low     |
| 8    | Desert                     | low     |
| 9    | Tundra                     | low     |
| 10   | Irrigated crops            | low     |
| 11   | Semidesert                 | low     |
| 12   | Ice caps and glaciers      | \-      |
| 13   | Bogs and marshes           | low     |
| 14   | Inland water               | \-      |
| 15   | Ocean                      | \-      |
| 16   | Evergreen shrubs           | low     |
| 17   | Deciduous shrubs           | low     |
| 18   | Mixed forest / woodland    | high    |
| 19   | Interrupted forest         | high    |
| 20   | Water and land mixtures    | low     |

    \begin{table}[t]
    \centering
    \caption{ECMWF TESSEL/BATS vegetation types for the \texttt{tvl} (low) and \texttt{tvh} (high) fields \citep{vandenhurk2000,dickinson1993}.}
    \label{tab:veg-type}
    \small
    \begin{tabular}{clc}
    \toprule
    Code & Vegetation type & Stratum \\
    \midrule
    1  & Crops, mixed farming        & low \\
    2  & Short grass                 & low \\
    3  & Evergreen needleleaf trees  & high \\
    4  & Deciduous needleleaf trees  & high \\
    5  & Deciduous broadleaf trees   & high \\
    6  & Evergreen broadleaf trees   & high \\
    7  & Tall grass                  & low \\
    8  & Desert                      & low \\
    9  & Tundra                      & low \\
    10 & Irrigated crops             & low \\
    11 & Semidesert                  & low \\
    12 & Ice caps and glaciers       & -- \\
    13 & Bogs and marshes            & low \\
    14 & Inland water                & -- \\
    15 & Ocean                       & -- \\
    16 & Evergreen shrubs            & low \\
    17 & Deciduous shrubs            & low \\
    18 & Mixed forest / woodland     & high \\
    19 & Interrupted forest          & high \\
    20 & Water and land mixtures     & low \\
    \bottomrule
    \end{tabular}
    \end{table}

### III-E. Derived Meteorological Variables

Several derived variables are computed from the raw ERA5-Land fields to
express meteorological mechanisms more directly. Air and soil
temperatures are converted from Kelvin to Celsius. Wind speed and wind
direction are derived from the horizontal wind components. Total
precipitation is converted from meters to millimeters. Relative humidity
and vapor-pressure deficit are derived from air temperature and
dew-point temperature using a Magnus-type saturation vapor pressure
relationship.

T_C = T_K - 273.15

WS = sqrt(u10^2 + v10^2)

RH = 100 \* e_s(T_d) / e_s(T)

VPD = e_s(T) - e_s(T_d)

These transformations preserve the ex-ante nature of the predictors
because all inputs are meteorological or land-surface conditions
available at or before ignition.

Saturation vapor pressure uses the Magnus-Tetens approximation
`e_s(T) = 6.112 * exp(17.625 * T_C / (T_C + 243.04))` in hPa, with
coefficients from Alduchov and Eskridge (1996), valid over the -40 to
+60 °C range \cite{alduchov1996}. Relative humidity and vapor-pressure
deficit are therefore reported in percent and hPa, respectively. Table
VII summarizes every derived field.

| Output column | Formula | Inputs | Output dimension |
|----|----|----|----|
| `t2m_celsius` | T_K − 273.15 | `t2m` | °C |
| `d2m_celsius` | T_K − 273.15 | `d2m` | °C |
| `stl1_celsius`-`stl4_celsius` | T_K − 273.15 | `stl1`-`stl4` | °C |
| `relative_humidity` | 100 · e_s(T_d) / e_s(T) | `t2m`, `d2m` | % (clipped 0-100) |
| `vpd_hpa` | e_s(T) − e_s(T_d) | `t2m`, `d2m` | hPa (clipped ≥ 0) |
| `wind_speed` | √(u10² + v10²) | `u10`, `v10` | m s⁻¹ |
| `wind_direction` | (atan2(−u, −v) · 180/π + 360) mod 360 | `u10`, `v10` | degrees (0=N, 90=E) |
| `tp_mm` | tp · 1000 | `tp` | mm |

    \begin{table}[t]
    \centering
    \caption{Derived meteorological variables computed from raw ERA5-Land fields. $e_s$ is the Magnus saturation vapor pressure (hPa).}
    \label{tab:derived}
    \small
    \begin{tabular}{llll}
    \toprule
    Output column & Formula & Inputs & Output dimension \\
    \midrule
    \texttt{t2m\_celsius}   & $T_K - 273.15$                       & \texttt{t2m}          & $^\circ$C \\
    \texttt{d2m\_celsius}   & $T_K - 273.15$                       & \texttt{d2m}          & $^\circ$C \\
    \texttt{stl1\_celsius}--\texttt{stl4\_celsius} & $T_K - 273.15$       & \texttt{stl1}--\texttt{stl4} & $^\circ$C \\
    \texttt{relative\_humidity} & $100\,e_s(T_d)/e_s(T)$           & \texttt{t2m}, \texttt{d2m} & \% (clip 0--100) \\
    \texttt{vpd\_hpa}       & $e_s(T) - e_s(T_d)$                  & \texttt{t2m}, \texttt{d2m} & hPa (clip $\geq 0$) \\
    \texttt{wind\_speed}    & $\sqrt{u_{10}^2 + v_{10}^2}$         & \texttt{u10}, \texttt{v10} & m\,s$^{-1}$ \\
    \texttt{wind\_direction}& $(\operatorname{atan2}(-u,-v)\tfrac{180}{\pi} + 360)\bmod 360$ & \texttt{u10}, \texttt{v10} & deg (0=N, 90=E) \\
    \texttt{tp\_mm}         & $tp \cdot 1000$                      & \texttt{tp}           & mm \\
    \bottomrule
    \end{tabular}
    \end{table}

### III-F. MODIS/FIRMS and Label Construction

The main target follows the Extreme Wildfire Event (EWE) paradigm
proposed by Tedim et al. \cite{tedim2018EWE}. In this framing, an
extreme fire is not defined only by its final burned area. It is a
behavioral and operational process characterized by high fire intensity,
rapid spread, spotting potential, pyroconvective behavior, and the
possibility of exceeding suppression capacity. This distinction is
central for the present study: a fire can become operationally extreme
before its final scar is known.

The project operationalizes this concept using MODIS active-fire
detections exposed through NASA FIRMS \cite{nasaFirmsMODIS}. MODIS
provides thermal anomaly detections from Terra and Aqua and reports Fire
Radiative Power (FRP), a satellite-derived measure of radiative energy
release \cite{kaufman1998MODIS,giglio2016MODIS}. FRP is then converted
into an estimated Fire-Line Intensity (FLI) following the fire radiative
energy literature \cite{wooster2003FRE}. In the implemented peak-local
interpretation, the highest matched MODIS FRP pixel is divided by a
one-kilometer MODIS pixel-front length and by a radiant fraction.

FLI_hat_i \[kW/m\] = (1000 \* FRP_max_i \[MW\] / eta_r) / L_p \[m\]

Using the current implementation, `eta_r = 0.17` and `L_p = 1000 m`. The
binary target is defined as:

label_l2_i = 1\[FLI_hat_i \>= 10000 kW/m and N_MODIS_i \> 0 and A_i \>=
50 ha\]

The 10,000 kW/m threshold corresponds to the category 5+ operational EWE
boundary discussed by Tedim et al. \cite{tedim2018EWE}. The MODIS
detection guard requires at least one valid active-fire match. The
additional minimum-area guard is used only to reduce false positives
caused by spatial matching to an adjacent large fire; it does not turn
final burned area into a model feature.

**Methodological note.** The FIRE-RES/CONAF field-annotation context
should be cited here once the correct project source is confirmed.
TODO_VERIFY_FIRE_RES_CITATION. The intended role of that reference is to
motivate future operational annotation of ROS, spotting distance, flame
length, plume behavior, and suppression capacity, not to replace the
MODIS/FIRMS target in the current retrospective dataset.

The MODIS/FIRMS fields produced by the matching step are retained in the
enriched table only for target construction and auditability. Table VIII
lists them together with the matching and conversion parameters. All of
these columns are excluded from the predictor matrix because they encode
post-ignition satellite information.

| Column | Dimension | Description | Exclusion reason |
|----|----|----|----|
| `modis_n_matches` | count | Number of MODIS detections within the (5 km, ±24 h) window. | ex-post (satellite) |
| `modis_frp_max_mw` | MW | Maximum Fire Radiative Power among matched detections. | ex-post |
| `fli_estimado_kw_m` | kW m⁻¹ | Estimated Fire-Line Intensity from FRP (Wooster 2003). | ex-post / target-derived |
| `label_l2` | binary | EWE target: 1 if FRP ≥ 1700 MW and burned area ≥ 50 ha. | target |

Matching and conversion constants: `MATCH_RADIUS_KM = 5.0`,
`MATCH_TIME_HOURS = 24.0`, `RADIANT_FRACTION = 0.17`,
`MODIS_PIXEL_LENGTH_M = 1000`, `FLI_EWE_THRESHOLD_KW_M = 10000`,
`MIN_AREA_HA_FOR_L2 = 50`. The FRP threshold of 1700 MW is the
FRP-domain equivalent of the 10,000 kW/m FLI boundary under the
peak-local conversion (1000 · 1700 / 0.17 / 1000 = 10000).

    \begin{table}[t]
    \centering
    \caption{MODIS/FIRMS audit columns retained for target construction and excluded from the predictor matrix.}
    \label{tab:modis-audit}
    \small
    \begin{tabular}{p{0.24\linewidth}llp{0.22\linewidth}}
    \toprule
    Column & Dimension & Description & Exclusion reason \\
    \midrule
    \texttt{modis\_n\_matches}   & count      & MODIS detections within (5\,km, $\pm$24\,h). & ex-post (satellite) \\
    \texttt{modis\_frp\_max\_mw} & MW         & Maximum FRP among matched detections.        & ex-post \\
    \texttt{fli\_estimado\_kw\_m}& kW\,m$^{-1}$ & Estimated FLI from FRP (Wooster 2003).     & ex-post / target-derived \\
    \texttt{label\_l2}           & binary     & EWE target: FRP $\geq$ 1700\,MW and area $\geq$ 50\,ha. & target \\
    \bottomrule
    \end{tabular}
    \end{table}

### III-G. Enrichment Pipeline

The data construction pipeline starts from the cleaned CONAF event table
and produces one enriched row per event. First, local ignition date and
ignition hour are combined and converted to UTC. Second, the UTC
timestamp and coordinates are used to extract ERA5-Land temporal and
invariant variables. Third, derived meteorological variables are
computed from the raw ERA5-Land fields. Fourth, FIRMS active-fire
detections are queried over padded event days and matched to CONAF
events within a local space-time window. Finally, the L2 label is
assigned from the matched FRP-to-FLI conversion.

This design keeps feature construction and target construction separate.
ERA5-Land and static land-surface variables form the candidate predictor
set. MODIS/FIRMS fields are retained only for target auditability and
are excluded from training features.

### III-H. Training Dataset

The supervised learning task is defined after ignition: given a reported
fire event, estimate whether it will reach the L2 extreme-fire
threshold. This is aligned with fire triage and resource-allocation
settings in which only information available at ignition should be used
\cite{coffield2019FireSize}. The retained predictors therefore include
location, ignition time features, ERA5-Land meteorological variables,
derived meteorological variables, and static land-surface descriptors.

The following fields are explicitly excluded from the training feature
matrix: MODIS detections, FRP, FLI, and all MODIS/FIRMS-derived audit
columns; `superficie_quemada_*`, `duracion_minutos`, `alerta`,
`escenario`, and other post-ignition outcomes; and any target columns
including `label_l1`, `label_l2`, or their intermediate computations.
The model is trained only on ex-ante predictors available at or before
ignition.

On the 2016-2017 study-region subset (8,664 events, of which 8,650 have
a valid ERA5-Land match, 99.8%), MODIS/FIRMS detections matched 1,234
events (14.2%), and the L2 extreme-fire criterion is satisfied by only 5
events (0.06%); all five national L2 positives fall within the study
regions. This severe class imbalance is consistent with the EWE
definition, under which operationally extreme fires are rare, and it
motivates both extending the training window to the 2012-2018 seasons
and adopting imbalance-aware training and evaluation. Class counts for
the full window remain `PENDING_VERIFICATION` until the corresponding
enriched parquet is produced and audited.

The enriched parquet produced by the pipeline contains every column
described in this section, including ex-post outcomes and
target-construction fields. Feature selection is therefore applied at
training time: only the ex-ante columns are kept, and the remaining
columns are dropped according to the role classification in Table X. The
ignition timestamp is itself not used as a raw feature; instead the
cyclical-free ordinal fields in Table IX are derived from it.

#### Ignition-Time Temporal Features

| Column        | Derivation            | Range | Training role     |
|---------------|-----------------------|-------|-------------------|
| `month`       | timestamp.month       | 1-12  | feature (ex-ante) |
| `hour`        | timestamp.hour        | 0-23  | feature (ex-ante) |
| `day_of_year` | timestamp.day_of_year | 1-366 | feature (ex-ante) |

    \begin{table}[t]
    \centering
    \caption{Ignition-time temporal features derived from the event timestamp (ordinal, no cyclical encoding).}
    \label{tab:temporal-features}
    \begin{tabular}{lll}
    \toprule
    Column & Derivation & Range \\
    \midrule
    \texttt{month}       & timestamp.month       & 1--12 \\
    \texttt{hour}        & timestamp.hour        & 0--23 \\
    \texttt{day\_of\_year} & timestamp.day\_of\_year & 1--366 \\
    \bottomrule
    \end{tabular}
    \end{table}

#### Feature Matrix Role Classification

| Role | Columns | Rationale |
|----|----|----|
| Included (ex-ante feature) | All ERA5-Land temporal variables (Table III); all static variables (Table IV); all derived variables (Table VII); `month`, `hour`, `day_of_year` (Table IX); `latitud`, `longitud`; `region`, `provincia`, `comuna` | Available at or before ignition. |
| Excluded - target | `superficie_quemada_total_ha`, `label_l1`, `label_l2`, `is_megafire` | Prediction targets and their precursors. |
| Excluded - ex-post / leakage | `superficie_quemada_*` (per fuel class), `duracion_minutos`, `alerta`, `escenario`, `causa` | Known only after the event develops or after investigation. |
| Excluded - audit | `modis_n_matches`, `modis_frp_max_mw`, `fli_estimado_kw_m`, `era5_dist_km`, `era5_dt_hours`, `era5_match_quality` | Retained for traceability of the target and match quality, not predictive use. |
| Excluded - metadata / id | `nombre`, `temporada`, `datum`, `geometry`, raw timestamp columns | Identifiers and non-predictive metadata. |

    \begin{table*}[t]
    \centering
    \caption{Role classification applied to the enriched parquet to build the ex-ante predictor matrix.}
    \label{tab:training-roles}
    \small
    \begin{tabular}{p{0.20\linewidth}p{0.50\linewidth}p{0.22\linewidth}}
    \toprule
    Role & Columns & Rationale \\
    \midrule
    Included (ex-ante feature) & All ERA5-Land temporal (Table~\ref{tab:era5-temporal}) and static (Table~\ref{tab:era5-invariant}) variables; all derived variables (Table~\ref{tab:derived}); \texttt{month}, \texttt{hour}, \texttt{day\_of\_year} (Table~\ref{tab:temporal-features}); \texttt{latitud}, \texttt{longitud}; \texttt{region}, \texttt{provincia}, \texttt{comuna} & Available at or before ignition. \\
    Excluded --- target & \texttt{superficie\_quemada\_total\_ha}, \texttt{label\_l1}, \texttt{label\_l2}, \texttt{is\_megafire} & Prediction targets and precursors. \\
    Excluded --- ex-post / leakage & \texttt{superficie\_quemada\_*} (per fuel class), \texttt{duracion\_minutos}, \texttt{alerta}, \texttt{escenario}, \texttt{causa} & Known only after the event develops or after investigation. \\
    Excluded --- audit & \texttt{modis\_n\_matches}, \texttt{modis\_frp\_max\_mw}, \texttt{fli\_estimado\_kw\_m}, \texttt{era5\_dist\_km}, \texttt{era5\_dt\_hours}, \texttt{era5\_match\_quality} & Traceability of target and match quality, not predictive. \\
    Excluded --- metadata / id & \texttt{nombre}, \texttt{temporada}, \texttt{datum}, \texttt{geometry}, raw timestamp columns & Identifiers and non-predictive metadata. \\
    \bottomrule
    \end{tabular}
    \end{table*}

#### III-H.1. Final Feature Set and Selection Process (To Be Defined)

The final set of features fed to the model and the procedure that
produces it are still to be defined. This subsection is a placeholder to
be completed once the feature-selection experiments are run; the figures
and the retained feature list below are pending.

**TODO_TRAINING_FEATURE_SELECTION.** Pending items:

- Final feature list retained for the model, as a subset of the ex-ante
  predictors in Table X — to be defined.
- Selection process: leakage and ex-post column removal (Table X);
  coverage filtering (drop `out_of_coverage` events); encoding of
  categorical fields (`region`, `provincia`, `comuna`); and treatment of
  missing values — to be defined.
- Redundancy and importance-based pruning (e.g. correlation filtering,
  model-based feature importance) — to be defined.
- Temporal train/validation/test split over the 2012-2018 window and
  class-imbalance handling — to be defined.
- Final feature count and per-feature justification —
  `PENDING_VERIFICATION`.

### III-I. Limitations and Future Field Annotation

The proposed dataset inherits limitations from both administrative fire
records and remote sensing. CONAF provides high-value historical event
information, but several variables are ex-post operational summaries
rather than conditions known at ignition. ERA5-Land offers spatially and
temporally consistent meteorology, but it cannot resolve all local wind,
fuel, and topographic effects that influence rapid spread. MODIS/FIRMS
provides historical active-fire detections and FRP, but detections
depend on overpass time, cloud and smoke conditions, pixel geometry, and
sensor saturation or omission in small fires.

For Chile, prior work emphasizes that fire activity is shaped by
climate, land cover, plantation structure, human ignition patterns, and
regional bioclimatic gradients
\cite{mcwethy2018Landscape,ubeda2016Chile}. The present dataset captures
only part of this causal structure. A stronger future dataset would
combine the current retrospective enrichment with field annotation of
rate of spread, flame length, spotting distance, crown-fire activity,
convective plume behavior, suppression difficulty, and resource
saturation. Those variables would allow the EWE concept to be tested
more directly rather than inferred from MODIS FRP alone.

### III-J. BibTeX References

    @misc{conaf_dataset,
      author       = {{Corporacion Nacional Forestal}},
      title        = {Registro historico de incendios forestales},
      year         = {2024},
      howpublished = {Datos para Resiliencia, V1},
      doi          = {10.71578/UXAUN5},
      url          = {https://datospararesiliencia.cl/dataset.xhtml?persistentId=doi:10.71578/UXAUN5}
    }

    @article{era5,
      author  = {Hersbach, Hans and Bell, Bill and Berrisford, Paul and Hirahara, Shoji and Horanyi, Andras and Munoz-Sabater, Joaquin and Nicolas, Julien and Peubey, Carole and Radu, Raluca and Schepers, Dinand and Simmons, Adrian and Soci, Cornel and Abdalla, Saleh and Abellan, Xavier and Balsamo, Gianpaolo and Bechtold, Peter and Biavati, Gionata and Bidlot, Jean and Bonavita, Massimo and De Chiara, Giovanna and Dahlgren, Per and Dee, Dick and Diamantakis, Michail and Dragani, Rossana and Flemming, Johannes and Forbes, Richard and Fuentes, Manuel and Geer, Alan and Haimberger, Leopold and Healy, Sean and Hogan, Robin J. and Holm, Elin and Janiskova, Marta and Keeley, Sarah and Laloyaux, Patrick and Lopez, Philippe and Lupu, Cristina and Radnoti, Gabor and de Rosnay, Patricia and Rozum, Iryna and Vamborg, Freja and Villaume, Sebastien and Thepaut, Jean-Noel},
      title   = {The {ERA5} global reanalysis},
      journal = {Quarterly Journal of the Royal Meteorological Society},
      year    = {2020},
      volume  = {146},
      number  = {730},
      pages   = {1999--2049},
      doi     = {10.1002/qj.3803}
    }

    @article{zakari2025spatio,
      author    = {Zakari, Rufai Yusuf and Malik, Owais Ahmed and Ong, Wee-Hong},
      title     = {Spatio-temporal wildfire forecasting in Australia using deep learning and explainable {AI}},
      journal   = {Modeling Earth Systems and Environment},
      year      = {2025},
      volume    = {11},
      number    = {6},
      pages     = {425},
      publisher = {Springer},
      doi       = {10.1007/s40808-025-02621-7}
    }

    @article{liao2025tackling,
      author    = {Liao, Bin and Zhou, Tao and Liu, Yanping and Li, Min and Zhang, Tao},
      title     = {Tackling the wildfire prediction challenge: an explainable artificial intelligence ({XAI}) model combining extreme gradient boosting ({XGBoost}) with {SHapley} additive ex{Planations} ({SHAP}) for enhanced interpretability and accuracy},
      journal   = {Forests},
      year      = {2025},
      volume    = {16},
      number    = {4},
      pages     = {689},
      publisher = {MDPI},
      doi       = {10.3390/f16040689}
    }

    @article{malamud1998,
      author  = {Malamud, B. D. and Morein, G. and Turcotte, D. L.},
      title   = {Forest fires: An example of self-organized critical behavior},
      journal = {Science},
      year    = {1998},
      volume  = {281},
      number  = {5384},
      pages   = {1840--1842},
      doi     = {10.1126/science.281.5384.1840}
    }

    @article{tedim2018EWE,
      author  = {Tedim, Fantina and Leone, Vittorio and Amraoui, Malik and Bouillon, Christophe and Coughlan, Michael R. and Delogu, Giuseppe M. and Fernandes, Paulo M. and Ferreira, Carmen and McCaffrey, Sarah and McGee, Tara K. and Parente, Joana and Paton, Douglas and Pereira, Mario G. and Ribeiro, Luis M. and Viegas, Domingos X. and Xanthopoulos, Gavriil},
      title   = {Defining extreme wildfire events: Difficulties, challenges, and impacts},
      journal = {Fire},
      year    = {2018},
      volume  = {1},
      number  = {1},
      pages   = {9},
      doi     = {10.3390/fire1010009}
    }

    @article{wooster2003FRE,
      author  = {Wooster, Martin J. and Zhukov, Boris and Oertel, Dieter},
      title   = {Fire radiative energy for quantitative study of biomass burning: Derivation from the {BIRD} experimental satellite and comparison to {MODIS} fire products},
      journal = {Remote Sensing of Environment},
      year    = {2003},
      volume  = {86},
      number  = {1},
      pages   = {83--107},
      doi     = {10.1016/S0034-4257(03)00070-1}
    }

    @article{giglio2016MODIS,
      author  = {Giglio, Louis and Schroeder, Wilfrid and Justice, Christopher O.},
      title   = {The Collection 6 {MODIS} active fire detection algorithm and fire products},
      journal = {Remote Sensing of Environment},
      year    = {2016},
      volume  = {178},
      pages   = {31--41},
      doi     = {10.1016/j.rse.2016.02.054}
    }

    @article{kaufman1998MODIS,
      author  = {Kaufman, Yoram J. and Justice, Christopher O. and Flynn, Luke P. and Kendall, Jackie D. and Prins, Elaine M. and Giglio, Louis and Ward, Darold E. and Menzel, W. Paul and Setzer, Alberto W.},
      title   = {Potential global fire monitoring from {EOS}-{MODIS}},
      journal = {Journal of Geophysical Research: Atmospheres},
      year    = {1998},
      volume  = {103},
      number  = {D24},
      pages   = {32215--32238},
      doi     = {10.1029/98JD01644}
    }

    @article{coffield2019FireSize,
      author  = {Coffield, Shane R. and Graff, Casey A. and Chen, Yang and Smyth, Padhraic and Foufoula-Georgiou, Efi and Randerson, James T.},
      title   = {Machine learning to predict final fire size at the time of ignition},
      journal = {International Journal of Wildland Fire},
      year    = {2019},
      volume  = {28},
      number  = {11},
      pages   = {861--873},
      doi     = {10.1071/WF19023}
    }

    @article{mcwethy2018Landscape,
      author  = {McWethy, David B. and Pauchard, Anibal and Garcia, Rafael A. and Holz, Andres and Gonzalez, Mauro E. and Veblen, Thomas T. and Stahl, Julian and Currey, Bryce},
      title   = {Landscape drivers of recent fire activity (2001-2017) in south-central {Chile}},
      journal = {PLOS ONE},
      year    = {2018},
      volume  = {13},
      number  = {8},
      pages   = {e0201195},
      doi     = {10.1371/journal.pone.0201195}
    }

    @article{ubeda2016Chile,
      author  = {Ubeda, Xavier and Sarricolea, Pablo},
      title   = {Wildfires in {Chile}: A review},
      journal = {Global and Planetary Change},
      year    = {2016},
      volume  = {146},
      pages   = {152--161},
      doi     = {10.1016/j.gloplacha.2016.10.004}
    }

    @misc{nasaFirmsMODIS,
      author       = {{NASA LANCE FIRMS}},
      title        = {{MODIS} active fire data via the Fire Information for Resource Management System ({FIRMS})},
      year         = {2026},
      howpublished = {NASA Earthdata},
      url          = {https://firms.modaps.eosdis.nasa.gov/api/},
      note         = {Accessed 2026-05-25}
    }

    @article{munozsabater2021,
      author  = {Munoz-Sabater, Joaquin and Dutra, Emanuel and Agusti-Panareda, Anna and Albergel, Clement and Arduini, Gabriele and Balsamo, Gianpaolo and Boussetta, Souhail and Choulga, Margarita and Harrigan, Shaun and Hersbach, Hans and Martens, Brecht and Miralles, Diego G. and Piles, Maria and Rodriguez-Fernandez, Nemesio J. and Zsoter, Ervin and Buontempo, Carlo and Thepaut, Jean-Noel},
      title   = {{ERA5-Land}: a state-of-the-art global reanalysis dataset for land applications},
      journal = {Earth System Science Data},
      year    = {2021},
      volume  = {13},
      number  = {9},
      pages   = {4349--4383},
      doi     = {10.5194/essd-13-4349-2021}
    }

    @article{balsamo2009,
      author  = {Balsamo, Gianpaolo and Beljaars, Anton and Scipal, Klaus and Viterbo, Pedro and van den Hurk, Bart and Hirschi, Martin and Betts, Alan K.},
      title   = {A revised hydrology for the {ECMWF} model: Verification from field site to terrestrial water storage and impact in the {Integrated Forecast System}},
      journal = {Journal of Hydrometeorology},
      year    = {2009},
      volume  = {10},
      number  = {3},
      pages   = {623--643},
      doi     = {10.1175/2008JHM1068.1}
    }

    @techreport{vandenhurk2000,
      author      = {van den Hurk, Bart J. J. M. and Viterbo, Pedro and Beljaars, Anton C. M. and Betts, Alan K.},
      title       = {Offline validation of the {ERA40} surface scheme},
      institution = {European Centre for Medium-Range Weather Forecasts},
      year        = {2000},
      number      = {295},
      type        = {ECMWF Technical Memorandum}
    }

    @techreport{dickinson1993,
      author      = {Dickinson, Robert E. and Henderson-Sellers, Ann and Kennedy, Paul J.},
      title       = {Biosphere-Atmosphere Transfer Scheme ({BATS}) Version 1e as coupled to the {NCAR} Community Climate Model},
      institution = {National Center for Atmospheric Research},
      year        = {1993},
      number      = {NCAR/TN-387+STR},
      type        = {NCAR Technical Note},
      doi         = {10.5065/D67W6959}
    }

    @article{alduchov1996,
      author  = {Alduchov, Oleg A. and Eskridge, Robert E.},
      title   = {Improved {Magnus} form approximation of saturation vapor pressure},
      journal = {Journal of Applied Meteorology},
      year    = {1996},
      volume  = {35},
      number  = {4},
      pages   = {601--609},
      doi     = {10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2}
    }
