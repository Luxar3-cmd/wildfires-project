# Insights

Bitácora de detalles finos del proyecto que no son obvios al leer el código y vale la pena tener a mano:
matices de cálculo, supuestos, gotchas y decisiones de diseño. Una entrada por hallazgo.

---

## Etiqueta L2: cálculo MODIS FRP→FLI

**Archivo:** `src/modis.py`

Cadena de cálculo (4 etapas):

1. **Descarga FRP** — `download_firms_for_conaf()`: CSVs del producto MODIS Thermal Anomalies/Fire
   L2 5-Min Swath 1 km (MOD14/MYD14, Collection 6.1) vía NASA FIRMS Area API, ±1 día por evento.
2. **Matching CONAF↔MODIS** — `match_modis_to_conaf()` (`modis.py:307`): por cada evento CONAF agrega
   el FRP de las detecciones dentro de `|Δt| ≤ 24 h` (`MATCH_TIME_HOURS`) y haversine `≤ 5 km`
   (`MATCH_RADIUS_KM`). Retorna `modis_frp_max_mw` (píxel pico), `modis_frp_sum_mw`, `modis_n_matches`.
3. **FRP→FLI** — `frp_to_fli()` (`modis.py:388`): `FLI[kW/m] = (FRP[MW]·1000 / η_r) / front_length[m]`.
   Se aplica sobre el píxel más caliente (`frp_max`) con su propia longitud ("peak local").
4. **Etiqueta L2** — `label_l2()` (`modis.py:420`).

Supuestos (constantes en `modis.py:42-58`):

| Constante | Valor | Naturaleza | Fuente |
|---|---|---|---|
| `RADIANT_FRACTION` (η_r) | 0.17 | **Supuesto del proyecto** (rango físico 0.10–0.20) | NO de Wooster |
| `MODIS_PIXEL_LENGTH_M` | 1000 m (1 km nadir) | **Supuesto del proyecto** (longitud del frente = pixel) | NO de Wooster |
| `FLI_EWE_THRESHOLD_KW_M` | 10.000 kW/m | Umbral categoría EWE (5+) | **Tedim 2018** |
| `MIN_AREA_HA_FOR_L2` | 50 ha | Guardia de coherencia área–FLI | Supuesto |
| `MATCH_RADIUS_KM` | 5 km | Tolerancia espacial del matching | Supuesto |
| `MATCH_TIME_HOURS` | 24 h | Semiventana temporal del matching | Supuesto |

El **marco** (FLI radiativa = FRE / longitud del frente) viene de Wooster et al. (2003, 2004), pero los
dos números físicos (η_r y longitud de 1 km) son supuestos del proyecto, **no** tomados de Wooster. Él
advierte que estimar FLI bien exige resolver el frente completo (~370 m, sensor BIRD), imposible con MODIS
a 1 km. El único valor anclado a literatura es el umbral de 10.000 kW/m (Tedim 2018, definición EWE).

Limitación de fondo: la definición EWE completa exige *spread rate* y *spot distance* (solo in-situ); aquí
solo hay FRP, por eso L2 es una **aproximación física**, no una medición EWE.

---

## L2 = 1 exige FLI ≥ umbral **AND** superficie ≥ 50 ha

**Archivo:** `src/modis.py:459-463`

```python
fli_ok = out["fli_estimado_kw_m"] >= fli_threshold       # FLI ≥ 10.000 kW/m
if "superficie_quemada_total_ha" in out.columns:
    area_ok = ... >= min_area_ha                          # superficie ≥ 50 ha
    fli_ok = fli_ok & area_ok
out["label_l2"] = fli_ok.astype(int)
```

`label_l2 = 1 ⟺ FLI ≥ 10.000 kW/m AND superficie ≥ 50 ha`. Equivale a `frp_max ≥ 1700 MW`. Sin detección
satelital → `label_l2 = 0`.

Matices:
- **La guardia de 50 ha es condicional a que exista la columna** `superficie_quemada_total_ha`. Si no
  está, L2 queda definido solo por la FLI. En el dataset canónico la columna existe, así que el AND siempre
  aplica en la práctica.
- **No es un criterio de intensidad, es una guardia anti-falso-positivo.** El criterio EWE real es la FLI.
  Las 50 ha existen porque el matching usa radio de 5 km: sin la guardia, un fuego chico podría heredar el
  FRP de un megaincendio vecino dentro del radio y marcarse EWE espuriamente ("un EWE no ocupa <50 ha").
- **No confundir con el umbral de L1** (`≥ 1000 ha` por área). Son cosas distintas: L2 etiqueta por
  intensidad (FLI); las 50 ha son solo un piso de sanidad, no el criterio de etiquetado.
