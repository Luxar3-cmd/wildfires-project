# Resumen ejecutivo

Esta sesión llevó el proyecto de la entrega 2 (E2) a la entrega 3 (E3). En concreto:
**(1)** se restringió el modeling a las cuatro regiones de estudio (decisión de coherencia
código↔paper); **(2)** se recomputaron todos los resultados sobre el dataset canónico de
44 features (2012–2018); **(3)** se crearon dos análisis nuevos —sensibilidad del umbral EWE
y **utilidad operacional (triage)**—; **(4)** se reescribió el paper LaTeX completo (resultados
finales, discusión de fidelidad, conclusiones) y se compiló el PDF sin errores. Todo se hizo
sobre un dataset respaldado de forma inmutable, con un punto de rollback explícito.

El hallazgo central: para predecir si un incendio escalará, la etiqueta de **área (L1)** y la
de **intensidad (L2, EWE)** son **complementarias, no intercambiables** —y el modelo, además de
explicable, tiene **valor operacional real** como priorizador de recursos.

---

# Salvaguardas y rollback

Antes de tocar nada se respaldó el dataset canónico de forma intocable:

- **Backup read-only**: `dataset_backups/baseline_44feat_20260615_193800/`
  (`conaf_enriched_2012_2018.parquet` + `MANIFEST.txt`, permisos `444/555`).
- **Integridad**: SHA-256 `f6b0fd37f7b736243cb63d3fdeee75ac8a4f480008a5fdb9611e90eeab5cf836` (verificado).
- **Flag de código**: git tag anotado `dataset-baseline-44feat` → commit `f91282f`.

**Rollback** (si se requiere):

```bash
git checkout dataset-baseline-44feat        # vuelve al código baseline
cp dataset_backups/baseline_44feat_20260615_193800/conaf_enriched_2012_2018.parquet \
   data/processed/conaf_enriched_2012_2018.parquet
```

El parquet **no se modificó** en toda la sesión (los re-etiquetados y filtros son en memoria),
así que el backup sigue siendo el baseline válido.

---

# Cambios en el pipeline de modeling

## Filtro a las 4 regiones de estudio

Se detectó que el modeling entrenaba sobre todas las filas con cobertura ERA5 del *bounding box*
(32.158), que incluían un 5% de eventos de Los Lagos, Los Ríos y Metropolitana —contradiciendo
la afirmación del paper de restringirse a las 4 regiones. Se alineó el **código** al texto:

- Nueva constante `STUDY_REGIONS = ["Maule", "Biobío", "Araucanía", "O'Higgins"]` en
  `src/modeling_features.py` (fuente única).
- Filtro `df[df["region"].isin(STUDY_REGIONS)]` aplicado en el `load()` de `modeling/02`, `03`,
  `04` y `05`.

| | Antes (bbox) | Después (4 regiones) |
|---|---|---|
| Filas modelables | 32.158 | **30.511** |
| L1 positivos | 78 | **76** |
| L2 positivos | 11 | **11** |

Cobertura ERA5 en las 4 regiones (de 31.118 eventos): `good` 30.021, `land_snapped` 493,
`water` 543, `out_of_coverage` 61 → **30.514 con ERA5 válido (98,1%)**; MODIS detecta 4.545
eventos (14,6%).

## Scripts nuevos

- **`modeling/04_l2_threshold_sensitivity.py`** — re-deriva la etiqueta L2 en memoria sobre un
  grid del umbral FLI (6.000–16.000 kW/m) y de la *radiant fraction* η_r (0,10–0,20), y mide si
  la conclusión del proxy aguanta. No toca el parquet.
- **`modeling/05_operational_triage.py`** — evalúa el modelo como herramienta de priorización
  (recall, lift y falsas alarmas por presupuesto de inspección), en tres modos (L1→L1, L2→L2 y
  el proxy L1→L2).

---

# Resultados E3

Todo sobre probabilidad **out-of-fold** (sin fuga); para L2 la validación cruzada se repite 20×
para obtener intervalos de confianza al 95%.

## Rendimiento predictivo

| | L1 (área) | L2 (intensidad EWE) |
|---|---|---|
| Positivos / total | 76 / 30.511 | 11 / 30.511 |
| ROC-AUC | **0,914** [0,899–0,930] | **0,854** [0,703–0,916] |
| PR-AUC | 0,112 [0,096–0,137] | 0,094 [0,003–0,139] |
| PR-AUC base (prevalencia) | 0,0025 | 0,00036 |

El intervalo ancho de L2 es la firma honesta de 11 positivos; su piso (0,70) sigue sobre el azar.
En el punto de operación a 0,5, L1 da precision 0,12 y recall 0,20 (matriz de confusión abajo):
el producto útil es el **ranking**, no la alarma binaria.

![Matriz de confusión OOF del modelo L1 (umbral 0,5, n=30.511).](latex/images/cm_l1.png){width=55%}

## ¿El área (L1) sirve de proxy de la intensidad (L2)? — la pregunta de investigación

| Test (objetivo = L2) | Score L1 | Modelo propio L2 |
|---|---|---|
| ROC-AUC para L2 | **0,890** | 0,854 |
| PR-AUC para L2 | 0,096 | 0,094 |
| Recall@top-5% | 0,55 | 0,64 |
| Spearman(riesgo L1, L2) | 0,492 | acuerdo moderado |
| Overlap (L2 también son L1) | 6 / 11 | solapamiento directo |

El score de área rankea los EWE casi tan bien como el modelo propio de intensidad (incluso un
poco mejor, porque L2 está hambriento de positivos), pero **solo 6 de los 11 EWE son también
megaincendios por área**. Bajo **leave-one-positive-out** los 11 EWE caen ≥ percentil 68 aun sin
haber estado en el entrenamiento (no memoriza), y los **dos mejores son fuegos pequeños pero
intensos** de Biobío (313 y 185 ha) que un modelo de área no marcaría —los 5 eventos
pequeños-intensos son justo lo que L1 no puede ver.

![Distribución de riesgo OOF para L2 (conteo log). Los 11 positivos (rojo) se concentran en la cola; los dos pure-L2 quedan cerca del máximo.](latex/images/l2_risk_hist.png){width=70%}

**Conclusión:** L1 es un proxy **parcial pero no sustituto** de L2.

## Robustez a los supuestos físicos del label

La etiqueta L2 depende de dos supuestos no medidos (umbral 10.000 kW/m y η_r = 0,17). Variándolos
en todo su rango plausible, el **proxy AUC se mantiene en 0,86–0,88** y la discriminación propia
de L2 en 0,81–0,91: la conclusión no es un artefacto de la elección del umbral.

![Sensibilidad de L2 y del proxy L1→L2 a los supuestos de la conversión FRP→FLI.](latex/images/l2_threshold_sensitivity.png){width=95%}

## Drivers (Tree SHAP): área vs intensidad

Ambos modelos se guían por meteorología (no por ubicación administrativa) y comparten
estacionalidad y viento, pero **el énfasis difiere de forma físicamente coherente**:

- **L1 (área):** evaporación total (`e`), temperatura del suelo (`stl2`), posición geográfica
  (`latitud`, `longitud`), humedad profunda (`swvl4`) — *dónde y cuándo se propaga* un fuego.
- **L2 (intensidad):** sequedad atmosférica (rocío `d2m`, déficit `vpd_hpa`) y estructura de la
  vegetación (transpiración `evavt`, cobertura alta `cvh`, índice foliar `lai_lv`) — *cuán
  caliente arde* el frente.

Que las explicaciones diverjan con los mismos predictores es evidencia directa de que área e
intensidad son fenómenos distintos.

![Atribuciones globales Tree SHAP. Izquierda L1 (área), derecha L2 (intensidad).](latex/images/shap_beeswarm_l1.png){width=48%}
![](latex/images/shap_beeswarm_l2.png){width=48%}

## Fidelidad de las explicaciones (Quantus)

| Métrica | L1 | L2 |
|---|---|---|
| Faithfulness Correlation (Bhatt et al.) | **+0,42** | **+0,37** |
| Faithfulness Estimate (Alvarez-Melis & Jaakkola) | **−0,62** | **−0,57** |

Tree SHAP es exacto por construcción. La FC positiva lo corrobora; la FE negativa **no** indica
explicaciones malas, sino la limitación —documentada por Miró-Nicolau et al. (2025)— de que las
métricas de fidelidad fallan en modelos no lineales (responden de forma no monótona a perturbar
una variable). La FC, que agrega sobre subconjuntos, es la lectura confiable aquí.

---

# Utilidad operacional (triage)

Más allá de la discriminación, se midió el valor de **decisión**: si solo hay capacidad para
inspeccionar el top-k% de mayor riesgo, ¿cuántos incendios extremos se capturan, con qué *lift*
sobre el azar y a qué costo de falsas alarmas? Sobre probabilidad OOF (sin fuga).

**L1 → L1** (el modelo de área tría megaincendios por área, 76 positivos)

| Presupuesto | Capturados | Lift | Falsas alarmas/acierto |
|---|---|---|---|
| top 1% | 30/76 (39%) | 39,5× | 9 |
| top 5% | 51/76 (67%) | 13,4× | 29 |
| **top 10%** | **61/76 (80%)** | 8,0× | 49 |
| top 20% | 69/76 (91%) | 4,5× | 87 |

**L2 → L2** (el modelo de intensidad tría EWE, 11 positivos)

| Presupuesto | Capturados | Lift | Falsas alarmas/acierto |
|---|---|---|---|
| top 1% | 5/11 (45%) | 45,5× | 60 |
| top 5% | 7/11 (64%) | 12,7× | 217 |
| **top 10%** | **9/11 (82%)** | 8,2× | 338 |
| top 20% | 9/11 (82%) | 4,1× | 677 |

**L1 → L2** (el modelo de área tría intensidad — el proxy operacional, 11 positivos)

| Presupuesto | Capturados | Lift | Falsas alarmas/acierto |
|---|---|---|---|
| top 1% | 6/11 (55%) | 54,5× | 50 |
| top 5% | 6/11 (55%) | 10,9× | 253 |
| top 10% | 7/11 (64%) | 6,4× | 435 |
| top 20% | 8/11 (73%) | 3,6× | 762 |

![Curva de captura (cumulative gains) y lift a presupuesto bajo, en los tres modos.](latex/images/operational_triage.png){width=95%}

**Lectura.** El modelo es un **concentrador de riesgo fuerte**: con el 10% de mayor riesgo se
captura el 80% de los megaincendios y el 82% de los EWE (8× sobre el azar); a presupuesto del 1%
el lift llega a 40–55×. El cuello de botella es la **precisión**, no el recall: por la rareza
(prevalencia 0,03% en L2) hay un techo estructural de falsas alarmas —pero en triage de catástrofes
una falsa alarma es vigilar un incendio que resultó no extremo, costo bajo frente a perder uno.
El proxy L1→L2 confirma el hallazgo: a la cabeza el área captura más EWE que el modelo de intensidad
(6/11 vs 5/11), pero **se estanca en 6/11** —no encuentra los pequeños-intensos, que sí rescata el
modelo propio de L2 (9/11 al 10%).

---

# Conceptos (apéndice didáctico)

**Score de riesgo.** El modelo entrega una probabilidad (0–1), no un sí/no. Triar = ordenar por
ese score.

**Out-of-fold (OOF) / cross-validation.** Cada incendio recibe su score de un modelo entrenado
sin haberlo visto (validación cruzada en 5 bloques). Evita el optimismo de evaluar sobre datos de
entrenamiento. Se repite 20× y se promedia para estabilidad.

**`scale_pos_weight`.** Con 11 positivos contra 30.500 negativos, el modelo perezoso diría "no"
siempre. Este parámetro pondera los positivos ~2.800× más, forzando atención sobre la clase rara.

**TP / FP / FN.** De lo marcado como alto riesgo: TP = extremo bien marcado (acierto); FP = no era
extremo (falsa alarma); FN = extremo no marcado (se escapó).

**Recall** = TP / (TP + FN): de los extremos reales, qué fracción atrapé.
**Precision** = TP / (TP + FP): de lo que marqué, qué fracción era extremo.
Son opuestos: ampliar el presupuesto sube recall y baja precision.

**Lift** = recall / fracción priorizada = precision / prevalencia: cuántas veces mejor que el azar.
Es la métrica que muestra que el modelo ordena bien aunque la precisión absoluta sea baja.

**Curva de ganancia (cumulative gains).** % de población priorizada (X) vs % de extremos
capturados (Y). La diagonal es el azar; cuanto más se despega hacia arriba-izquierda, mejor.

**Prevalencia y techo de precisión.** La fracción de positivos (0,034% en L2) impone un **techo
matemático** a la precisión: en el top 1% (305 eventos) solo existen 11 EWE, así que la precisión
máxima posible es 11/305 = 0,036. La baja precisión la impone la rareza, no el modelo.

**ROC-AUC vs PR-AUC / lift.** El ROC-AUC trata por igual aciertos y falsas alarmas y se ve
optimista con clases desbalanceadas; las métricas operacionales (precision, lift, gains) sí
sienten el peso de las falsas alarmas y cuentan la historia completa.

*Para leer más:* Fawcett (2006, "An introduction to ROC analysis"); Saito & Rehmsmeier (2015,
PLoS ONE); Davis & Goadrich (2006, ICML); Provost & Fawcett (*Data Science for Business*);
Hastie, Tibshirani & Friedman (*The Elements of Statistical Learning*, cap. 7).

---

# Estado del paper LaTeX

El paper (`latex/`, **"From Spark to Catastrophe"**) se migró de E2 (preliminar, subset 2016-2017,
solo L1) a E3 (final):

| Archivo | Cambio |
|---|---|
| `data.tex` | Conteos 2012-2018 (31.118/30.514/30.511, L1=76, L2=11); fuera notas de entregas previas |
| `model_construction.tex` | Setup final, ambos labels, CV repetida, software real (Py 3.12 / shap 0.52 / sklearn 1.9) |
| `results.tex` | **Reescrita**: L1+L2 con IC, proxy, LOPO, robustez, contraste SHAP |
| `discussion.tex` | **Nueva**: qué se explica + fidelidad Quantus + limitación Miró-Nicolau |
| `conclusions.tex` | **Nueva**: síntesis del hallazgo |
| `future_work.tex` | Recortada (lo ya hecho dejó de ser futuro) |
| `references.bib` | +4 citas: Bhatt 2020, Alvarez-Melis 2018, Miró-Nicolau 2025, Hedström 2023 |
| `apendix.tex` | Corregido conteo Derived 12→11 |
| abstract | +hallazgo L1/L2 |

**PDF compilado**: 16 páginas (≈9 de cuerpo + apéndice de tablas), **0 errores, 0 citas/refs sin
resolver**. Salida en `latex/main.pdf`.

---

# Pendientes / decisiones abiertas

- **`latex/` está en `.gitignore`** → el paper no se versiona; la fuente de verdad es Overleaf.
  Estos `.tex` editados **no llegan a Overleaf solos**: hay que subirlos o importar el zip.
- **Bhatt: año 2020** (correcto: IJCAI-20) vs **2021** (como lo cita el enunciado). Ajustable.
- **Waterfall local** quitado de results (era de un evento 2016-2017); el contraste ahora es
  beeswarm L1 vs L2. Se puede reañadir uno de E3.
- **`CLAUDE.md` / READMEs** mencionan `modeling/02`–`03` pero no `04`/`05` ni el filtro a 4 regiones.
- **Cambios de código de modeling sin commitear** (el backup/flag de rollback sigue válido).

---

# Cómo regenerar todo

```bash
# Resultados E3 (robusto + proxy + LOPO)        -> eda/L2_Robust_Eval_Report.html
python modeling/03_l2_robust_eval.py
# Sensibilidad del umbral                        -> latex/images/l2_threshold_sensitivity.png
python modeling/04_l2_threshold_sensitivity.py
# Triage operacional                             -> latex/images/operational_triage.png
python modeling/05_operational_triage.py
# SHAP + Quantus                                 -> eda/L1_vs_L2_Experiment_Report.html
python modeling/02_l1_vs_l2_experiment.py
# Compilar el paper                              -> latex/main.pdf
cd latex && latexmk -pdf main.tex
# Regenerar este reporte                         -> docs/reporte_e3.html
make report
```
