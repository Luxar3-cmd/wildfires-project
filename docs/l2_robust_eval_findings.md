# L2 robust evaluation & L1-as-proxy — findings (paper notes)

Source: `modeling/03_l2_robust_eval.py` → `eda/L2_Robust_Eval_Report.html`.
Dataset: `conaf_enriched_2012_2018` after the **ERA5 grid de-duplication fix** (see note below),
**32,162 usable rows**, **L2 = 11 positives** (EWE, FLI ≥ 10,000 kW/m via FRP→FLI), L1 = 78 positives
(area ≥ 1,000 ha). Model: XGBoost, identical config to the L1-vs-L2 contrast (`modeling/02`).
Protocol: **repeated stratified 5-fold CV (20 repeats)** for confidence intervals; **leave-one-positive-out
(LOPO)** for the strictest held-out test; rank-based metrics on averaged out-of-fold (OOF) risk.

## A. Robustness (repeated 20×5 CV, 95% CI from 2.5/97.5 percentiles)

| Model | ROC-AUC | ROC-AUC 95% CI | PR-AUC | PR-AUC 95% CI | PR base (prevalence) |
|---|---|---|---|---|---|
| **L2 (intensity)** | 0.836 | **[0.753, 0.896]** | 0.114 | [0.052, 0.157] | 0.00034 |
| L1 (area, ref.) | 0.916 | [0.900, 0.925] | 0.101 | [0.077, 0.120] | 0.00243 |

The L2 ROC-AUC interval is wide (±~0.07) — the honest signature of an 11-positive class — yet its **lower
bound (0.75) stays well above chance**. PR-AUC (0.114) is ~335× the prevalence baseline. By contrast L1's
ROC interval is tight ([0.900, 0.925], 78 positives).

Rank metrics (OOF L2 risk): recall@top-1% = 0.36 (4/11), **recall@top-5% = 0.45 (5/11)**, recall@top-10% = 0.64 (7/11).

## B. Is L1 (area) a proxy for L2 (intensity)?

| Test | Value | Reference |
|---|---|---|
| AUC of L1 score predicting L2 | **0.873** | vs L2's own model 0.836 |
| PR-AUC of L1 score for L2 | 0.090 | vs L2 own 0.114 |
| Recall@top-5% using L1 score | 0.55 | vs L2 own 0.45 |
| Spearman(L1 score, L2 score) | 0.429 | agreement of the two risk surfaces |
| Label overlap (L2 that are also L1) | **6 / 11** | area ≥ 1,000 ha |

## LOPO — per held-out megafire (strictest test)

| # | LOPO percentile | type | region | area (ha) |
|---|---|---|---|---|
| 1 | 99.97% | pure-L2 | Biobío | 313 |
| 2 | 99.96% | pure-L2 | Biobío | 185 |
| 3 | 99.60% | L1∧L2 | O'Higgins | 46,490 |
| 4 | 99.11% | L1∧L2 | Maule | 159,813 |
| 5 | 99.01% | L1∧L2 | O'Higgins | 1,298 |
| 6 | 97.39% | L1∧L2 | Biobío | 10,830 |
| 7 | 85.87% | pure-L2 | Araucanía | 291 |
| 8 | 85.48% | L1∧L2 | Maule | 13,833 |
| 9 | 75.04% | L1∧L2 | Araucanía | 1,849 |
| 10 | 73.12% | pure-L2 | O'Higgins | 140 |
| 11 | **41.76%** | pure-L2 | Araucanía | 578 |

## Findings

1. **The intensity model generalizes to all held-out events and is more stable than the pre-fix baseline.**
   Under LOPO, **all 11/11** held-out megafires rank above the 41st risk percentile, 9/11 above the 73rd and
   6/11 above the 97th — the model is not memorizing. The weakest case (Araucanía, 578 ha, pure-L2) now lands
   at the **41.8th percentile**; in the pre-fix data it had collapsed to the 7.7th. That jump is attributable
   to the grid fix: its meteorological features now come from the real ERA5 cell rather than a snapped
   neighbour. Residual instability (wide CI, one event near the median) remains the small-sample signature of
   an 11-positive class.

2. **L2 is not reducible to an area proxy.** The two top-ranked held-out events are **pure-L2** (Biobío,
   313 and 185 ha) — small fires of high intensity placed in the top 0.04% by meteorological features
   alone. An area-based model (L1) would rank these low. The intensity label therefore carries genuine
   signal that area does not.

3. **L1 is a partial, not a substitute, proxy.** Operationally the area-trained L1 score ranks EWE events
   at least as well as the intensity-trained model (AUC 0.873 vs 0.836, and recall@top-5% 0.55 vs 0.45) —
   largely because 6/11 EWEs are also large fires that L1 is trained to detect, and because L2's own model is
   data-starved. But moderate rank agreement (Spearman 0.429) and 6/11 label overlap confirm that **area and
   intensity diverge for a non-trivial subset** (5 small-area high-intensity events) that L1 structurally
   cannot flag.

4. **Net.** L1 and L2 are **complementary**, not interchangeable. This is consistent with the divergent
   Tree-SHAP explanations observed in the L1-vs-L2 contrast (`modeling/02`).

## Implication / limitation

With only 11 EWE positives the intensity model is **capable but high-variance**: it captures fire-behavior
signal that area cannot, yet the confidence intervals are wide and the weakest held-out event sits near the
median. The grid de-duplication removed a data artifact and improved LOPO stability, but it does not change
the central limitation: the FRP-only L2 label is too sparse for a robust intensity model, and a defensible
EWE classifier would require the in-situ measurements the full EWE definition relies on (rate of spread, spot
distance), which are not currently available. Until then, L1 (area) is the more stable signal and serves as a
partial proxy for ranking EWE risk, with the explicit caveat that it misses small-area high-intensity events.
