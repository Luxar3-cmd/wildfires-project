# XAI-project

EDA del registro histórico de incendios forestales CONAF (2002-2020) y pipeline de enriquecimiento con ERA5-Land. Insumo para modelos predictivos con XAI (SHAP/LIME).

## Quick start

```bash
git clone <repo-url>
cd XAI-project
make setup          # crea .venv e instala dependencias
# edita .env con tus credenciales (ITREND_API_KEY, CDSAPI_KEY)
make notebook       # lanza Jupyter Lab
```

Python 3.12 recomendado. En Fedora/RHEL: `dnf install python3.12`. En macOS: `brew install python@3.12`.

Ver `make help` para todos los targets disponibles.

## Estructura

```
.
├── data/                     # gitignored
│   ├── raw/                  # CONAF + NetCDF ERA5
│   ├── interim/              # CONAF limpio
│   ├── processed/            # dataset enriquecido
│   └── runs/                 # estado de ejecuciones
├── src/                      # módulos reutilizables
├── eda/01_conaf_eda.ipynb
└── scripts/
    ├── preprocess.py         # CLI pipeline
    └── monitor.py            # dashboard local
```

## Credenciales

**itrend (CONAF):** obtén tu API token en https://datospararesiliencia.cl → perfil → API Token. Pégalo en `.env` como `ITREND_API_KEY=tu-uuid`.

**Copernicus CDS (ERA5):** crea cuenta en https://cds.climate.copernicus.eu, acepta términos de ERA5-Land, y crea `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: TU-API-KEY-AQUI
```

Alternativa: define `CDSAPI_KEY` en `.env` (sobreescribe `~/.cdsapirc`).

## Uso

```bash
# EDA
make notebook

# Pipeline — smoke test
python scripts/preprocess.py --years 2002-2002 --skip-download

# Pipeline — producción (requiere ERA5-Land descargado)
python scripts/preprocess.py --years 2002-2020

# Dashboard de monitoreo
python scripts/monitor.py --port 8877
# → http://127.0.0.1:8877
```

## Fuentes

- CONAF / itrend: https://datospararesiliencia.cl (DOI: `10.71578/UXAUN5`)
- ERA5-Land: Muñoz Sabater, J. (2019). Copernicus C3S CDS. https://doi.org/10.24381/cds.e2161bac — CC BY 4.0
