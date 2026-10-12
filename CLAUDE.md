# futbol-ml — Instrucciones para Claude

Proyecto de Machine Learning (2 personas, 12 días) que estima P(victoria local / empate / victoria visitante)
en la Premier League usando **solo información previa al partido**, con demo en Streamlit.
Prioridad: **calidad de las probabilidades y evaluación honesta**, no accuracy alto.

## Alcance (recortado)
- Una sola liga: Premier League, ~8 temporadas de football-data.co.uk (resultados + cuotas en el mismo CSV).
- Features: forma últimos 5 partidos, Elo previo al partido, días de descanso. H2H solo si sobra tiempo.
- Modelos: baseline de frecuencias, regresión logística, Random Forest, XGBoost. Poisson simple opcional.
- Fuera de alcance (trabajo futuro): xG/FBref, Liga Pro Ecuador, Dixon-Coles.
- Demo mínima: dos selectores de equipo, las tres probabilidades y un gráfico SHAP.

## Reglas innegociables
1. **Sin fuga de datos.** Toda feature de un partido usa únicamente partidos anteriores (`shift(1)` / Elo previo al partido).
   Test obligatorio: cambiar resultados futuros no debe alterar features pasadas.
2. **Validación temporal.** Entrenar con el pasado, probar con el futuro. Nunca `train_test_split` aleatorio.
   Hiperparámetros con `TimeSeriesSplit` y `RandomizedSearchCV` pequeño.
3. **El conjunto de prueba se toca una sola vez** (mié 21), al final. Antes, solo validación.
4. **Orden de clases fijo: H, D, A.** `sklearn.metrics.log_loss` asume orden alfabético (A, D, H), lo que calcularía
   mal las métricas sin error. Usar siempre las funciones de `src/evaluation` y `to_hda()` para reordenar salidas de modelos.
5. **Comparar siempre contra** el baseline de frecuencias y las probabilidades implícitas de las cuotas (sin margen de la casa).
6. Si el accuracy sale muy por encima de ~55 %, sospechar fuga antes de celebrar.

## Estructura
```
data/raw/        CSV descargados (no se suben a git)
data/processed/  tablas limpias y de features
src/data/        download.py, clean.py
src/features/    form.py, elo.py, rest.py, build.py
src/models/      baseline.py, logistic.py, forest.py, xgb.py
src/evaluation/  métricas, calibración, probabilidades de cuotas
app/             Streamlit
tests/           pytest (incluye test anti-fuga)
docs/PLAN.md    plan día a día
```

## Flujo de trabajo (Git)
- `main` protegida: solo se fusiona vía Pull Request con CI en verde y 1 revisión del compañero.
- Ramas: `feat/<tarea>`, `fix/<tarea>`. Commits pequeños, en español o inglés, imperativos.
- Un Issue por tarea del plan (ver `docs/PLAN.md`); el PR referencia `Closes #N`.
- Los PR que tocan `src/features/` requieren revisión anti-fuga explícita.

## Convenciones de código
- Python 3.11+, pandas, scikit-learn, xgboost, shap, streamlit.
- Tipado en funciones públicas, docstrings cortos. Aleatoriedad con `random_state=42`.
- Notebooks solo para exploración; la lógica vive en `src/` y se prueba en `tests/`.
- Comandos: `pip install -r requirements.txt`, `pytest -q`, `python -m src.data.download`.

## Cómo ayudar (Claude)
- Antes de proponer una feature, verificar que no usa información posterior al partido.
- Revisar PRs buscando fuga de datos, orden de clases y uso indebido del conjunto de prueba.
- Reportar resultados tal cual, incluso si no mejoran al baseline.
- Herramienta educativa y analítica; no es consejo para apostar.
