# Contexto para Claude (y para el equipo)

Proyecto académico de 2 personas: predecir H/D/A (local/empate/visitante) en la Premier League
con información disponible ANTES del partido. Entrega: sábado 24 de octubre de 2026.
Foco: calidad de probabilidades y evaluación honesta, no accuracy alto.

## Reglas que no se rompen
1. **Sin fuga de datos.** Toda feature de un partido usa solo partidos de fechas anteriores
   (`shift(1)` antes de cualquier `rolling`). `tests/test_leakage.py` debe pasar siempre.
   Las cuotas (`odds_*`) son referencia de comparación, NO features del modelo.
2. **Cortes temporales fijos** en `src/config.py` y aplicados con `src.splits.split()`.
   Nada de `train_test_split` aleatorio. Ajuste de hiperparámetros solo con `TimeSeriesSplit`
   dentro de train.
3. **El conjunto de prueba (2024-25) se usa una sola vez**, el miércoles 21, con
   `split(df, use_test=True)`. Antes de eso todo se compara en validación (2023-24).
4. **Probabilidades siempre en orden H, D, A.** Con modelos de sklearn/xgboost usar
   `to_hda(model, model.predict_proba(X))`. No usar `sklearn.metrics.log_loss` directo.
5. **Toda métrica pasa por `src.evaluation.metrics.evaluate()`** para que la comparación sea justa.

## Estructura
- `src/data/` ingesta y limpieza (Persona A)
- `src/features/` forma, Elo, descanso (Persona A)
- `src/models/` baseline, logística, RF, XGBoost, Poisson
- `src/evaluation/` métricas, calibración (Persona B)
- `notebooks/` exploración: `NN_tema_iniciales.ipynb`. La lógica reutilizable va a `src/`.
- `app/` demo en Streamlit
- `data/` no se versiona; se regenera con los scripts

## Comandos
```
python -m src.data.download    # CSV crudos a data/raw/
python -m src.data.clean       # data/processed/matches.csv
python -m src.features.build   # data/processed/features.csv
python -m src.models.final     # models/modelo_final.joblib (modelo final de la app)
python -m pytest -q            # tests (correr antes de cada PR)
```

## Flujo de trabajo
- Nunca push directo a `main`. Rama `feat/<tema>` → PR → revisión del compañero → merge.
- Un PR = una tarea de `docs/PLAN.md`. PRs pequeños.
- Al revisar, Claude debe buscar primero fuga de datos y errores de orden de columnas.

## Resultados esperables
Referencias en validación (2023-24, 380 partidos con cuotas): baseline de frecuencias ≈ 1.05,
regresión logística ≈ 0.93 y cuotas del mercado ≈ 0.91 de log loss (accuracy ~50–59 %).
Un modelo razonable queda entre el baseline y las cuotas. Un log loss por debajo de las cuotas
o un accuracy > 60 % casi seguro indica fuga de datos.
Los valores fuera de muestra en entrenamiento (`TimeSeriesSplit`) son algo peores, porque
2023-24 fue una temporada más predecible.
