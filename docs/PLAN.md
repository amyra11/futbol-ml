# Plan de trabajo (12 – 24 de octubre de 2026)

A = datos y features · B = modelos y evaluación · ✅ = ya hecho en la base inicial

## Hecho en la base inicial
- ✅ Estructura del repo, `CLAUDE.md`, configuración y cortes temporales
- ✅ Descarga (`src/data/download.py`) y limpieza (`src/data/clean.py`)
- ✅ Forma (últimos 5), Elo y días de descanso (`src/features/build.py`)
- ✅ Tests anti-fuga (`tests/test_leakage.py`)
- ✅ Métricas: log loss, Brier, accuracy, matriz de confusión, probabilidades de cuotas
- ✅ Baseline de frecuencias

## Tareas (cada una = un Issue y un PR)

| # | Día | Quién | Tarea | Listo cuando… |
|---|---|---|---|---|
| 1 | Mar 13 | A | Correr descarga y limpieza con datos reales; revisar nombres de equipos y temporadas con columnas raras | `matches.csv` sin duplicados, 3 040 partidos (8 × 380) |
| 2 | Mar 13 | B | Notebook `01_baseline_B`: baseline y cuotas en validación | Tabla con log loss/Brier/accuracy de ambos |
| 3 | Mié 14 | A | EDA: % H/D/A por temporada, efecto de la temporada sin público (2020-21) | 3–4 gráficos para la presentación |
| 4 | Mié 14–Jue 15 | A | Ajustar Elo: K y ventaja local con log loss en train; regresión a la media entre temporadas; Elo inicial de ascendidos | Elo solo mejora a la logística en validación |
| 5 | Jue 15 | B | Regresión logística (con `StandardScaler`, imputar NaN de inicio) | Mejora al baseline en validación |
| 6 | Vie 16 | A | H2H (opcional) y entrega de `features.csv` final | Tests anti-fuga pasan |
| 7 | Sáb 17 | B | Random Forest y XGBoost con `TimeSeriesSplit` en train | Fila de cada modelo en la tabla comparativa |
| 8 | Sáb 17 | A | Poisson simple (opcional): goles esperados con Elo → matriz de marcadores → H/D/A | Si no está el domingo, se corta |
| **9** | **Dom 18** | **A+B** | **HITO: tabla comparativa de todos los modelos en validación** | |
| 10 | Lun 19 | B | Curvas de calibración; calibrar el mejor modelo si hace falta | Gráfico de calibración vs cuotas |
| 11 | Lun 19–Mar 20 | A | App Streamlit: elegir 2 equipos → probabilidades y factores | Corre local con `streamlit run app/app.py` |
| 12 | Mar 20 | B | SHAP global y por partido; guardar modelo final (`joblib`) | Gráficos SHAP listos |
| **13** | **Mié 21** | **A+B** | **Única corrida en test (2024-25) y resultados congelados** | |
| 14 | Jue 22 | A | Pulir demo, capturas | |
| 15 | Jue 22 | B | README con resultados, limitaciones y figuras | |
| 16 | Vie 23 | A+B | Presentación y ensayo | |
| — | Sáb 24 | — | Margen y entrega | |
