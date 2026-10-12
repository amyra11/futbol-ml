# Plan de trabajo (12 días: lun 12 – sáb 24)

**Persona A:** datos y features · **Persona B:** modelos y evaluación.
Cada tarea es un Issue con su rama y PR. Hito duro: **domingo 18**.

| # | Día | Persona A (datos/features) | Persona B (modelos/evaluación) |
|---|-----|----------------------------|--------------------------------|
| 1 | Lun 12 | Juntos: repo, CLAUDE.md, cortes temporales, esquema de features | Igual |
| 2 | Mar 13 | Ingesta y limpieza de los CSV (`python -m src.data.download`) | Módulo de evaluación: log loss, Brier, accuracy, matriz de confusión, calibración |
| 3 | Mié 14 | Forma de los últimos 5 partidos | Baseline y probabilidades implícitas de las cuotas |
| 4 | Jue 15 | Elo y días de descanso | Regresión logística con features provisionales |
| 5 | Vie 16 | Test anti-fuga y entrega de la tabla de features final | Logística con las features reales |
| 6 | Sáb 17 | Poisson simple (opcional) | Random Forest y XGBoost con `TimeSeriesSplit` |
| 7 | Dom 18 | 🔒 **Hito:** todos los modelos corriendo en validación. Si no se llega, se corta Poisson | Igual |
| 8 | Lun 19 | Esqueleto de la app en Streamlit | Calibración y comparación contra las cuotas |
| 9 | Mar 20 | Conectar el modelo final a la app | SHAP global y por partido |
| 10 | Mié 21 | Juntos: **una sola corrida** sobre el conjunto de prueba; se congelan los resultados | Igual |
| 11 | Jue 22 | Pulir la demo | README y figuras finales |
| 12 | Vie 23 | Juntos: presentación y ensayo | Igual |
| — | Sáb 24 | Margen y entrega (no es día de trabajo) | Igual |

## Cortes temporales (a fijar el lunes 12)
- **Entrenamiento:** temporadas más antiguas.
- **Validación:** penúltima temporada (y `TimeSeriesSplit` dentro de entrenamiento para hiperparámetros).
- **Prueba:** última temporada completa; se usa una sola vez el mié 21.

## Esquema de features (una fila por partido)
`date, home, away, result(H/D/A), home_form_pts5, home_gf5, home_ga5, away_form_pts5, away_gf5, away_ga5,
home_elo, away_elo, elo_diff, home_rest_days, away_rest_days` + cuotas (solo para comparar, no como feature del modelo base).

## Reglas de recorte
- Si el hito del domingo 18 no se cumple: se corta Poisson y H2H, no la evaluación.
- Fuera de alcance: xG/FBref, Liga Pro Ecuador, Dixon-Coles → "trabajo futuro" en la presentación.
- Demo mínima: dos selectores, tres probabilidades, un gráfico SHAP.

## Definición de "hecho" por PR
- CI en verde (`pytest`).
- Si toca features: el test anti-fuga pasa y el compañero lo revisó.
- Referencia al Issue (`Closes #N`).

## Resultados esperables
Accuracy ~50–55 % y log loss mejor que el baseline. Más alto = revisar fuga de datos.
