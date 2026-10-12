# Predicción de resultados de fútbol con Machine Learning

Estimación de la probabilidad de victoria local, empate o victoria visitante en la Premier League,
usando solo información disponible antes del partido.

> Herramienta educativa y analítica. No es un consejo para apostar.

## Preguntas
1. ¿Cuánto mejoran las features (forma, Elo, descanso) frente a la frecuencia histórica?
2. ¿Qué familia funciona mejor: lineal, árboles o Poisson?
3. ¿Qué tan cerca llegamos de las probabilidades implícitas de las casas de apuestas?
4. ¿Qué variables pesan más en cada predicción?

## Datos
Premier League 2017-18 a 2024-25, de [football-data.co.uk](https://www.football-data.co.uk/).

| Conjunto | Temporadas | Uso |
|---|---|---|
| Entrenamiento | 2017-18 a 2022-23 | Ajuste de modelos (`TimeSeriesSplit` para hiperparámetros) |
| Validación | 2023-24 | Comparación y selección de modelos |
| Prueba | 2024-25 | Resultado final, evaluado una sola vez |

## Cómo correrlo
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.data.download
python -m src.data.clean
python -m src.features.build
python -m pytest -q
```

## Resultados
_Pendiente (miércoles 21)._

| Modelo | Log loss | Brier | Accuracy |
|---|---|---|---|
| Baseline (frecuencias) | | | |
| Regresión logística | | | |
| Random Forest | | | |
| XGBoost | | | |
| Cuotas de mercado | | | |

## Limitaciones
- El azar pesa mucho y los empates son muy difíciles de predecir.
- No hay datos de lesiones, alineaciones ni motivación.
- La temporada 2020-21 se jugó sin público, lo que reduce la ventaja local.

## Equipo
- Amyra Loayza: _rol_
- _Compañero_: _rol_
