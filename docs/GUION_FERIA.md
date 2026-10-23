# Guion para la feria "Be a Dragon" (domingo 25 de octubre)

Todo lo que sigue usa los resultados congelados en `docs/` y el modelo final guardado en
`models/modelo_final.joblib`. Los números de la demo se calcularon con `matchup_features()` y el
modelo final, y se comprobaron ejecutando la app.

---

## 1. Pitch de 30 segundos

> "¿Se puede predecir el fútbol? Entrenamos un modelo con 8 temporadas de la Premier League que,
> antes de cada partido, da la probabilidad de que gane el local, empate o gane el visitante. Lo
> comparamos con las casas de apuestas, que son la referencia más difícil: en una temporada que el
> modelo nunca había visto, llegó al 89 % del camino entre adivinar al azar y las casas, y la
> diferencia con ellas no es estadísticamente significativa. Y además explica por qué predice lo
> que predice. ¿Quieres elegir un partido?"

## 2. Explicación de 2 minutos (público general)

1. **El problema (20 s).** "Queremos saber, antes de que empiece un partido, qué tan probable es
   cada resultado. No decimos 'gana el Liverpool', decimos 'el Liverpool tiene 87 %'. En fútbol el
   azar pesa mucho, así que lo importante es que esos porcentajes sean confiables."
2. **Los datos (20 s).** "Usamos 3 040 partidos de la Premier, de 2017 a 2025. Para cada partido
   calculamos tres cosas que se conocen antes de jugar: la fuerza de cada equipo (un puntaje
   llamado Elo, como en el ajedrez), cómo vienen en sus últimos 5 partidos y cuántos días
   descansaron."
3. **El modelo (20 s).** "Probamos varios: desde lo más simple (usar cuántas veces gana el local
   en general) hasta árboles de decisión. Ganó una regresión logística: un modelo sencillo que
   además se puede explicar."
4. **La evaluación honesta (30 s).** "Separamos los datos como un examen: con unas temporadas
   estudia, con otra elegimos el mejor modelo, y la última, 2024-25, la guardamos bajo llave.
   Elegimos el modelo y lo dejamos escrito **antes** de abrir esa temporada, y la usamos una sola
   vez. Así nadie puede decir que hicimos trampa mirando las respuestas."
5. **El resultado (20 s).** "Con frecuencias históricas se acierta 41 %; nuestro modelo, 52 %; las
   casas de apuestas, 54 %. Pero lo que medimos es la calidad de las probabilidades: ahí llegamos
   al 89 % del camino hasta las casas, y con 380 partidos no se puede afirmar que sean mejores que
   nosotros, aunque probablemente lo sean un poco."
6. **El cierre (10 s).** "Y la app muestra por qué: casi siempre lo que más pesa es la fuerza del
   equipo. Es una herramienta educativa, no para apostar."

---

## 3. Recorrido de la demo (3 partidos)

Antes de empezar: la app abre con Arsenal vs Aston Villa y 7 días de descanso para ambos. Decir
siempre al principio: **"los equipos y su estado son los del final de la temporada 2024-25 (mayo
de 2025)"**.

Secciones de la app, en orden: selección de equipos y descanso → **Probabilidades** → **Goles
esperados (modelo Poisson)** → **Factores clave** → **¿Por qué predice esto el modelo?** (y un
desplegable "Ver detalle por variable").

### Partido 1 · Favorito claro: Liverpool (local) vs Southampton (visitante), descanso 7 / 7

| Sección | Qué se ve | Qué decir |
|---|---|---|
| Elo | Liverpool 1737 · Southampton 1274 | "Liverpool fue campeón 2024-25 y tiene el Elo más alto; Southampton terminó último con el más bajo." |
| Probabilidades | **Gana Liverpool 87 %** · Empate 10 % · Gana Southampton 3 % | "Muy favorito, pero ojo: 87 % significa que más o menos 1 de cada 8 veces *no* gana." |
| Goles esperados | Liverpool 3.79 · Southampton 0.50 · marcador más probable **3–0 (12 %)** | "Aun el marcador más probable sale solo el 12 % de las veces: hay muchísimos marcadores posibles." |
| Factores clave | Liverpool 1.00 pts/partido, racha VDEDE · Southampton 0.20, racha DDEDD | "Liverpool llegó flojo al final porque ya era campeón, pero Southampton venía mucho peor." |
| ¿Por qué…? | Fuerza (Elo) **+1.49** verde, enorme · Forma +0.07 · Descanso 0.00 | "Casi todo lo explica la diferencia de fuerza. La forma apenas suma, y con el mismo descanso, el descanso no cuenta." |

**No abrir** "Ver detalle por variable" en este partido: ahí "Puntos del visitante = 0.2" sale en
rojo, que parece raro (ver pregunta sobre signos raros en la sección 5).

### Partido 2 · Parejo: Brentford (local) vs Chelsea (visitante), descanso 7 / 7

| Sección | Qué se ve | Qué decir |
|---|---|---|
| Elo | Brentford 1569 · Chelsea 1657 | "Chelsea es más fuerte, pero Brentford juega en casa: la ventaja de local vale unos 60 puntos de Elo y casi empata la diferencia." |
| Probabilidades | Gana Brentford **37 %** · Empate 28 % · Gana Chelsea 35 % | "Partido de moneda al aire. Fíjense que el empate llega a 28 %, más alto que en el partido anterior." |
| Goles esperados | Brentford 1.24 · Chelsea 1.45 · marcador más probable **1–1 (12 %)** | "El modelo de goles solo mira el Elo, por eso ve a Chelsea un poco arriba; el modelo principal además mira la forma. Son dos modelos distintos y aquí casi coinciden." |
| Factores clave | Brentford 2.00 pts, 2.0 goles a favor, VVVDE · Chelsea 2.40 pts, 0.6 goles en contra, VVDVV | "Los dos vienen bien. Brentford marca mucho; Chelsea recibe poco." |
| ¿Por qué…? | Fuerza (Elo) **−0.28** rojo · Forma **+0.08** verde · Descanso 0.00 | "El rojo dice: la fuerza de Chelsea le quita opciones a Brentford. El verde: la buena racha goleadora de Brentford le devuelve algo. Por eso queda tan parejo." |

### Partido 3 · El descanso mueve algo: Arsenal (local) vs Chelsea (visitante)

Mover los deslizadores de descanso en este orden:

| Descanso (local / visitante) | Probabilidades (local · empate · visitante) | Barra "Descanso" |
|---|---|---|
| 7 / 7 (por defecto) | **55 %** · 25 % · 20 % | 0.00 |
| 3 / 14 (Arsenal cansado, Chelsea descansado) | **51 %** · 32 % · 18 % | **−0.09** rojo |
| 14 / 3 (Arsenal descansado, Chelsea cansado) | **58 %** · 20 % · 22 % | **+0.08** verde |

Qué decir:
- "Si Arsenal juega con 3 días de descanso y Chelsea con dos semanas, Arsenal baja de 55 % a 51 %
  y sube el empate. Al revés, Arsenal sube a 58 %."
- "El descanso mueve unos 4 puntos; la fuerza del equipo mueve decenas. Coincide con lo que vimos
  en todos los partidos: el descanso pesa como un 5 % de lo que pesa el Elo."
- Goles esperados (1.71 vs 1.07, marcador más probable 1–1 con 11 %) **no cambian** al mover el
  descanso: "el modelo de goles solo usa el Elo".
- Elo: Arsenal 1726 · Chelsea 1657. ¿Por qué…? Fuerza (Elo) +0.22, Forma +0.05 (casi neutra).

Si alguien nota que la victoria de Chelsea sube un poco (20 % → 22 %) cuando Chelsea está cansado:
"Bien visto. Lo que el modelo aprendió del descanso es sobre todo que con un visitante muy
descansado hay más empates. Son efectos pequeños, de pocos puntos, y pueden ser en parte ruido;
además, '14 días' mezcla inicios de temporada y parones de selecciones."

Si preguntan por la forma (+0.05 aunque Chelsea venía mejor, 2.4 pts frente a 1.6): "El modelo
mira más los goles que los puntos: Chelsea ganó mucho pero marcando poco (1.2 por partido). Al
final la forma casi se cancela."

---

## 4. Qué figura mostrar según la pregunta

| Si preguntan por… | Mostrar | Frase clave |
|---|---|---|
| Resultados / "¿qué tan bueno es?" | `docs/fig_comparacion_test.png` | "En la temporada que guardamos bajo llave, quedamos a 0.012 de log loss de las casas (89 % del camino); los árboles son significativamente peores." |
| Calibración / "¿se puede confiar en el %?" | `docs/fig_calibracion.png` | "Cuando decimos 60 %, pasa más o menos el 60 % de las veces; casi igual que las casas." |
| Explicabilidad / "¿qué mira el modelo?" | `docs/fig_shap_global.png` | "El Elo es lo que más pesa, de lejos; la forma ajusta; el descanso casi nada." |
| Explicabilidad de un partido | `docs/fig_shap_ejemplo_agrupado.png` | "Arsenal 5–0 Sheffield: 87 % para el local, sobre todo por la diferencia de Elo." |
| Datos / el empate o el local | `docs/eda_resultados_por_temporada.png`, `docs/eda_ventaja_local.png` | "El local gana el 44 % de las veces; en 2020-21, sin público, ganaron más los visitantes (40 %) que los locales (38 %)." |

---

## 5. Preguntas probables del jurado

1. **¿Qué es Elo?**
   Un puntaje de fuerza, como en el ajedrez: todos empiezan en 1500 (los ascendidos en 1450), y
   después de cada partido el ganador le quita puntos al perdedor; ganarle a un rival fuerte da más
   puntos que ganarle a uno débil. El local recibe 60 puntos extra al calcular quién es favorito.

2. **¿Por qué log loss y no accuracy?**
   Porque queremos probabilidades confiables, no solo acertar el ganador. La accuracy trata igual
   decir 51 % y 99 %; el log loss castiga mucho estar muy seguro y equivocarse, y premia decir
   "esto es parejo" cuando lo es. Además, como el empate nunca es lo más probable, la accuracy
   ignora un cuarto de los partidos.

3. **¿Cómo evitaron la fuga de datos?**
   Cada variable de un partido se calcula solo con partidos de fechas anteriores (por ejemplo, la
   forma se desplaza un partido antes de promediar). Tenemos tests automáticos que lo comprueban, y
   las cuotas nunca entran al modelo, solo se usan para comparar. Los cortes son por temporada,
   nunca al azar.

4. **¿Por qué el test se usó una sola vez?**
   Si miras el test varias veces y vas ajustando, terminas adaptando el modelo a esa temporada y
   el resultado sale optimista. Usarlo una sola vez, con el modelo elegido y escrito antes, es lo
   que hace creíble el número que reportamos.

5. **¿Por qué no eligieron el modelo que salió mejor en test (Poisson, 0.981)?**
   Porque elegir mirando el test es justamente usarlo para seleccionar, y ya no sería una
   evaluación honesta. Además la diferencia es mínima (0.0016) y el bootstrap dice que no es
   significativa: en la práctica están empatados.

6. **¿Por qué el modelo nunca predice empate?**
   Sí le da probabilidad al empate (por ejemplo 28 % en Brentford–Chelsea), pero casi nunca es el
   resultado individual más probable: en test, 0 de 380. Hubo 24 % de empates, pero son muy
   difíciles de anticipar; a las casas de apuestas les pasa lo mismo.

7. **¿Por qué no le ganan a las casas de apuestas?**
   Las casas tienen información que nosotros no: lesiones, alineaciones, noticias y el dinero de
   miles de apostadores que corrige sus cuotas. Usamos el promedio del mercado y le quitamos el
   margen de la casa para compararnos de forma justa. Llegar al 89 % del camino con solo
   resultados pasados es un buen resultado.

8. **¿Por qué los árboles (Random Forest, XGBoost) no ganaron?**
   Con menos de 3 000 partidos de entrenamiento y relaciones casi lineales (más Elo, más probabilidad
   de ganar), su flexibilidad extra termina aprendiendo ruido. En test son significativamente
   peores que el modelo final y perdieron más terreno que los lineales de validación a test.

9. **¿Qué es SHAP?**
   Un método para repartir una predicción entre las variables: cuánto empujó cada una hacia el
   resultado. En un modelo lineal como el nuestro es exacto y se puede sumar por grupos, por eso la
   app muestra "Fuerza", "Forma" y "Descanso". Explica al modelo, no las causas del juego.

10. **¿Qué significa estar calibrado?**
    Que los porcentajes digan la verdad: de todos los partidos a los que el modelo da 60 %, más o
    menos el 60 % termina así. El nuestro tiene un error de calibración de 0.055 y las casas 0.048,
    y con 380 partidos eso está dentro de lo esperable por azar.

11. **¿Qué harían con más tiempo?**
    Agregar lesiones y alineaciones, y datos de calidad de juego como los goles esperados (xG) de
    cada partido; probar con más ligas para tener más datos; y modelar el empate de forma
    específica (por ejemplo, Poisson con correlación entre los goles de ambos equipos).

12. **¿Sirve para apostar?**
    No. Es una herramienta educativa: no le gana a las casas, y aunque se acercara, las casas cobran
    un margen que se come cualquier ventaja pequeña. Su valor está en mostrar cómo se evalúa un
    modelo con honestidad y cómo explicar sus predicciones.

**Extra · "¿Por qué en el detalle por variable sale un signo raro?"**
Puntos, goles y Elo están relacionados entre sí, así que el modelo reparte su efecto entre ellas y
alguna variable suelta puede salir con signo contraintuitivo. Por eso en la app mostramos la suma
por grupo, que sí es estable.

---

## 6. Reparto sugerido

| Parte | Quién | Por qué |
|---|---|---|
| Pitch inicial y problema | Amyra | Abre con la demo en pantalla |
| Datos, Elo, forma y descanso | Amyra | Hizo las variables y el ajuste del Elo |
| Manejo de la app (los 3 partidos) | Amyra maneja, Sebastián explica "¿Por qué…?" | La app es de Amyra; SHAP es de Sebastián |
| Goles esperados (Poisson) | Amyra | Hizo el modelo de Poisson |
| Comparación de modelos y evaluación honesta (train/validación/test) | Sebastián | Hizo los modelos, el bootstrap y la corrida del test |
| Calibración y explicabilidad | Sebastián | Hizo las curvas de calibración y SHAP |
| Preguntas 1, 3, 6, 11 | Amyra | Elo, fuga de datos en las variables, empates, mejoras de datos |
| Preguntas 2, 4, 5, 7, 8, 9, 10, 12 | Sebastián | Métricas, test, modelos, SHAP, calibración |

---

## 7. Checklist del día

**La noche anterior**
- [ ] Laptop cargada al 100 % y **cargador** en la mochila (y un adaptador/extensión si se puede).
- [ ] `git pull` de la rama final y `python -m pytest -q` en verde.
- [ ] Confirmar que existen `data\processed\features.csv` y `models\modelo_final.joblib`. Con
      eso la demo **no necesita internet** (comprobado: la app corre con la red bloqueada).
- [ ] Probar los 3 partidos del recorrido y que los números coinciden con este guion.
- [ ] **Plan B:** carpeta `feria_plan_b` en el escritorio con capturas de la app (los 3 partidos,
      Arsenal–Chelsea en sus 3 descansos) y las figuras de `docs/` (comparación, calibración, SHAP
      global y SHAP del partido). Abrirlas con el visor de fotos si la app falla.
- [ ] Desactivar actualizaciones automáticas y notificaciones de Windows durante la feria.

**Al llegar (antes de que pase el jurado)**
- [ ] App abierta y probada en el navegador; dejarla en Liverpool vs Southampton.
- [ ] Zoom del navegador al 110–125 % para que se lea de lejos.
- [ ] Figuras abiertas en otra pestaña o ventana.

**Comandos para (re)iniciar la app** (PowerShell, dentro de la carpeta `futbol-ml`):

```powershell
.venv\Scripts\activate
streamlit run app/app.py --server.address localhost --browser.gatherUsageStats false
```

- Abre `http://localhost:8501`. Con `--server.address localhost` Streamlit no intenta consultar
  la IP pública ni expone la app a la red wifi de la feria.
- Para detenerla: `Ctrl + C` en la terminal. Para reiniciar: volver a ejecutar el comando.
- Si dice que el puerto está ocupado: añadir `--server.port 8502` y abrir `http://localhost:8502`.
- Si la página se queda en blanco o con error: en la app, menú ⋮ → **Clear cache** y luego
  **Rerun** (o tecla `R`); si sigue, `Ctrl + C` y reiniciar.
- Si falta `models\modelo_final.joblib`, la app lo entrena sola con `features.csv` (tarda unos
  segundos y no necesita internet).
