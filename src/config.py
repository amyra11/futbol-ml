"""Configuración central del proyecto. Cambiar aquí, no en cada script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# Liga: código de football-data.co.uk (E0 = Premier League, SP1 = LaLiga)
LEAGUE = "E0"

# Temporadas en formato football-data: "1718" = 2017-18
SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]

# Cortes temporales FIJOS. No cambiar sin acuerdo de ambos.
TRAIN_SEASONS = ["1718", "1819", "1920", "2021", "2122", "2223"]
VAL_SEASONS = ["2324"]
TEST_SEASONS = ["2425"]  # no se usa hasta el miércoles 21

# Clases del objetivo, en este orden en todas las matrices de probabilidad
CLASSES = ["H", "D", "A"]

# Parámetros de features
FORM_WINDOW = 5
ELO_INIT = 1500.0
ELO_K = 20.0
ELO_HOME_ADV = 60.0

RANDOM_STATE = 42
