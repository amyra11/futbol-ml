"""Descarga los CSV de football-data.co.uk a data/raw/.

Uso:  python -m src.data.download
"""
import urllib.request

from src.config import DATA_RAW, LEAGUE, SEASONS

URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"


def download(seasons=SEASONS, league=LEAGUE, force=False):
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        dest = DATA_RAW / f"{league}_{season}.csv"
        if dest.exists() and not force:
            print(f"ya existe {dest.name}")
            continue
        url = URL.format(season=season, league=league)
        print(f"descargando {url}")
        urllib.request.urlretrieve(url, dest)


if __name__ == "__main__":
    download()
