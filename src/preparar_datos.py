"""Limpia los CSV descargados de DHIME (IDEAM) y arma el dataset diario.

Entrada:  data/raw/*.csv  (formato DHIME: CodigoEstacion, Parametro, Fecha, Valor, ...)
Salida:   data/processed/dataset.csv  (una fila por día, con la etiqueta llueve_manana)
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
SALIDA = RAIZ / "data" / "processed" / "dataset.csv"

INICIO, FIN = "2010-01-01", "2025-12-31"
UMBRAL_LLUVIA_MM = 1.0  # un día "llueve" si acumula >= 1 mm

# Nombre del parámetro en DHIME -> nombre corto de la columna
PARAMETROS = {
    "Día pluviométrico (convencional)": "lluvia_mm",
    "Temperatura máxima diaria": "tmax",
    "Temperatura mínima diaria": "tmin",
    "Húmeda relativa calculada máxima diaria": "hr_max",
    "Húmeda relativa calculada mínima diaria": "hr_min",
}

# Partición temporal declarada en el Hito 1 (no aleatoria, para evitar fuga de datos)
PARTICION = {"entrenamiento": (2010, 2020), "validacion": (2021, 2022), "prueba": (2023, 2025)}


def cargar_crudos() -> pd.DataFrame:
    archivos = sorted(RAW.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay CSV en {RAW}")
    df = pd.concat([pd.read_csv(f) for f in archivos], ignore_index=True)
    # DHIME acumula descargas: el mismo dato puede venir en varios archivos
    df = df.drop_duplicates(subset=["CodigoEstacion", "Parametro", "Fecha"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="mixed", dayfirst=False)
    df["variable"] = df["Parametro"].map(PARAMETROS)
    desconocidos = df.loc[df["variable"].isna(), "Parametro"].unique()
    if len(desconocidos):
        print("Aviso: se ignoran parámetros no usados:", list(desconocidos))
    return df.dropna(subset=["variable"])


def limpiar(ancho: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    reporte = {}
    # Humedad relativa: físicamente no puede pasar de 100 %
    for c in ["hr_max", "hr_min"]:
        reporte[f"{c}_mayor_100"] = int((ancho[c] > 100).sum())
        ancho[c] = ancho[c].clip(upper=100)
    # Temperatura máxima atípica para Cartagena (> 39 °C): se marca como faltante
    reporte["tmax_mayor_39"] = int((ancho["tmax"] > 39).sum())
    ancho.loc[ancho["tmax"] > 39, "tmax"] = pd.NA
    # Consistencia: la mínima no puede superar la máxima
    malos = ancho["tmin"] > ancho["tmax"]
    reporte["tmin_mayor_tmax"] = int(malos.sum())
    ancho.loc[malos, ["tmin", "tmax"]] = pd.NA
    return ancho, reporte


def construir_dataset() -> pd.DataFrame:
    df = cargar_crudos()
    estaciones = df["CodigoEstacion"].unique().tolist()
    ancho = df.pivot_table(index="Fecha", columns="variable", values="Valor", aggfunc="mean")
    ancho = ancho.reindex(pd.date_range(INICIO, FIN, freq="D"))
    ancho.index.name = "fecha"
    for c in PARAMETROS.values():
        if c not in ancho.columns:
            ancho[c] = pd.NA
    ancho = ancho[list(PARAMETROS.values())].astype("float64")

    ancho, reporte = limpiar(ancho)

    ancho["llueve_hoy"] = (ancho["lluvia_mm"] >= UMBRAL_LLUVIA_MM).astype("float64")
    ancho.loc[ancho["lluvia_mm"].isna(), "llueve_hoy"] = pd.NA
    # Etiqueta: ¿lloverá mañana? (se desplaza un día hacia atrás)
    ancho["llueve_manana"] = ancho["llueve_hoy"].shift(-1)
    ancho["mes"] = ancho.index.month

    anio = ancho.index.year
    ancho["particion"] = pd.NA
    for nombre, (a, b) in PARTICION.items():
        ancho.loc[(anio >= a) & (anio <= b), "particion"] = nombre

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    ancho.to_csv(SALIDA)

    print(f"Estación(es): {estaciones}")
    print(f"Días: {len(ancho)} | con todas las variables: {int(ancho[list(PARAMETROS.values())].notna().all(axis=1).sum())}")
    print(f"Limpieza: {reporte}")
    print(f"Días con lluvia >= {UMBRAL_LLUVIA_MM} mm: {ancho['llueve_hoy'].mean() * 100:.1f} %")
    print(f"Guardado en {SALIDA.relative_to(RAIZ)}")
    return ancho


if __name__ == "__main__":
    construir_dataset()
