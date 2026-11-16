"""Hito 2: corre las líneas base y un primer modelo sin ajustar sobre VALIDACIÓN (2021-2022).

El conjunto de prueba (2023-2025) NO se usa aquí: se evalúa una sola vez al final del proyecto.

Uso:  python src/lineas_base.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preparar_datos import construir_dataset

SEMILLA = 42
RAIZ = Path(__file__).resolve().parents[1]
EXP = RAIZ / "experiments"

PREDICTORES = ["lluvia_mm", "llueve_hoy", "tmax", "tmin", "hr_max", "hr_min", "mes_sin", "mes_cos"]


def agregar_mes_ciclico(df: pd.DataFrame) -> pd.DataFrame:
    # El mes es cíclico (diciembre está cerca de enero): se codifica con seno y coseno
    df = df.copy()
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    return df


def evaluar(nombre: str, y_real, y_pred, y_score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    return {
        "modelo": nombre,
        "f1_llueve": round(f1_score(y_real, y_pred, zero_division=0), 3),
        "auc": round(roc_auc_score(y_real, y_score), 3),
        "precision": round(precision_score(y_real, y_pred, zero_division=0), 3),
        "recall": round(recall_score(y_real, y_pred, zero_division=0), 3),
        "exactitud": round(float((y_real == y_pred).mean()), 3),
        "VP": int(tp), "FP": int(fp), "FN": int(fn), "VN": int(tn),
    }


def main():
    np.random.seed(SEMILLA)
    df = agregar_mes_ciclico(construir_dataset())
    # Solo días con la etiqueta y todos los predictores: todos los modelos se comparan sobre los mismos días
    df = df.dropna(subset=PREDICTORES + ["llueve_manana"])
    entren = df[df["particion"] == "entrenamiento"]
    valid = df[df["particion"] == "validacion"]
    y_entren = entren["llueve_manana"].astype(int)
    y_valid = valid["llueve_manana"].astype(int).to_numpy()

    resultados = []

    # Línea base 1: mayoritario ("nunca llueve")
    clase_mayor = int(y_entren.mode()[0])
    pred = np.full(len(valid), clase_mayor)
    resultados.append(evaluar("Mayoritario (nunca llueve)", y_valid, pred, pred.astype(float)))

    # Línea base 2: persistencia ("mañana será igual que hoy")
    pred = valid["llueve_hoy"].astype(int).to_numpy()
    resultados.append(evaluar("Persistencia (igual que hoy)", y_valid, pred, pred.astype(float)))

    # Primer resultado: regresión logística SIN ajuste de hiperparámetros (preliminar)
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEMILLA))
    modelo.fit(entren[PREDICTORES], y_entren)
    prob = modelo.predict_proba(valid[PREDICTORES])[:, 1]
    resultados.append(evaluar("Regresión logística (preliminar, sin ajuste)", y_valid, (prob >= 0.5).astype(int), prob))

    tabla = pd.DataFrame(resultados)
    EXP.mkdir(exist_ok=True)
    tabla.to_csv(EXP / "hito2_resultados_validacion.csv", index=False)
    meta = {
        "semilla": SEMILLA,
        "conjunto_evaluado": "validacion 2021-2022",
        "dias_entrenamiento": int(len(entren)),
        "dias_validacion": int(len(valid)),
        "porcentaje_lluvia_validacion": round(float(y_valid.mean()) * 100, 1),
        "predictores": PREDICTORES,
        "resultados": resultados,
    }
    (EXP / "hito2_resultados_validacion.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nEntrenamiento: {len(entren)} días | Validación: {len(valid)} días "
          f"({meta['porcentaje_lluvia_validacion']} % con lluvia al día siguiente)\n")
    print(tabla[["modelo", "f1_llueve", "auc", "precision", "recall", "exactitud"]].to_string(index=False))
    print(f"\nResultados guardados en experiments/ (semilla {SEMILLA})")


if __name__ == "__main__":
    main()
