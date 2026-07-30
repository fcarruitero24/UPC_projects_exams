"""Genera docs/index.html a partir de los datasets del repositorio.

Reentrena los dos pipelines con semilla fija, serializa el Random Forest de
MetalX para que el navegador pueda hacer inferencia sin servidor, e inyecta
todo en la plantilla.

Uso:
    python docs/src/build.py

Requiere: pandas, numpy, scikit-learn  (ver requirements.txt)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

SRC = Path(__file__).resolve().parent
REPO = SRC.parent.parent
SEED = 42
FEATS = ["Temp_Matriz_C", "Presion_Inyeccion_Bar", "Tiempo_Ciclo_s", "Vibracion_RMS"]

# Las metricas de las redes neuronales se toman de los notebooks ejecutados:
# reentrenarlas aqui exigiria TensorFlow, que no es necesario para el resto.
RNA_METALX = {"mae": 9.7960, "mse": 240.0543, "r2": 0.5528,
              "config": "Densa 16→8→1, ReLU, Adam, 100 épocas"}
RNA_RUTAS = {"mae": 18.8976, "rmse": 27.5680, "r2": 0.8440,
             "config": "Keras, capas densas + ReLU"}


def met(name, yt, yp, rmse=False):
    d = {"modelo": name,
         "mae": round(mean_absolute_error(yt, yp), 4),
         "r2": round(r2_score(yt, yp), 4)}
    mse = mean_squared_error(yt, yp)
    d["rmse" if rmse else "mse"] = round(np.sqrt(mse) if rmse else mse, 4)
    return d


def build_metalx():
    datos = pd.read_csv(REPO / "02-control-calidad-metalx" / "df_metalx.csv", sep=";")
    X, y = datos[FEATS], datos["Defectos_Lote"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    gt = GridSearchCV(DecisionTreeRegressor(random_state=SEED),
                      {"max_depth": [3, 5], "min_samples_split": [2, 5, 10]}, cv=5)
    gt.fit(Xtr, ytr)
    tree = gt.best_estimator_

    gr = GridSearchCV(RandomForestRegressor(random_state=SEED),
                      {"n_estimators": [100], "max_depth": [3, 5],
                       "min_samples_split": [2, 5, 10]}, cv=5)
    gr.fit(Xtr, ytr)
    rf = gr.best_estimator_
    p_rf = rf.predict(Xte)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_t = cross_val_score(tree, Xtr, ytr, cv=kf, scoring="neg_mean_squared_error")
    cv_r = cross_val_score(rf, Xtr, ytr, cv=kf, scoring="neg_mean_squared_error")

    counts, edges = np.histogram(y, bins=40)

    # Cada arbol como arrays planos: en JS la inferencia es un while-loop.
    # Los umbrales van a precision completa a proposito — redondearlos desplaza
    # decisiones porque varios caen exactamente sobre valores del dataset.
    trees = [{"f": t.tree_.feature.tolist(),
              "t": t.tree_.threshold.tolist(),
              "l": t.tree_.children_left.tolist(),
              "r": t.tree_.children_right.tolist(),
              "v": t.tree_.value.ravel().tolist()} for t in rf.estimators_]

    return {
        "n": len(datos),
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "skew": round(float(y.skew()), 4),
        "mediana": round(float(y.median()), 2),
        "media": round(float(y.mean()), 2),
        "maximo": round(float(y.max()), 2),
        "std": round(float(y.std()), 2),
        "metricas": [
            {"modelo": "Red Neuronal (RNA)", **RNA_METALX},
            {**met("Random Forest", yte, p_rf),
             "config": f"max_depth={gr.best_params_['max_depth']}, 100 árboles"},
            {**met("Árbol de Decisión", yte, tree.predict(Xte)),
             "config": f"max_depth={gt.best_params_['max_depth']}"},
        ],
        "importancias": [{"variable": v, "valor": round(float(i), 4)}
                         for v, i in sorted(zip(FEATS, rf.feature_importances_),
                                            key=lambda p: -p[1])],
        "correlaciones": [{"variable": v, "valor": round(float(datos[v].corr(y)), 4)}
                          for v in sorted(FEATS, key=lambda v: -abs(datos[v].corr(y)))],
        "histograma": {"conteos": counts.tolist(),
                       "bordes": [round(float(e), 2) for e in edges]},
        "dispersion": [[round(float(a), 2), round(float(b), 2)]
                       for a, b in zip(yte.to_numpy(), p_rf)],
        "cv": [{"modelo": "Random Forest", "mse": round(float(-cv_r.mean()), 2),
                "std": round(float(cv_r.std()), 2)},
               {"modelo": "Árbol de Decisión", "mse": round(float(-cv_t.mean()), 2),
                "std": round(float(cv_t.std()), 2)}],
        "rangos": [{"variable": v,
                    "min": round(float(datos[v].min()), 2),
                    "max": round(float(datos[v].max()), 2),
                    "mediana": round(float(datos[v].median()), 2)} for v in FEATS],
        "modelo_rf": {"features": FEATS, "trees": trees},
    }


def build_rutas():
    d = pd.read_csv(REPO / "01-optimizacion-rutas-ecommerce" /
                    "dirty_ecommerce_logistics_route_planning_dataset.csv")
    n_raw = len(d)
    nulos = {c: int(v) for c, v in d.isnull().sum().items() if v > 0}

    # Limpieza, replicando el notebook del proyecto 01.
    c = d.drop_duplicates().copy()
    c.loc[(c["distance_km"] <= 0) | (c["distance_km"] > 1000), "distance_km"] = c["distance_km"].median()
    c["distance_km"] = c["distance_km"].fillna(c["distance_km"].median())
    c.loc[~c["order_priority"].isin([1, 2, 3, 4]), "order_priority"] = c["order_priority"].mode()[0]
    c["order_weight_kg"] = c["order_weight_kg"].fillna(c["order_weight_kg"].median())
    c["vehicle_utilization_ratio"] = c["vehicle_utilization_ratio"].fillna(c["vehicle_utilization_ratio"].median())
    c.loc[(c["average_speed_kmph"] <= 0) | (c["average_speed_kmph"] > 150), "average_speed_kmph"] = c["average_speed_kmph"].median()
    c["average_speed_kmph"] = c["average_speed_kmph"].fillna(c["average_speed_kmph"].median())
    c.loc[(c["time_of_day"] < 0) | (c["time_of_day"] > 23), "time_of_day"] = c["time_of_day"].mode()[0]
    c["fuel_cost_per_km"] = c["fuel_cost_per_km"].fillna(c["fuel_cost_per_km"].median())
    c["driver_cost_per_hour"] = c["driver_cost_per_hour"].fillna(c["driver_cost_per_hour"].median())
    c.loc[c["optimized_route_cost"] > 10000, "optimized_route_cost"] = c["optimized_route_cost"].median()
    c.loc[(c["delivery_efficiency_score"] < 0) | (c["delivery_efficiency_score"] > 1), "delivery_efficiency_score"] = c["delivery_efficiency_score"].median()

    # Feature engineering.
    fe = c.copy()
    fe["fuel_cost_total"] = fe["distance_km"] * fe["fuel_cost_per_km"]
    fe["driver_cost_total"] = fe["delivery_time_window_hrs"] * fe["driver_cost_per_hour"]
    fe["is_peak_hour"] = fe["time_of_day"].apply(lambda h: 1 if (7 <= h <= 9) or (18 <= h <= 20) else 0)
    fe["adverse_conditions_index"] = (fe["traffic_density_index"] + fe["weather_impact_index"]) / 2
    fe = fe.drop(columns=["order_latitude", "order_longitude"])

    X = fe.drop(columns=["optimized_route_time_min", "optimized_route_cost",
                         "delivery_efficiency_score", "route_reliability_index"])
    y = fe["optimized_route_time_min"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(Xtr), sc.transform(Xte)

    gt = GridSearchCV(DecisionTreeRegressor(random_state=SEED), {"max_depth": [3, 5, 7]}, cv=5)
    gt.fit(Xtr, ytr)
    tree = gt.best_estimator_
    p_tree = tree.predict(Xte)

    gk = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": list(range(1, 21))}, cv=5)
    gk.fit(Xtr_s, ytr)
    knn = gk.best_estimator_

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    return {
        "n_raw": n_raw,
        "n_limpio": len(c),
        "duplicados": n_raw - len(c),
        "nulos": nulos,
        "n_features": X.shape[1],
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "metricas": [
            {**met("Árbol de Decisión", yte, p_tree, rmse=True),
             "config": f"max_depth={gt.best_params_['max_depth']}",
             "cv_r2": round(float(cross_val_score(tree, Xtr, ytr, cv=kf, scoring="r2").mean()), 4)},
            {"modelo": "Red Neuronal", **RNA_RUTAS, "cv_r2": None},
            {**met("KNN", yte, knn.predict(Xte_s), rmse=True),
             "config": f"k={gk.best_params_['n_neighbors']}",
             "cv_r2": round(float(cross_val_score(knn, Xtr_s, ytr, cv=kf, scoring="r2").mean()), 4)},
        ],
        "importancias": [{"variable": v, "valor": round(float(i), 4)}
                         for v, i in sorted(zip(X.columns, tree.feature_importances_),
                                            key=lambda p: -p[1]) if i > 0],
        "dispersion": [[round(float(a), 2), round(float(b), 2)]
                       for a, b in zip(yte.to_numpy(), p_tree)],
        "spearman_dist_costo": 0.8358,
    }


def main():
    data = {"metalx": build_metalx(), "rutas": build_rutas()}
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    # El JSON vive dentro de un <script>: hay que neutralizar cierres de etiqueta.
    blob = blob.replace("</", "<\\/")

    tpl = (SRC / "template.html").read_text(encoding="utf-8")
    if "/*__DATA__*/" not in tpl:
        raise SystemExit("template.html: falta el marcador /*__DATA__*/")

    dest = REPO / "docs" / "index.html"
    dest.write_text(tpl.replace("/*__DATA__*/", blob), encoding="utf-8", newline="")

    print(f"{dest.relative_to(REPO)}  ({dest.stat().st_size / 1024:.1f} KB)")
    for proj in ("metalx", "rutas"):
        print(f"\n{proj}:")
        for m in data[proj]["metricas"]:
            print(f"  {m['modelo']:22} R² = {m['r2']}")


if __name__ == "__main__":
    main()
