"""Genera las figuras del reporte (docs/REPORTE.md) con etiquetas en español.

Uso (desde la raíz del proyecto):
    python docs/make_figures.py
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from src.explain import ensemble_shap, group_by_original_feature
from src.models import make_final_model
from src.preprocessing import remove_outliers

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, GREEN, RED, ORANGE, GRAY, PURPLE = "#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8C8C8C", "#8172B2"
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
    "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓", name)


def usd(x, _=None):
    return f"${x/1000:,.0f}k"


raw = pd.read_csv(ROOT / "data" / "train.csv")

# ---------------------------------------------------------------- 1. EDA
print("EDA")

# 1.1 Distribución del precio, normal vs log
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].hist(raw["SalePrice"], bins=50, color=BLUE, edgecolor="white", linewidth=0.5)
axes[0].axvline(raw["SalePrice"].median(), color="black", ls="--", lw=1)
axes[0].text(raw["SalePrice"].median() * 1.05, axes[0].get_ylim()[1] * 0.9, "mediana\n$163k", fontsize=9)
axes[0].annotate("pocas casas\nmuy caras\n(cola larga)", xy=(550_000, 5), xytext=(480_000, 70),
                 arrowprops=dict(arrowstyle="->", color=GRAY), fontsize=9)
axes[0].xaxis.set_major_formatter(usd)
axes[0].set_title("Precio original: asimétrico (sesgo 1.88)")
axes[0].set_xlabel("Precio de venta"); axes[0].set_ylabel("Número de casas")
axes[1].hist(np.log1p(raw["SalePrice"]), bins=50, color=GREEN, edgecolor="white", linewidth=0.5)
axes[1].set_title("log(precio): casi simétrico (sesgo 0.12)")
axes[1].set_xlabel("log(precio de venta)")
save(fig, "01_precio_distribucion")

# 1.2 Datos faltantes
miss = (raw.isna().mean() * 100).loc[lambda s: s > 0].sort_values()
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(miss.index, miss.values, color=[RED if v > 40 else ORANGE if v > 10 else GRAY for v in miss.values],
        edgecolor="white", linewidth=0.5)
for i, v in enumerate(miss.values):
    ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("% de casas sin dato")
ax.set_title("19 variables con datos faltantes")
save(fig, "02_faltantes")

# 1.3 Qué significa la correlación (datos sintéticos)
rng = np.random.default_rng(0)
x = rng.normal(size=200)
examples = [(0.9, "r = 0.9\nfuerte positiva", "Más metros → más precio"),
            (0.0, "r = 0\nsin relación", "Mes de venta → precio"),
            (-0.6, "r = −0.6\nnegativa moderada", "Más antigüedad → menos precio")]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, (r, title, ex) in zip(axes, examples):
    yy = r * x + np.sqrt(1 - r ** 2) * rng.normal(size=200)
    ax.scatter(x, yy, s=10, alpha=0.6, color=BLUE if r > 0 else RED if r < 0 else GRAY)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(ex, fontsize=9, style="italic")
save(fig, "03_correlacion_ejemplos")

# 1.4 Correlaciones con el precio
corr = raw.select_dtypes("number").drop(columns=["Id", "SalePrice"]).corrwith(raw["SalePrice"]).sort_values()
top = pd.concat([corr.head(4), corr.tail(10)])
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh(top.index, top.values, color=[RED if v < 0 else BLUE for v in top.values], edgecolor="white", linewidth=0.5)
for i, v in enumerate(top.values):
    ax.text(v + (0.01 if v >= 0 else -0.01), i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.axvline(0.5, color=GRAY, ls="--", lw=0.8)
ax.set_xlim(-0.25, 0.9)
ax.set_xlabel("Correlación con el precio (r)")
ax.set_title("Las 10 variables más correlacionadas (y las 4 más negativas)")
save(fig, "04_correlaciones")

# 1.5 Calidad vs precio (boxplot explicado)
fig, ax = plt.subplots(figsize=(10, 4.2))
groups = [raw.loc[raw["OverallQual"] == q, "SalePrice"] for q in range(1, 11)]
ax.boxplot(groups, tick_labels=range(1, 11), patch_artist=True, showfliers=True,
           boxprops=dict(facecolor=BLUE, alpha=0.6), medianprops=dict(color="white", lw=2),
           flierprops=dict(markersize=3, alpha=0.5))
ax.yaxis.set_major_formatter(usd)
ax.set_xlabel("Calidad general (1 = muy mala, 10 = excelente)"); ax.set_ylabel("Precio")
ax.set_title("A mayor calidad, mayor precio (r = 0.79)")
ax.annotate("línea blanca = mediana\ncaja = 50% central de las casas", xy=(8, 270_000), xytext=(3.2, 480_000),
            arrowprops=dict(arrowstyle="->", color=GRAY), fontsize=9)
save(fig, "05_calidad_precio")

# 1.6 Outliers
out = raw[(raw["GrLivArea"] > 4000) & (raw["SalePrice"] < 200_000)]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(raw["GrLivArea"], raw["SalePrice"], s=10, alpha=0.4, color=GREEN)
ax.scatter(out["GrLivArea"], out["SalePrice"], s=70, color=RED, zorder=5, label="2 outliers")
ax.annotate("casas enormes\na precio de casa pequeña", xy=(5000, 180_000), xytext=(3300, 520_000),
            arrowprops=dict(arrowstyle="->", color=RED), fontsize=9, color=RED)
ax.yaxis.set_major_formatter(usd)
ax.set_xlabel("Superficie habitable (pies²)"); ax.set_ylabel("Precio")
ax.set_title("Superficie vs precio: la tendencia y sus excepciones")
ax.legend()
save(fig, "06_outliers")

# 1.7 Barrios (mediana)
med = raw.groupby("Neighborhood")["SalePrice"].median().sort_values()
fig, ax = plt.subplots(figsize=(8, 6.5))
ax.barh(med.index, med.values, color=PURPLE, edgecolor="white", linewidth=0.5)
ax.xaxis.set_major_formatter(usd)
ax.set_xlabel("Precio mediano")
ax.set_title("Precio mediano por barrio: de $88k a $315k (×3.6)")
save(fig, "07_barrios")

# ---------------------------------------------------------------- 2. Preprocesado
print("Preprocesado")

# 2.1 Transformación logarítmica en una variable (LotArea)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(raw["LotArea"], bins=60, color=ORANGE, edgecolor="white", linewidth=0.5)
axes[0].set_title(f"LotArea original (sesgo {raw['LotArea'].skew():.1f})")
axes[0].set_xlabel("Tamaño del terreno (pies²)")
axes[1].hist(np.log1p(raw["LotArea"]), bins=60, color=GREEN, edgecolor="white", linewidth=0.5)
axes[1].set_title(f"log(LotArea) (sesgo {np.log1p(raw['LotArea']).skew():.1f})")
axes[1].set_xlabel("log(tamaño del terreno)")
save(fig, "08_log_transform")

# ---------------------------------------------------------------- 3. Modelado
print("Modelado")

# 3.1 Validación cruzada (diagrama)
fig, ax = plt.subplots(figsize=(9, 3.2))
for i in range(5):
    for j in range(5):
        test = i == j
        ax.add_patch(plt.Rectangle((j, 4 - i), 0.95, 0.8, color=ORANGE if test else BLUE, alpha=0.9 if test else 0.55))
        ax.text(j + 0.475, 4 - i + 0.4, "examen" if test else "estudio", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold" if test else "normal")
    ax.text(-0.1, 4 - i + 0.4, f"Ronda {i+1}", ha="right", va="center", fontsize=9)
    ax.text(5.1, 4 - i + 0.4, "→ error", ha="left", va="center", fontsize=9)
ax.text(5.1, -0.45, "Nota final = promedio\nde los 5 errores", ha="left", va="center", fontsize=9, fontweight="bold")
ax.set_xlim(-1, 6.6); ax.set_ylim(-1, 5)
ax.axis("off")
ax.set_title("Validación cruzada de 5 folds: cada casa es 'examen' exactamente una vez")
save(fig, "09_validacion_cruzada")

# 3.2 Sobreajuste (ilustración)
rng = np.random.default_rng(3)
xs = np.sort(rng.uniform(0, 1, 18))
ys = np.sin(2 * np.pi * xs) * 0.8 + xs + rng.normal(0, 0.18, len(xs))
grid = np.linspace(0, 1, 300)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
for ax, deg, title in zip(axes, [1, 4, 15], ["Demasiado simple\n(no aprende)", "Equilibrado\n(aprende el patrón)",
                                             "Demasiado complejo\n(memoriza el ruido)"]):
    coefs = np.polyfit(xs, ys, deg)
    ax.scatter(xs, ys, color=BLUE, s=18, zorder=3)
    ax.plot(grid, np.clip(np.polyval(coefs, grid), -2, 3), color=RED if deg != 4 else GREEN, lw=2)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylim(-1.2, 2.5)
save(fig, "10_sobreajuste")

# 3.3 Comparación de modelos (resultados de notebooks/03_modeling.ipynb)
results = pd.Series({
    "Mediana (referencia)": 0.3999, "Regresión lineal": 0.2090, "Random Forest": 0.1290,
    "LightGBM": 0.1147, "Ridge": 0.1127, "XGBoost": 0.1108, "ElasticNet": 0.1107, "Lasso": 0.1106,
    "Voting (4 modelos)": 0.1081, "Stacking (4 modelos)": 0.1077, "Voting Lasso + XGBoost": 0.1070,
}).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
colors = [GREEN if n == "Voting Lasso + XGBoost" else GRAY if n in ("Mediana (referencia)", "Regresión lineal") else BLUE
          for n in results.index]
ax.barh(results.index, results.values, color=colors, edgecolor="white", linewidth=0.5)
for i, v in enumerate(results.values):
    ax.text(v + 0.004, i, f"{v:.3f}  (~{(np.exp(v)-1)*100:.0f}%)", va="center", fontsize=8)
ax.set_xlim(0, 0.5)
ax.set_xlabel("Error (RMSE en log) — más bajo es mejor. Entre paréntesis: error típico aproximado en %")
ax.set_title("Comparación de modelos (validación cruzada, 5 folds)")
save(fig, "11_modelos")

# ---------------------------------------------------------------- 4. Evaluación
print("Evaluación (entrena el modelo, ~1 min)")
train = remove_outliers(raw.copy())
y = np.log1p(train.pop("SalePrice"))
X = train
oof = cross_val_predict(make_final_model(), X, y, cv=KFold(5, shuffle=True, random_state=42))
actual, pred = np.expm1(y), np.expm1(oof)
pct = (pred - actual) / actual

# 4.1 Predicho vs real
fig, ax = plt.subplots(figsize=(6.5, 6.5))
lim = [30_000, 800_000]
ax.fill_between(lim, [l * 0.9 for l in lim], [l * 1.1 for l in lim], color=GREEN, alpha=0.18,
                label=f"±10% ({(pct.abs() <= .1).mean():.0%} de las casas)")
ax.plot(lim, lim, color="black", lw=1, label="Predicción perfecta")
ax.scatter(actual, pred, s=9, alpha=0.5, color=BLUE)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
ticks = [50_000, 100_000, 200_000, 400_000, 700_000]
ax.set_xticks(ticks, [usd(t) for t in ticks]); ax.set_yticks(ticks, [usd(t) for t in ticks]); ax.minorticks_off()
ax.set_xlabel("Precio real"); ax.set_ylabel("Precio predicho")
ax.set_title("Predicción vs realidad (casas no vistas por el modelo)")
ax.legend(loc="upper left")
save(fig, "12_pred_vs_real")

# 4.2 Error por rango de precio
rng_lbl = pd.qcut(actual, 5, labels=["Muy baratas", "Baratas", "Medias", "Caras", "Muy caras"])
g = pd.DataFrame({"abs": pct.abs(), "bias": pct, "r": rng_lbl}).groupby("r", observed=True).mean() * 100
fig, ax = plt.subplots(figsize=(9, 3.8))
xx = np.arange(5)
ax.bar(xx - 0.2, g["abs"], 0.4, color=BLUE, label="Error medio (%)")
ax.bar(xx + 0.2, g["bias"], 0.4, color=ORANGE, label="Sesgo: + sobreestima / − subestima (%)")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xx, g.index); ax.set_ylabel("%")
ax.set_title("¿Dónde se equivoca más el modelo?")
ax.legend(fontsize=9)
save(fig, "13_error_rango")

# 4.3 Error por tipo de venta
sale = pd.DataFrame({"abs": pct.abs() * 100, "c": X["SaleCondition"].values}).groupby("c")["abs"].agg(["mean", "size"])
names = {"Normal": "Normal", "Partial": "Casa nueva", "AdjLand": "Terreno contiguo", "Abnorml": "Anormal (embargo…)",
         "Alloca": "Asignación", "Family": "Entre familiares"}
sale = sale.sort_values("mean")
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh([f"{names[i]} (n={n})" for i, n in zip(sale.index, sale["size"])], sale["mean"],
        color=[GREEN if i == "Normal" else RED if m > 12 else BLUE for i, m in zip(sale.index, sale["mean"])])
for i, v in enumerate(sale["mean"]):
    ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Error medio (%)")
ax.set_title("Error según el tipo de venta")
save(fig, "14_error_tipo_venta")

# ---------------------------------------------------------------- 5. Interpretabilidad
print("Interpretabilidad (SHAP)")
model = make_final_model().fit(X, y)
expl = group_by_original_feature(ensemble_shap(model, X), model, X)
pct_effect = np.exp(expl.values) - 1

# 5.1 Importancia global
imp = pd.Series(np.abs(expl.values).mean(0), index=expl.feature_names).sort_values().tail(12)
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(imp.index, (np.exp(imp.values) - 1) * 100, color=BLUE, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Cuánto mueve el precio, en promedio (%)")
ax.set_title("¿Qué variables pesan más en la predicción? (SHAP)")
save(fig, "15_shap_importancia")

# 5.2 Efectos de calidad, superficie y antigüedad
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, feat, lbl in zip(axes, ["OverallQual", "TotalSF", "HouseAge"],
                         ["Calidad (1–10)", "Superficie total (pies²)", "Antigüedad al vender (años)"]):
    i = expl.feature_names.index(feat)
    ax.scatter(expl.display_data[:, i].astype(float), pct_effect[:, i] * 100, s=7, alpha=0.4, color=BLUE)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel(lbl); ax.set_ylabel("Efecto en el precio (%)")
    ax.set_title(feat)
save(fig, "16_shap_efectos")

# 5.3 Explicación de una casa concreta (en %)
in_pred = np.expm1(model.predict(X))
idx = int(np.where(X["Id"].values == 393)[0][0])
contrib = pd.Series(expl.values[idx], index=expl.feature_names)
main = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(7)
rest = contrib.drop(main.index).sum()
main = pd.concat([main, pd.Series({"Resto (83 variables)": rest})])[::-1]
labels = []
for f in main.index:
    if f.startswith("Resto"):
        labels.append(f)
    else:
        labels.append(f"{f} = {expl.display_data[idx, expl.feature_names.index(f)]}")
fig, ax = plt.subplots(figsize=(8.5, 4.2))
vals = (np.exp(main.values) - 1) * 100
ax.barh(labels, vals, color=[RED if v < 0 else GREEN for v in vals])
for i, v in enumerate(vals):
    ax.text(v + (0.4 if v >= 0 else -0.4), i, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(vals.min() - 6, vals.max() + 6)
ax.set_xlabel("Efecto sobre el precio (%)")
ax.set_title(f"Casa 393: punto de partida {usd(np.expm1(expl.base_values[idx]))} → predicción {usd(in_pred[idx])} "
             f"(real {usd(np.expm1(y.iloc[idx]))})", fontsize=11)
save(fig, "17_shap_casa")

print(f"\nFiguras guardadas en {OUT}")
