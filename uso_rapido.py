"""
TFG M&A — Uso rápido de los modelos
====================================
Ejemplo end-to-end: carga los modelos, evalúa el caso Amadeus del TFG,
valora un target y muestra el top de recomendados.

    pip install catboost pandas numpy
    python uso_rapido.py
"""

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

# Para que `valoracion` y `recomendador` se importen correctamente
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# 1) CARGA DE MODELOS
# ============================================================
def cargar_modelos():
    lp = CatBoostClassifier(); lp.load_model("modelo_LP_definitivo.cbm")
    cp = CatBoostClassifier(); cp.load_model("modelo_CP_definitivo.cbm")
    lp_meta = json.load(open("lp_def_meta.json"))
    cp_meta = json.load(open("cp_def_meta.json"))
    return lp, cp, lp_meta, cp_meta


# ============================================================
# 2) CONSTRUCCIÓN DEL DEAL PARA EL MODELO LP
# ============================================================
# Medianas del dataset modelar (precalculadas, para rellenar lo que no se controla).
# Si dispones de TFG_DATASET_MODELAR.xlsx puedes recalcularlas con:
#   df = pd.read_excel("TFG_DATASET_MODELAR.xlsx")
#   med = {c: pd.to_numeric(df[c], errors="coerce").median() for c in df.columns}
MEDIANAS_DEFAULT = {
    "US_Inflation_Pct": 2.5, "VIX_Avg": 18.0, "Credit_Spread_BAA_AAA": 0.95,
    "Yield_Curve_Inverted": 0, "SP500_Return_Pct": 12.0, "GDP_Growth": 0.025,
    "Date": 2020, "Distance_KM": 5000, "Deal_Value": 500e6,
}


def construir_fila_deal(acquirer, target, lp_meta, medianas=None):
    """Construye una fila lista para el modelo LP a partir de adquirente + target."""
    medianas = medianas or MEDIANAS_DEFAULT
    feats, cat = lp_meta["features"], lp_meta["cat"]

    row = {f: ("Unknown" if f in cat else medianas.get(f, 0)) for f in feats}

    # Adquirente
    row["Market_Cap"]      = acquirer.get("Market_Cap", 20e9)
    row["Revenue"]         = acquirer.get("Revenue", 5e9)
    row["Beta"]            = acquirer.get("Beta", 1.0)
    row["Prior_Deals"]     = acquirer.get("Prior_Deals", 5)
    row["Serial_Acquiror"] = acquirer.get("Serial_Acquiror", 1)
    row["acquiror_region"] = acquirer.get("region", "Europe")
    row["Sector_YF"]       = acquirer.get("Sector", "Technology")
    if "Acquiror_In_EU" in row:
        row["Acquiror_In_EU"] = acquirer.get("in_eu", 1)

    # Target (geografía)
    row["target_region"] = target.get("region", "Europe")
    if "Geographic_Scope" in row:
        row["Geographic_Scope"] = "Intra-regional" if target.get("region") == acquirer.get("region") else "Inter-regional"
    if "Both_Developed" in row:
        row["Both_Developed"] = int(target.get("developed", True))

    # Interacciones
    mc, rev = row["Market_Cap"], row["Revenue"]
    infl, vix = medianas["US_Inflation_Pct"], medianas["VIX_Avg"]
    cs, yc = medianas["Credit_Spread_BAA_AAA"], medianas["Yield_Curve_Inverted"]
    row["Small_x_HighInflation"] = int(mc <= 5e9 and infl > 4)
    row["Large_x_Serial"]        = int(mc > 5e9 and row["Serial_Acquiror"] == 1)
    row["Expert_x_Crisis"]       = 0
    row["MarketCap_Log"]         = float(np.log1p(max(mc, 0)))
    row["Revenue_Log"]           = float(np.log1p(max(rev, 0)))
    row["Beta_Dist_Optimal"]     = abs(row["Beta"] - 1.0)
    row["Large_x_HighVIX"]       = int(mc > 5e9 and vix > 15.5)
    row["Serial_x_HighVIX"]      = int(row["Serial_Acquiror"] == 1 and vix > 15.5)
    row["Small_x_NoExp"]         = int(mc <= 5e9 and row["Prior_Deals"] == 0)
    row["Beta_Optimal"]          = int(0.8 <= row["Beta"] <= 1.2)
    row["CreditSpread_x_YieldCurve"]  = cs * yc
    row["Inverted_x_YieldCurve"]      = yc
    row["Inflation_div_CreditSpread"] = infl / (cs + 0.01)
    row["INT_Size_x_VIX"]        = row["MarketCap_Log"] * vix
    row["INT_Size_x_Exp"]        = row["MarketCap_Log"] * row["Prior_Deals"]
    row["Beta_x_VIX"]            = row["Beta"] * vix

    df = pd.DataFrame([row])[feats]
    for c in cat:
        if c in df: df[c] = df[c].fillna("Unknown").astype(str)
    return df


# ============================================================
# 3) DEMO END-TO-END
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TFG M&A — Demo de uso de los modelos")
    print("=" * 70)

    lp, cp, lp_meta, cp_meta = cargar_modelos()
    print(f"\n[1] Modelos cargados — LP: {lp.tree_count_} árboles | CP: {cp.tree_count_} árboles")

    # Caso Amadeus del TFG
    amadeus = {"Market_Cap": 21480e6, "Revenue": 6520e6, "Beta": 0.58,
               "Prior_Deals": 9, "Serial_Acquiror": 1, "region": "Europe",
               "Sector": "Technology", "in_eu": 1, "developed": True}

    print(f"\n[2] Probabilidad LP por geografía del target (adquirente Amadeus):")
    for region in ["Europe", "North America", "Asia", "South America"]:
        X = construir_fila_deal(amadeus, {"region": region, "developed": True}, lp_meta)
        p = lp.predict_proba(X)[0, 1]
        print(f"      Target en {region:18}: {p:.1%}")
    print(f"      (TFG reporta base 64,1% para Amadeus)")

    # Valoración + recomendador
    print(f"\n[3] Valoración (caso NICE simulado):")
    from valoracion import valorar
    nice = {"Market_Cap": 12e9, "Revenue": 2.5e9, "Beta": 1.0, "EBITDA": 650e6,
            "Free_Cash_Flow": 500e6, "Total_Debt": 200e6, "Total_Cash": 1e9,
            "Dividend_Yield": 0, "Revenue_Growth": 0.10, "Sector": "Technology"}
    r = valorar(nice)
    print(f"      DCF: {r['dcf_equity']/1e9:.1f}B | EV/EBITDA: {r['multiplos_equity']/1e9:.1f}B")
    print(f"      Standalone: {r['standalone']/1e9:.1f}B → precio: {r['precio_recomendado']/1e9:.1f}B")
    print(f"      (TFG NICE: ~17,5B)")

    print(f"\n[4] Recomendador (top 5 Technology, adquirente Amadeus):")
    from recomendador import recomendar, cargar_universo, PERFIL_AMADEUS, semaforo
    uni = cargar_universo()
    df_mod = None  # las medianas ya van en MEDIANAS_DEFAULT
    rank = recomendar(uni, modelo=lp, feats=lp_meta["features"], cat=lp_meta["cat"],
                      medianas=MEDIANAS_DEFAULT, sectores_afines=["Technology"], top_n=5)
    rank["Sem"] = rank["Score_comb"].apply(semaforo)
    print(rank[["Rank", "Target", "MCap_B", "P_modelo", "Score_comb", "Sem"]].to_string(index=False))

    print(f"\n[OK] Demo completa. Tienes el sistema listo para usar.")
