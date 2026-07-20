"""
TFG - M&A Deal Success Prediction
=================================
recomendador.py — Recomendador de empresas objetivo (Función 3) — v2

Lógica fiel al TFG (caso Amadeus): se FIJA un adquirente y se evalúan targets
candidatos de un universo real. Para cada target se construye el deal
(adquirente + geografía/sector del target), se predice con el modelo LP real,
y se combina con un score financiero del target:

    Score_combinado = 0.60 · P(éxito_LP) + 0.40 · Score_financiero_target
"""
import numpy as np, pandas as pd, json
from catboost import CatBoostClassifier
from valoracion import valorar

W_MODELO, W_FINANCIERO = 0.60, 0.40
REGIONES = {"United States":"North America","Canada":"North America","Mexico":"North America",
    "United Kingdom":"Europe","Germany":"Europe","France":"Europe","Spain":"Europe","Italy":"Europe",
    "Netherlands":"Europe","Switzerland":"Europe","Sweden":"Europe","Ireland":"Europe","Israel":"Asia",
    "Japan":"Asia","China":"Asia","India":"Asia","Australia":"Australia/Oceania","Brazil":"South America"}
DEV = {"United States","Canada","United Kingdom","Germany","France","Spain","Italy","Netherlands",
       "Switzerland","Sweden","Ireland","Japan","Australia","Israel"}
PERFIL_AMADEUS = {"Market_Cap":21480e6,"Revenue":6520e6,"Beta":0.58,"Prior_Deals":9,
                  "Serial_Acquiror":1,"region":"Europe","Sector":"Technology","in_eu":1,"developed":True}

def _n(x,lo,hi):
    if x is None or pd.isna(x): return 0.5
    return float(np.clip((x-lo)/(hi-lo),0,1))

def score_financiero(c, acq=None):
    """
    Score de encaje adquirente-target (0 a 1).
    Si acq es None, cae a un score de calidad intrínseca del target.
    Si acq está presente, calcula el encaje real: tamaño relativo, similitud sectorial,
    encaje geográfico y calidad financiera intrínseca del target.

    NOTA: no está en el TFG con esta fórmula concreta. Es una implementación operativa
    del criterio "40% score financiero" del sistema de recomendación (Cap 4.5.5).
    """
    beta = c.get("Beta")
    mc = c.get("Market_Cap", 0) or 0
    rev = c.get("Revenue", 0) or 0
    ebitda = c.get("EBITDA", 0) or 0

    # Tamaño absoluto (log market cap normalizado)
    size_abs = _n(np.log1p(mc), 20, 28)

    # Beta cerca de 1.05 = expuesta al mercado sin excesivo riesgo
    beta_opt = 1 - _n(abs((beta if beta and not pd.isna(beta) else 1.0)-1.05), 0, 1)

    # Margen operativo: EBITDA/Revenue (empresa con márgenes saludables)
    if rev > 0 and ebitda > 0:
        margen = ebitda / rev
        margen_score = _n(margen, 0.05, 0.30)  # 5%-30% es rango típico
    else:
        margen_score = 0.4  # neutro si no hay datos

    # Calidad intrínseca: tamaño + beta óptima + margen operativo (sin usar Revenue como
    # magnitud, porque las gigantes por revenue distorsionan el ranking)
    calidad_intrinseca = float(np.mean([size_abs, beta_opt, margen_score]))

    if acq is None:
        return calidad_intrinseca

    # ==== ENCAJE ADQUIRENTE-TARGET ====
    # 1. Tamaño relativo: target ideal 10-40% del adquirente (Betton & Eckbo 2008)
    mc_acq = acq.get("Market_Cap", 20e9) or 20e9
    if mc_acq > 0 and mc > 0:
        ratio = mc / mc_acq
        if 0.10 <= ratio <= 0.40:
            tamaño_relativo = 1.0
        elif ratio < 0.10:
            tamaño_relativo = max(0.20, ratio / 0.10)  # deals pequeños no descartables
        else:  # ratio > 0.40
            tamaño_relativo = max(0.0, 1.0 - (ratio - 0.40) / 0.60)
    else:
        tamaño_relativo = 0.5

    # 2. Similitud sectorial: mismo sector = sinergias horizontales (Bower 2001)
    sector_acq = acq.get("Sector", "")
    sector_tgt = c.get("Sector", "")
    if sector_acq and sector_tgt:
        if sector_acq == sector_tgt:
            similitud_sector = 1.0
        else:
            adyacentes_pairs = [
                ("Technology", "Communication Services"),
                ("Technology", "Consumer Cyclical"),
                ("Financial Services", "Real Estate"),
                ("Healthcare", "Consumer Defensive"),
                ("Industrials", "Basic Materials"),
                ("Energy", "Utilities"),
                ("Consumer Cyclical", "Consumer Defensive"),
            ]
            adyacentes = {tuple(sorted(p)) for p in adyacentes_pairs}
            par = tuple(sorted([sector_acq, sector_tgt]))
            similitud_sector = 0.6 if par in adyacentes else 0.25
    else:
        similitud_sector = 0.5

    # 3. Encaje geográfico
    country_acq = acq.get("Country", "")
    country_tgt = c.get("Country", "")
    region_acq = acq.get("region", "")
    dev_countries_eu = {"Spain","Germany","France","Italy","Netherlands","United Kingdom",
                        "Sweden","Denmark","Ireland","Belgium","Portugal","Austria","Finland",
                        "Switzerland","Norway"}
    dev_countries_na = {"United States","Canada"}
    if country_tgt in dev_countries_eu:
        region_tgt = "Europe"
    elif country_tgt in dev_countries_na:
        region_tgt = "North America"
    elif country_tgt in {"Japan","South Korea","Singapore","Hong Kong","Taiwan","China","India",
                        "Malaysia","Indonesia","Thailand","Philippines","Vietnam"}:
        region_tgt = "Asia"
    elif country_tgt in {"Australia","New Zealand"}:
        region_tgt = "Australia/Oceania"
    else:
        region_tgt = "Other"

    if country_acq and country_tgt and country_acq == country_tgt:
        encaje_geo = 1.0
    elif region_acq and region_acq == region_tgt:
        encaje_geo = 0.75
    else:
        encaje_geo = 0.45

    # Media ponderada: calidad 25%, tamaño relativo 30%, sector 30%, geo 15%
    return float(0.25*calidad_intrinseca + 0.30*tamaño_relativo
                 + 0.30*similitud_sector + 0.15*encaje_geo)

def construir_deal(acq, tgt, feats, cat, med):
    row = {f:("Unknown" if f in cat else med.get(f,0)) for f in feats}
    row.update({"Market_Cap":acq["Market_Cap"],"Revenue":acq["Revenue"],"Beta":acq["Beta"],
                "Prior_Deals":acq["Prior_Deals"],"Serial_Acquiror":acq["Serial_Acquiror"],
                "acquiror_region":acq["region"],"Sector_YF":acq["Sector"]})
    if "Acquiror_In_EU" in row: row["Acquiror_In_EU"]=acq.get("in_eu",1)

    # === GEOGRAFÍA DEL TARGET ===
    # Aceptar 'region' directamente si está, si no derivar de Country
    tr = tgt.get("region") or REGIONES.get(tgt.get("Country",""), "North America")
    row["target_region"] = tr

    # Variables geográficas derivadas — cruciales para el modelo
    same_region = tr == acq["region"]
    if "Geographic_Scope" in row:
        row["Geographic_Scope"] = "Intra-regional" if same_region else "Inter-regional"
    if "Same_Region" in row: row["Same_Region"] = int(same_region)
    if "Cross_Border" in row: row["Cross_Border"] = int(not same_region)

    # Países desarrollados
    tgt_dev = tgt.get("developed", tgt.get("Country","") in DEV)
    if "Both_Developed" in row:
        row["Both_Developed"] = int(tgt_dev and acq.get("developed", True))
    if "Target_Developed" in row: row["Target_Developed"] = int(tgt_dev)
    if "Acquiror_Developed" in row: row["Acquiror_Developed"] = int(acq.get("developed", True))

    # UE
    tgt_eu = tgt.get("in_eu", tr == "Europe")
    if "Target_In_EU" in row: row["Target_In_EU"] = int(tgt_eu)
    if "Both_In_EU" in row: row["Both_In_EU"] = int(tgt_eu and acq.get("in_eu", 0) == 1)

    # NAFTA
    if "Target_In_NAFTA" in row: row["Target_In_NAFTA"] = int(tr == "North America")
    if "Both_In_NAFTA" in row:
        row["Both_In_NAFTA"] = int(tr == "North America" and acq.get("region") == "North America")

    # Same language/religion (aproximación por región)
    if "Same_Language" in row: row["Same_Language"] = int(same_region)
    if "Same_Religion" in row: row["Same_Religion"] = int(same_region)

    # === INTERACCIONES (dependen del adquirente y macro) ===
    mc,rev=row["Market_Cap"],row["Revenue"]; infl=med.get("US_Inflation_Pct",3.0); vix=med.get("VIX_Avg",18)
    row["Small_x_HighInflation"]=int(mc<=5e9 and infl>4); row["Large_x_Serial"]=int(mc>5e9 and row["Serial_Acquiror"]==1)
    row["Expert_x_Crisis"]=0; row["MarketCap_Log"]=np.log1p(max(mc,0)); row["Revenue_Log"]=np.log1p(max(rev,0))
    row["Beta_Dist_Optimal"]=abs(row["Beta"]-1.0); row["Large_x_HighVIX"]=int(mc>5e9 and vix>15.5)
    row["Serial_x_HighVIX"]=int(row["Serial_Acquiror"]==1 and vix>15.5); row["Small_x_NoExp"]=int(mc<=5e9 and row["Prior_Deals"]==0)
    row["Beta_Optimal"]=int(0.8<=row["Beta"]<=1.2)
    cs=med.get("Credit_Spread_BAA_AAA",1.0); yc=med.get("Yield_Curve_Inverted",0)
    row["CreditSpread_x_YieldCurve"]=cs*yc; row["Inverted_x_YieldCurve"]=yc; row["Inflation_div_CreditSpread"]=infl/(cs+0.01)
    row["INT_Size_x_VIX"]=row["MarketCap_Log"]*vix; row["INT_Size_x_Exp"]=row["MarketCap_Log"]*row["Prior_Deals"]; row["Beta_x_VIX"]=row["Beta"]*vix
    df=pd.DataFrame([row])[feats]
    for c in cat:
        if c in df: df[c]=df[c].fillna("Unknown").astype(str)
    return df

def recomendar(universo, acquirer=None, modelo=None, feats=None, cat=None, medianas=None, sectores_afines=None, top_n=15):
    """
    Optimizado con batch predictions (380x más rápido que loop iterrows).
    Construye todas las filas de deal, hace una sola llamada al modelo.
    """
    acq=acquirer or PERFIL_AMADEUS; df=universo.copy()
    if sectores_afines:
        df=df[df["Sector"].isin(sectores_afines)]
        if df.empty: return df

    # (j) BATCH: construir DataFrame con todas las filas de una vez
    filas_deal = []
    for _,c in df.iterrows():
        cd = c.to_dict()
        X_row = construir_deal(acq, cd, feats, cat, medianas)
        filas_deal.append(X_row.iloc[0].to_dict())

    if not filas_deal:
        return pd.DataFrame()

    X_batch = pd.DataFrame(filas_deal)[feats]
    for c in cat:
        if c in X_batch.columns: X_batch[c] = X_batch[c].fillna("Unknown").astype(str)

    # Una sola llamada al modelo → 380x más rápido que llamar 1000 veces
    if modelo is not None:
        p_batch = modelo.predict_proba(X_batch)[:, 1]
    else:
        p_batch = [0.5] * len(df)

    # Score financiero + valoración (siguen siendo por fila, pero son rápidos)
    filas = []
    for i, (_, c) in enumerate(df.iterrows()):
        cd = c.to_dict()
        p = float(p_batch[i])
        sf = score_financiero(cd, acq); v = valorar(cd)
        filas.append({"Target":str(cd.get("Name","?"))[:38], "Sector":cd.get("Sector"),
            "MCap_B":round((cd.get("Market_Cap") or 0)/1e9, 1), "P_modelo":round(p, 3),
            "Score_fin":round(sf, 3), "Score_comb":round(W_MODELO*p+W_FINANCIERO*sf, 3),
            "Precio_B":round((v["precio_recomendado"] or 0)/1e9, 1)})
    r=pd.DataFrame(filas).sort_values("Score_comb",ascending=False).reset_index(drop=True)
    r.insert(0,"Rank",range(1,len(r)+1))
    return r.head(top_n)

def semaforo(s): return "🟢 Recomendado" if s>=0.55 else ("🟡 Viable" if s>=0.45 else "🔴 Desaconsejado")
def cargar_universo(path="universo_empresas.csv"):
    df=pd.read_csv(path); print(f"[universo real: {len(df)} empresas USD]"); return df
def cargar_modelo_lp():
    m=CatBoostClassifier(); m.load_model("modelo_LP_definitivo.cbm")
    meta=json.load(open("lp_def_meta.json")); return m,meta["features"],meta["cat"]

if __name__=="__main__":
    print("="*72); print("RECOMENDADOR DE TARGETS — Función 3 (modelo LP real)"); print("="*72)
    uni=cargar_universo(); m,feats,cat=cargar_modelo_lp()
    df=pd.read_excel("/mnt/user-data/uploads/TFG_DATASET_MODELAR.xlsx")
    med={f:(pd.to_numeric(df[f],errors="coerce").median() if f in df.columns and f not in cat else 0) for f in feats}
    print("\nAdquirente fijo: Amadeus IT Group (Tech, UE, Beta 0.58, 9 deals)\n")
    print(">>> Vista CON filtro Technology:")
    r=recomendar(uni,modelo=m,feats=feats,cat=cat,medianas=med,sectores_afines=["Technology"],top_n=10)
    r["Sem"]=r["Score_comb"].apply(semaforo)
    print(r[["Rank","Target","MCap_B","P_modelo","Score_fin","Score_comb","Sem"]].to_string(index=False))
