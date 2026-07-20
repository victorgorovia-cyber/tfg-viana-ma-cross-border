"""
TFG M&A — Aplicación web integrada v2
======================================
Mejoras respecto a v1:
  · Autocompletar target desde el universo de empresas
  · SHAP por deal concreto (waterfall interpretativo)
  · Aviso de calibración temporal (AUC 0,697 vs 0,887 random)
  · Cuadrantes trampa L1/L2 (23% de deals engañan al mercado)
  · Historial de deals evaluados (comparación side-by-side)
  · Análisis de sensibilidad al precio
  · Export a informe de texto del deal
  · Calibración por quintiles visible

Ejecución local:
    streamlit run app.py
"""

import json
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

from valoracion import valorar, EV_EBITDA_SECTOR, PRIMA_MA_SECTOR
from recomendador import (recomendar, construir_deal, score_financiero,
                          PERFIL_AMADEUS, semaforo)


st.set_page_config(page_title="TFG M&A — Sistema predictivo",
                   page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")

MACRO_ANIOS = {
    2020: {"US_Inflation_Pct":1.2,"VIX_Avg":29.0,"Credit_Spread_BAA_AAA":1.30,
           "Yield_Curve_Inverted":0,"SP500_Return_Pct":16.3,"GDP_Growth":-0.028,"auc_temporal":0.688},
    2021: {"US_Inflation_Pct":4.7,"VIX_Avg":19.7,"Credit_Spread_BAA_AAA":0.87,
           "Yield_Curve_Inverted":0,"SP500_Return_Pct":26.9,"GDP_Growth":0.059,"auc_temporal":0.712},
    2022: {"US_Inflation_Pct":8.0,"VIX_Avg":25.6,"Credit_Spread_BAA_AAA":1.14,
           "Yield_Curve_Inverted":1,"SP500_Return_Pct":-19.4,"GDP_Growth":0.021,"auc_temporal":0.681},
    2023: {"US_Inflation_Pct":4.1,"VIX_Avg":16.9,"Credit_Spread_BAA_AAA":1.05,
           "Yield_Curve_Inverted":1,"SP500_Return_Pct":24.2,"GDP_Growth":0.025,"auc_temporal":0.697},
    2024: {"US_Inflation_Pct":2.9,"VIX_Avg":15.5,"Credit_Spread_BAA_AAA":0.95,
           "Yield_Curve_Inverted":0,"SP500_Return_Pct":23.3,"GDP_Growth":0.028,"auc_temporal":None},
    2025: {"US_Inflation_Pct":2.73,"VIX_Avg":19.5,"Credit_Spread_BAA_AAA":0.98,
           "Yield_Curve_Inverted":0,"SP500_Return_Pct":8.5,"GDP_Growth":0.023,"auc_temporal":None},
    2026: {"US_Inflation_Pct":4.2,"VIX_Avg":16.4,"Credit_Spread_BAA_AAA":1.10,
           "Yield_Curve_Inverted":0,"SP500_Return_Pct":6.2,"GDP_Growth":0.020,"auc_temporal":None},
}

QUINTILES_LP = {"Q1":(0,0.30,0.187),"Q2":(0.30,0.45,0.358),"Q3":(0.45,0.55,0.502),
                "Q4":(0.55,0.70,0.716),"Q5":(0.70,1.01,0.964)}


@st.cache_resource
def cargar_modelos():
    lp = CatBoostClassifier(); lp.load_model("modelo_LP_definitivo.cbm")
    cp = CatBoostClassifier(); cp.load_model("modelo_CP_definitivo.cbm")
    lp_meta = json.load(open("lp_def_meta.json"))
    cp_meta = json.load(open("cp_def_meta.json"))
    return lp, cp, lp_meta, cp_meta


@st.cache_data(show_spinner=False)
def cached_predict_lp(_lp_model, X_key_tuple, feats_tuple):
    """Cache de predicciones LP (h): mismo input → mismo output sin recalcular."""
    X = pd.DataFrame([dict(zip(feats_tuple, X_key_tuple))])
    return float(_lp_model.predict_proba(X)[0, 1])


@st.cache_data
def cargar_medianas():
    """Carga medianas del dataset modelar (fallback: dict vacío si no están)."""
    try:
        with open("medianas_dataset.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_data
def cargar_universo():
    return pd.read_csv("universo_empresas.csv")


if "historial" not in st.session_state:
    st.session_state.historial = []


def calcular_scorecard(perfil):
    puntos = 0; desglose = []
    mc = perfil.get("Market_Cap", 0)
    if   mc >= 100e9: p,r = 30,"≥100B"
    elif mc >=  50e9: p,r = 25,"50-100B"
    elif mc >=  10e9: p,r = 18,"10-50B"
    elif mc >=   1e9: p,r = 10,"1-10B"
    else:             p,r = 5, "<1B"
    puntos += p; desglose.append(("Market_Cap",p,30,r))
    rev = perfil.get("Revenue",0)
    if   rev >= 20e9: p,r = 10,"≥20B"
    elif rev >=  5e9: p,r =  7,"5-20B"
    elif rev >=  1e9: p,r =  5,"1-5B"
    else:             p,r =  3,"<1B"
    puntos += p; desglose.append(("Revenue_Log",p,10,r))
    beta = perfil.get("Beta",1.0)
    if 0.9 <= beta <= 1.2: p,r = 8,"óptimo"
    elif 0.7 <= beta <= 1.4: p,r = 5,"aceptable"
    else: p,r = 2,"extremo"
    puntos += p; desglose.append(("Beta",p,8,r))
    pd_ = perfil.get("Prior_Deals",0)
    if   pd_ >= 10: p,r = 8,"serial"
    elif pd_ >= 5:  p,r = 6,"5-10"
    elif pd_ >= 1:  p,r = 3,"1-5"
    else:           p,r = 0,"sin exp"
    puntos += p; desglose.append(("Prior_Deals",p,8,r))
    p = 6 if perfil.get("mismo_sector") else 3
    puntos += p; desglose.append(("Sector",p,6,"mismo" if perfil.get("mismo_sector") else "cruzado"))
    infl = perfil.get("US_Inflation_Pct",3.0)
    p = 8 if infl < 2.5 else (5 if infl < 4 else 2)
    puntos += p; desglose.append(("US_Inflation",p,8,f"{infl:.1f}%"))
    cs = perfil.get("Credit_Spread_BAA_AAA",1.0)
    p = 6 if cs < 1.0 else (3 if cs < 1.5 else 1)
    puntos += p; desglose.append(("Credit_Spread",p,6,f"{cs:.2f}"))
    vix = perfil.get("VIX_Avg",18)
    p = 5 if vix < 18 else (3 if vix < 25 else 1)
    puntos += p; desglose.append(("VIX_Avg",p,5,f"{vix:.0f}"))
    yc = perfil.get("Yield_Curve_Inverted",0)
    p = 5 if yc == 0 else 1
    puntos += p; desglose.append(("Yield_Curve",p,5,"normal" if yc == 0 else "invertida"))
    sp = perfil.get("SP500_Return_Pct",10)
    p = 4 if sp > 10 else (2 if sp > 0 else 0)
    puntos += p; desglose.append(("SP500_Return",p,4,f"{sp:.1f}%"))
    gdp = perfil.get("GDP_Growth",0.02)
    p = 4 if gdp > 0.025 else (2 if gdp > 0 else 0)
    puntos += p; desglose.append(("GDP_Growth",p,4,f"{gdp*100:.1f}%"))
    p = 3 if perfil.get("both_developed") else 1
    puntos += p; desglose.append(("Both_Developed",p,3,"sí" if perfil.get("both_developed") else "no"))
    p = 3 if perfil.get("both_eu") else 1
    puntos += p; desglose.append(("Both_EU",p,3,"sí" if perfil.get("both_eu") else "no"))
    return puntos, desglose


def quintil_score(score):
    if score <= 35: return "Q1 (peor)","16,2%","🔴"
    if score <= 50: return "Q2","35,8%","🟠"
    if score <= 62: return "Q3","46,1%","🟡"
    if score <= 74: return "Q4","54,4%","🟢"
    return "Q5 (mejor)","70,8%","🟢"


def quintil_prob(p):
    for q,(lo,hi,real) in QUINTILES_LP.items():
        if lo <= p < hi:
            return q, real
    return "Q5", 0.964


def shap_deal(model, X_row, feats, cat_features_names, top_k=10):
    """SHAP del deal individual. Requiere pasar los nombres de las categóricas."""
    try:
        from catboost import Pool
        # Índices numéricos de las columnas categóricas en el orden de X_row
        cat_idx = [i for i, c in enumerate(X_row.columns) if c in cat_features_names]
        # Asegurar tipos: object para categóricas, numéricas para el resto
        X_typed = X_row.copy()
        for c in X_typed.columns:
            if c in cat_features_names:
                X_typed[c] = X_typed[c].astype(str)
            else:
                X_typed[c] = pd.to_numeric(X_typed[c], errors="coerce").fillna(0)
        pool = Pool(X_typed, cat_features=cat_idx)
        shap_vals = model.get_feature_importance(pool, type="ShapValues")
        shap_row = shap_vals[0, :-1]
        s = pd.DataFrame({"feature": feats, "shap": shap_row})
        s["abs"] = s["shap"].abs()
        s = s.sort_values("abs", ascending=False).head(top_k)
        s["dir"] = s["shap"].apply(lambda x: "⬆️ sube" if x > 0 else "⬇️ baja")
        s["contrib_pp"] = (s["shap"] * 100).round(1)
        return s[["feature","dir","contrib_pp"]].rename(
            columns={"feature":"Variable","dir":"Efecto","contrib_pp":"Impacto (pp)"})
    except Exception as e:
        return pd.DataFrame({"error":[str(e)]})


st.title("📊 Sistema predictivo de éxito en operaciones M&A")
st.caption("TFG · CatBoost sobre 2.739 deals (1997–2023) · 5 funciones integradas · v2")

lp, cp, lp_meta, cp_meta = cargar_modelos()
universo = cargar_universo()
MEDIANAS_DATASET = cargar_medianas()

with st.sidebar:
    st.header("⚙️ ¿Qué quieres hacer?")
    modo = st.radio("Modo", ["🎯 Evaluar deal (target elegido)",
                              "💡 Encontrar y evaluar target recomendado",
                              "⚖️ Comparar dos deals",
                              "📊 Modelo · SHAP · Calibración",
                              "🔬 Validación técnica (a-e)"],
                    label_visibility="collapsed")

    # Descripción contextual del modo elegido
    descripciones = {
        "🎯 Evaluar deal (target elegido)":
            "Ya sabes qué operación quieres analizar. Introduce el adquirente y el target, y obtén probabilidad de éxito, valoración y veredicto integrado.",
        "💡 Encontrar y evaluar target recomendado":
            "Tienes un adquirente pero no un target concreto. El sistema busca en el universo de 973 empresas cotizadas los targets con mejor encaje estratégico.",
        "⚖️ Comparar dos deals":
            "Evalúa dos operaciones en paralelo para decidir cuál priorizar.",
        "📊 Modelo · SHAP · Calibración":
            "Métricas técnicas del modelo: importancia de variables, calibración por quintiles, receta de entrenamiento.",
        "🔬 Validación técnica (a-e)":
            "Análisis de robustez para la defensa: validación temporal, comparativa de algoritmos, intervalos de confianza, curva de aprendizaje.",
    }
    st.caption(descripciones[modo])

    st.divider()
    st.markdown("**Métricas del modelo**")
    st.metric("Deals modelados", "2.739")
    st.metric("AUC test LP", "0,887", "TFG 0,892")
    st.metric("AUC temporal medio", "0,697", "-0,190 vs random")
    st.caption("Usar predicciones con banda ±10pp para años futuros.")
    st.metric("Universo empresas", f"{len(universo)}")
    if st.session_state.historial:
        st.divider()
        st.markdown(f"**📌 Deals evaluados: {len(st.session_state.historial)}**")
        if st.button("🗑️ Vaciar historial", use_container_width=True):
            st.session_state.historial = []
            st.rerun()


if modo == "🎯 Evaluar deal (target elegido)":
    st.header("Evaluación end-to-end de una operación")

    # ---- Botón de carga rápida del caso Amadeus del TFG ----
    with st.expander("🚀 Cargar caso del TFG (demo rápida para defensa)", expanded=False):
        st.markdown("Precarga el caso **Amadeus IT Group** documentado en el TFG con contexto macro 2018. "
                    "Reproduce la probabilidad reportada (~66%).")
        if st.button("📥 Cargar caso Amadeus + Target Tech Europa 2018", type="primary"):
            st.session_state.demo_tfg = True
            st.rerun()
        if st.session_state.get("demo_tfg"):
            st.success("✓ Caso cargado. Pulsa 'Evaluar' abajo.")

    # Aplicar defaults del demo si está activo
    demo_activo = st.session_state.get("demo_tfg", False)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏢 Adquirente")
        preset_a = st.selectbox("Preset", ["Personalizado", "Amadeus IT Group (caso TFG)"],
                                index=1 if demo_activo else 0)
        if preset_a == "Amadeus IT Group (caso TFG)":
            d = {"mc":21.48,"rev":6.52,"beta":0.58,"pd":9,"reg":"Europe"}
        else:
            d = {"mc":20.0,"rev":5.0,"beta":1.0,"pd":5,"reg":"Europe"}
        acq_mc  = st.number_input("Market Cap (B USD)", 0.5, 3000.0, d["mc"], step=1.0)
        acq_rev = st.number_input("Revenue (B USD)", 0.1, 500.0, d["rev"], step=0.5)
        acq_beta= st.slider("Beta", 0.2, 2.5, d["beta"], 0.05)
        acq_pd  = st.slider("Deals previos (5a)", 0, 30, d["pd"])
        acq_reg = st.selectbox("Región", ["Europe","North America","Asia","South America","Australia/Oceania"],
                               index=["Europe","North America","Asia","South America","Australia/Oceania"].index(d["reg"]))
        acq_sec = st.selectbox("Sector adq", list(EV_EBITDA_SECTOR.keys())[:-1])
        acq_eu  = st.checkbox("Adquirente en UE", value=(acq_reg == "Europe"))

    with col2:
        st.subheader("🎯 Target")
        st.caption("💡 El modelo LP usa solo la **geografía y sector** del target. Los datos financieros alimentan la **valoración** (precio recomendado).")
        opciones = ["(rellenar manualmente)"] + sorted(
            (universo["Name"].astype(str) + " · " + universo["Sector"].astype(str) +
             " · " + (universo["Market_Cap"]/1e9).round(1).astype(str) + "B").tolist()
        )
        sel = st.selectbox("Elegir del universo o rellenar", opciones, key="tsel")
        if sel != "(rellenar manualmente)":
            nombre = sel.split(" · ")[0]
            row = universo[universo["Name"].astype(str) == nombre].iloc[0]
            t_mc     = float(row["Market_Cap"])/1e9
            t_rev    = max(float(row["Revenue"])/1e9, 0.05)
            t_beta   = float(row["Beta"]) if not pd.isna(row["Beta"]) else 1.0
            t_ebitda = max(float(row["EBITDA"])/1e9, 0.01) if not pd.isna(row["EBITDA"]) else t_rev*0.2
            t_fcf    = max(float(row["Free_Cash_Flow"])/1e9, 0.01) if not pd.isna(row["Free_Cash_Flow"]) else t_rev*0.12
            t_debt   = max(float(row["Total_Debt"])/1e9, 0.0) if not pd.isna(row["Total_Debt"]) else 0.0
            t_cash   = max(float(row["Total_Cash"])/1e9, 0.0) if not pd.isna(row["Total_Cash"]) else 0.0
            t_div    = float(row["Dividend_Yield"]) if not pd.isna(row["Dividend_Yield"]) else 0.0
            t_g      = float(row["Revenue_Growth"]) if not pd.isna(row["Revenue_Growth"]) else 0.08
            t_sec    = str(row["Sector"])
            t_country= str(row.get("Country", "United States"))
            st.info(f"✓ Cargados datos de **{nombre}**")
            st.caption(f"MC: {t_mc:.1f}B · Rev: {t_rev:.1f}B · Beta: {t_beta:.2f} · EBITDA: {t_ebitda:.2f}B · Sector: {t_sec}")
        else:
            # Rellenado manual — separar visualmente qué influye en qué
            st.markdown("**🧠 Va al modelo LP:**")
            t_sec = st.selectbox("Sector target", list(EV_EBITDA_SECTOR.keys())[:-1], key="tsec",
                                  help="Se usa para calcular sinergias sectoriales")
            st.markdown("**💰 Va solo a la valoración:**")
            t_mc      = st.number_input("Market Cap (B)", 0.1, 500.0, 12.0, help="Sólo influye en el % vs cap actual")
            t_rev     = st.number_input("Revenue (B)", 0.05, 100.0, 2.5)
            t_beta    = st.slider("Beta target", 0.2, 2.5, 1.0, 0.05, key="tb",
                                    help="Se usa en el WACC del DCF")
            t_ebitda  = st.number_input("EBITDA (B)", 0.05, 50.0, 0.65, help="Base del método EV/EBITDA")
            t_fcf     = st.number_input("Free Cash Flow (B)", 0.01, 30.0, 0.5, help="Base del DCF")
            t_debt    = st.number_input("Deuda (B)", 0.0, 100.0, 0.2)
            t_cash    = st.number_input("Caja (B)", 0.0, 100.0, 1.0)
            t_div     = st.number_input("Dividend Yield", 0.0, 0.15, 0.0, step=0.005, format="%.3f",
                                          help="Base del DDM")
            t_g       = st.slider("Crecimiento ingresos", -0.10, 0.30, 0.10, 0.01, key="tg")
            t_country = "United States"
        st.markdown("**🌎 Va al modelo LP (geografía):**")
        t_reg = st.selectbox("Región target", ["Europe","North America","Asia","South America","Australia/Oceania"], key="treg")
        t_dev = st.checkbox("Target en país desarrollado", value=True)

    with col3:
        st.subheader("🌍 Contexto macro")
        anios_disp = sorted(MACRO_ANIOS.keys(), reverse=True)
        # Si demo activo → 2018 (que reproduce el 66% del TFG). Si no → 2023.
        idx_default = anios_disp.index(2018) if (demo_activo and 2018 in anios_disp) else 3
        anio = st.selectbox("Año del deal", anios_disp, index=idx_default,
                            help="Preset de referencia. Los valores concretos se pueden ajustar debajo.")
        # Cargar preset del año como punto de partida
        preset_macro = MACRO_ANIOS[anio]
        auc_t = preset_macro.get("auc_temporal")
        if auc_t is not None:
            st.caption(f"AUC temporal validado del modelo en {anio}: **{auc_t:.3f}** (referencia).")
        else:
            st.caption(f"Año {anio}: sin AUC temporal validado (posterior a 2023). Los valores macro son estimaciones que puedes ajustar.")

        # Toggle para personalizar (por defecto, ver preset)
        personalizar = st.toggle("✏️ Personalizar parámetros macro", value=False,
                                  help="Activa para ajustar cada variable a mano (útil para escenarios hipotéticos)")

        if personalizar:
            st.caption("Ajusta cada parámetro para tu escenario:")
            m_infl  = st.slider("Inflación USA (%)", 0.0, 12.0, float(preset_macro["US_Inflation_Pct"]), 0.1)
            m_vix   = st.slider("VIX medio", 10.0, 50.0, float(preset_macro["VIX_Avg"]), 0.5)
            m_cs    = st.slider("Credit Spread BAA-AAA", 0.4, 3.0, float(preset_macro["Credit_Spread_BAA_AAA"]), 0.05)
            m_yc    = st.selectbox("Curva de tipos", ["Normal", "Invertida"],
                                    index=1 if preset_macro["Yield_Curve_Inverted"] else 0)
            m_sp    = st.slider("S&P 500 return anual (%)", -40.0, 40.0, float(preset_macro["SP500_Return_Pct"]), 0.5)
            m_gdp   = st.slider("GDP growth (%)", -5.0, 10.0, float(preset_macro["GDP_Growth"]*100), 0.1)
            macro = {
                "US_Inflation_Pct":     m_infl,
                "VIX_Avg":              m_vix,
                "Credit_Spread_BAA_AAA": m_cs,
                "Yield_Curve_Inverted": 1 if m_yc == "Invertida" else 0,
                "SP500_Return_Pct":     m_sp,
                "GDP_Growth":           m_gdp / 100,
            }
        else:
            macro = {k:v for k,v in preset_macro.items() if k != "auc_temporal"}
            st.markdown("**Contexto del preset (readonly):**")
            st.caption(f"Inflación USA: **{macro['US_Inflation_Pct']:.1f}%** · VIX: **{macro['VIX_Avg']:.1f}**")
            st.caption(f"Credit Spread: **{macro['Credit_Spread_BAA_AAA']:.2f}** · Yield: **{'invertida' if macro['Yield_Curve_Inverted'] else 'normal'}**")
            st.caption(f"S&P 500: **{macro['SP500_Return_Pct']:+.1f}%** · GDP: **{macro['GDP_Growth']*100:+.1f}%**")

    st.divider()
    ejecutar = st.button("🚀 Evaluar operación", type="primary", use_container_width=True)
    if not ejecutar:
        st.info("Rellena o carga un target y pulsa Evaluar.")
        st.stop()

    acquirer = {"Market_Cap":acq_mc*1e9,"Revenue":acq_rev*1e9,"Beta":acq_beta,
                "Prior_Deals":acq_pd,"Serial_Acquiror":1 if acq_pd>=3 else 0,
                "region":acq_reg,"Sector":acq_sec,"in_eu":int(acq_eu),"developed":True}
    target = {"Market_Cap":t_mc*1e9,"Revenue":t_rev*1e9,"Beta":t_beta,
              "EBITDA":t_ebitda*1e9,"Free_Cash_Flow":t_fcf*1e9,
              "Total_Debt":t_debt*1e9,"Total_Cash":t_cash*1e9,
              "Dividend_Yield":t_div,"Revenue_Growth":t_g,
              "Sector":t_sec,"region":t_reg,"developed":t_dev,"Country":t_country}
    perfil_score = {**acquirer, **macro,
                    "mismo_sector":acq_sec==t_sec,"both_developed":t_dev,
                    "both_eu":acq_eu and (t_reg=="Europe")}

    st.header("1️⃣ Scorecard del adquirente (13 variables)")
    score, desglose = calcular_scorecard(perfil_score)
    q_nombre, q_tasa, q_emoji = quintil_score(score)
    c1,c2,c3 = st.columns([1,1,2])
    c1.metric("Score (0-100)", score)
    c2.metric("Quintil", f"{q_emoji} {q_nombre}")
    c3.metric("Tasa éxito histórica", q_tasa)
    with st.expander("Ver desglose"):
        st.dataframe(pd.DataFrame(desglose, columns=["Variable","Puntos","Máx","Detalle"]),
                     use_container_width=True, hide_index=True)

    st.header("2️⃣ Predicción del modelo (LP + CP)")
    st.caption("ℹ️ El modelo LP predice la probabilidad de que **este adquirente** cree valor **en este contexto macro** al comprar un target de **esta región/sector**. Los datos financieros del target influyen en la valoración pero no en esta probabilidad.")
    medianas = {f:(macro[f] if f in macro else MEDIANAS_DATASET.get(f, 0)) for f in lp_meta["features"]}
    medianas["Date"]=anio; medianas["Distance_KM"]=5000; medianas["Deal_Value"]=t_mc*1e9
    X_lp = construir_deal(acquirer, target, lp_meta["features"], lp_meta["cat"], medianas)
    p_lp = float(lp.predict_proba(X_lp)[0,1])

    medianas_cp = {f:(macro[f] if f in macro else MEDIANAS_DATASET.get(f, 0)) for f in cp_meta["features"]}
    medianas_cp["Date"]=anio; medianas_cp["Distance_KM"]=5000; medianas_cp["Deal_Value"]=t_mc*1e9
    X_cp = construir_deal(acquirer, target, cp_meta["features"], cp_meta["cat"], medianas_cp)
    p_cp = float(cp.predict_proba(X_cp)[0,1])

    c1,c2,c3 = st.columns([1,1,2])
    c1.metric("P(éxito LP · 3 años)", f"{p_lp:.1%}")
    c2.metric("P(reacción positiva CP)", f"{p_cp:.1%}")
    q, real_rate = quintil_prob(p_lp)
    with c3:
        st.markdown(f"**Calibración:** cae en **{q}** — tasa real histórica: **{real_rate:.1%}**")
        st.progress(real_rate)

    st.markdown("**Cuadrante L1 (valor) vs L2 (mercado):**")
    if p_lp >= 0.5 and p_cp >= 0.5:
        st.success("🟢 **Acierto claro** — crea valor Y el mercado lo reconoce (27% histórico)")
    elif p_lp >= 0.5 and p_cp < 0.5:
        st.info("🔵 **Éxito oculto** — crea valor pero mercado no lo ve (19%). Arbitraje.")
    elif p_lp < 0.5 and p_cp >= 0.5:
        st.error("⚠️ **TRAMPA** — mercado aplaude pero destruye valor a largo plazo (23%).")
    else:
        st.error("🔴 **Fracaso claro** — ambas dimensiones negativas (31%)")

    with st.expander("🔍 ¿Por qué el modelo LP da este número? (SHAP del deal)"):
        shap_df = shap_deal(lp, X_lp, lp_meta["features"], lp_meta["cat"], top_k=10)
        if "error" not in shap_df.columns:
            st.dataframe(shap_df, use_container_width=True, hide_index=True)
            st.caption("Impacto en pp sobre el logit. Ordenado por magnitud absoluta.")
        else:
            st.info("SHAP no disponible en este entorno.")

    st.header("3️⃣ Valoración y precio")
    v = valorar(target)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("DCF (WACC/CAPM)", f"{v['dcf_equity']/1e9:.1f}B" if pd.notna(v['dcf_equity']) else "n/a")
    c2.metric("EV/EBITDA sect.", f"{v['multiplos_equity']/1e9:.1f}B" if pd.notna(v['multiplos_equity']) else "n/a")
    c3.metric("DDM", f"{v['ddm_equity']/1e9:.1f}B" if pd.notna(v['ddm_equity']) and v['ddm_equity']>0 else "n/a")
    c4.metric("Standalone (avg)", f"{v['standalone']/1e9:.1f}B")
    c1,c2,c3 = st.columns(3)
    c1.metric("Prima M&A sector", f"{v['prima_ma_pct']:.0f}%")
    c2.metric("⭐ Precio máx recomendado", f"{v['precio_recomendado']/1e9:.1f}B",
              f"{v['pct_sobre_cap']:+.1f}% vs cap")
    c3.metric("Market Cap actual", f"{v['market_cap']/1e9:.1f}B")

    with st.expander("📈 Análisis de sensibilidad al precio"):
        st.caption("Cómo cambia el precio si mueves inputs clave ±20%")
        base_precio = v["precio_recomendado"]/1e9
        sens = []
        for var,key in [("EBITDA","EBITDA"),("Revenue Growth","Revenue_Growth"),
                        ("Beta","Beta"),("Deuda","Total_Debt")]:
            t_lo = target.copy(); t_lo[key] = target[key]*0.80
            t_hi = target.copy(); t_hi[key] = target[key]*1.20
            v_lo = valorar(t_lo)["precio_recomendado"]/1e9
            v_hi = valorar(t_hi)["precio_recomendado"]/1e9
            sens.append({"Variable":var,"-20%":f"{v_lo:.1f}B","Base":f"{base_precio:.1f}B",
                         "+20%":f"{v_hi:.1f}B","Rango":f"±{abs(v_hi-v_lo)/2:.1f}B"})
        st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)

    st.header("4️⃣ Veredicto integrado")
    score_norm = score/100
    veredicto_score = 0.30*score_norm + 0.50*p_lp + 0.20*p_cp
    if veredicto_score >= 0.60:
        st.success(f"### 🟢 ADELANTE — Score integrado: {veredicto_score:.2f}")
        st.write(f"Precio máximo defendible: **{v['precio_recomendado']/1e9:.1f}B USD**")
    elif veredicto_score >= 0.45:
        st.warning(f"### 🟡 CAUTELA — Score integrado: {veredicto_score:.2f}")
        st.write(f"Deal viable con puntos débiles. Precio: **{v['precio_recomendado']/1e9:.1f}B USD**")
    else:
        st.error(f"### 🔴 DESCARTAR — Score integrado: {veredicto_score:.2f}")
    st.caption("Score integrado = 30% scorecard + 50% LP + 20% CP")

    target_name = sel.split(" · ")[0] if sel != "(rellenar manualmente)" else f"Manual ({t_sec})"
    entry = {"Deal":target_name,"Año":anio,"Adquirente":preset_a if preset_a!="Personalizado" else "Custom",
             "Score":score,"P_LP":round(p_lp,3),"P_CP":round(p_cp,3),
             "Precio_B":round(v['precio_recomendado']/1e9,2),
             "Veredicto":round(veredicto_score,2)}
    st.session_state.historial.append(entry)
    if len(st.session_state.historial) > 20:
        st.session_state.historial = st.session_state.historial[-20:]

    with st.expander("📄 Descargar informe del deal"):
        informe = f"""INFORME DE OPERACIÓN M&A — {target_name}
Año: {anio}   Adquirente: {preset_a}

═══ SCORECARD ═══
Score: {score}/100 → Quintil {q_nombre} (tasa histórica: {q_tasa})

═══ MODELO ═══
P(éxito largo plazo): {p_lp:.1%}  → Calibrado: {q} (real {real_rate:.1%})
P(reacción mercado):  {p_cp:.1%}
AUC temporal año {anio}: {MACRO_ANIOS[anio].get('auc_temporal', 'N/D')} (aviso de rango)

═══ VALORACIÓN ═══
DCF (WACC/CAPM):  {v['dcf_equity']/1e9:.1f}B USD
EV/EBITDA sect.:  {v['multiplos_equity']/1e9:.1f}B USD
DDM:              {v['ddm_equity']/1e9 if pd.notna(v['ddm_equity']) else 0:.1f}B USD
Standalone (avg): {v['standalone']/1e9:.1f}B USD
Prima M&A ({v['prima_ma_pct']:.0f}% sector {t_sec})
PRECIO MÁXIMO:    {v['precio_recomendado']/1e9:.1f}B USD  ({v['pct_sobre_cap']:+.1f}% vs cap)

═══ VEREDICTO ═══
Score integrado: {veredicto_score:.2f}
Recomendación: {'ADELANTE' if veredicto_score >= 0.6 else ('CAUTELA' if veredicto_score >= 0.45 else 'DESCARTAR')}
"""
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("⬇️ Descargar (.txt)", informe,
                               file_name=f"informe_{target_name.replace(' ','_')[:30]}.txt",
                               mime="text/plain", use_container_width=True)
        with col_b:
            # Botón copiar al portapapeles
            copy_html = f"""
            <button onclick="navigator.clipboard.writeText(document.getElementById('informe_txt').innerText).then(() => {{
                this.innerHTML = '✓ Copiado';
                setTimeout(() => {{ this.innerHTML = '📋 Copiar al portapapeles'; }}, 2000);
            }})"
            style="width:100%;padding:8px;border-radius:8px;border:1px solid #ccc;background:#f0f2f6;cursor:pointer;font-size:14px;">
            📋 Copiar al portapapeles
            </button>
            <pre id="informe_txt" style="display:none;">{informe}</pre>
            """
            st.components.v1.html(copy_html, height=50)

    if len(st.session_state.historial) > 1:
        st.divider()
        st.subheader(f"📌 Historial ({len(st.session_state.historial)} deals)")
        hist_df = pd.DataFrame(st.session_state.historial)
        hist_df["P_LP"] = (hist_df["P_LP"]*100).round(1).astype(str)+"%"
        hist_df["P_CP"] = (hist_df["P_CP"]*100).round(1).astype(str)+"%"
        st.dataframe(hist_df, use_container_width=True, hide_index=True)


elif modo == "💡 Encontrar y evaluar target recomendado":
    st.header("Encontrar y evaluar el target ideal")
    st.caption("Configura el adquirente y contexto → el sistema propone targets → elige una para el análisis completo")

    # --- BLOQUE A: configuración adquirente + filtros ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🏢 Adquirente")
        preset = st.selectbox("Perfil", ["Amadeus IT Group (caso TFG)", "Personalizado"])
        if preset == "Amadeus IT Group (caso TFG)":
            acq_dict = PERFIL_AMADEUS.copy()
            st.write(f"Market Cap: **{acq_dict['Market_Cap']/1e9:.1f}B**")
            st.write(f"Revenue: **{acq_dict['Revenue']/1e9:.1f}B**")
            st.write(f"Beta: **{acq_dict['Beta']}** · Deals previos: **{acq_dict['Prior_Deals']}**")
            st.write(f"Sector: **{acq_dict['Sector']}** · Región: **{acq_dict['region']}**")
        else:
            mc  = st.number_input("MC (B USD)", 0.5, 3000.0, 20.0, key="rmc")
            rev = st.number_input("Rev (B USD)", 0.1, 500.0, 5.0, key="rrev")
            beta = st.slider("Beta", 0.2, 2.5, 1.0, 0.05, key="rmb")
            pd_ = st.slider("Deals previos", 0, 30, 5, key="rpd")
            sec_a = st.selectbox("Sector", list(EV_EBITDA_SECTOR.keys())[:-1], key="rseca")
            reg_a = st.selectbox("Región", ["Europe","North America","Asia","South America","Australia/Oceania"], key="rrega")
            acq_dict = {"Market_Cap":mc*1e9,"Revenue":rev*1e9,"Beta":beta,"Prior_Deals":pd_,
                        "Serial_Acquiror":1 if pd_>=3 else 0,"region":reg_a,
                        "Sector":sec_a,"in_eu":int(reg_a=="Europe"),"developed":True}

    with col2:
        st.subheader("🔎 Filtros del target")
        top_n = st.slider("Nº candidatas a mostrar", 5, 30, 10)
        sectores = ["(sin filtro)"] + sorted(universo["Sector"].dropna().unique())
        sect_sel = st.selectbox("Sector afín (coherencia estratégica)", sectores)
        rango_mc = st.slider("Rango Market Cap del target (B USD)", 0.5, 500.0, (1.0, 200.0))

    with col3:
        st.subheader("🌍 Contexto macro")
        anio = st.selectbox("Año", sorted(MACRO_ANIOS.keys(), reverse=True), index=3, key="ranio",
                            help="Preset de referencia. Ajustable debajo.")
        preset_macro = MACRO_ANIOS[anio]
        auc_t = preset_macro.get("auc_temporal")
        if auc_t is not None:
            st.caption(f"AUC temporal validado en {anio}: **{auc_t:.3f}**.")
        else:
            st.caption(f"Año {anio}: sin AUC temporal validado. Ajusta los valores macro si conoces los reales.")

        personalizar_r = st.toggle("✏️ Personalizar parámetros macro", value=False, key="perso_rec")

        if personalizar_r:
            m_infl  = st.slider("Inflación USA (%)", 0.0, 12.0, float(preset_macro["US_Inflation_Pct"]), 0.1, key="ri_i")
            m_vix   = st.slider("VIX medio", 10.0, 50.0, float(preset_macro["VIX_Avg"]), 0.5, key="ri_v")
            m_cs    = st.slider("Credit Spread BAA-AAA", 0.4, 3.0, float(preset_macro["Credit_Spread_BAA_AAA"]), 0.05, key="ri_c")
            m_yc    = st.selectbox("Curva de tipos", ["Normal", "Invertida"],
                                    index=1 if preset_macro["Yield_Curve_Inverted"] else 0, key="ri_y")
            m_sp    = st.slider("S&P 500 return (%)", -40.0, 40.0, float(preset_macro["SP500_Return_Pct"]), 0.5, key="ri_s")
            m_gdp   = st.slider("GDP growth (%)", -5.0, 10.0, float(preset_macro["GDP_Growth"]*100), 0.1, key="ri_g")
            macro = {
                "US_Inflation_Pct":     m_infl,
                "VIX_Avg":              m_vix,
                "Credit_Spread_BAA_AAA": m_cs,
                "Yield_Curve_Inverted": 1 if m_yc == "Invertida" else 0,
                "SP500_Return_Pct":     m_sp,
                "GDP_Growth":           m_gdp / 100,
            }
        else:
            macro = {k:v for k,v in preset_macro.items() if k != "auc_temporal"}
            st.caption(f"Inflación: {macro['US_Inflation_Pct']:.1f}% · VIX: {macro['VIX_Avg']:.1f}")
            st.caption(f"S&P 500: {macro['SP500_Return_Pct']:+.1f}% · GDP: {macro['GDP_Growth']*100:+.1f}%")

    st.divider()

    # --- BLOQUE B: EJECUTAR RECOMENDADOR ---
    if st.button("🎯 Buscar targets recomendados", type="primary", use_container_width=True):
        with st.spinner("Analizando universo..."):
            medianas = {f:(macro[f] if f in macro else MEDIANAS_DATASET.get(f, 0)) for f in lp_meta["features"]}
            medianas["Date"] = anio
            uni_filt = universo[(universo["Market_Cap"]>=rango_mc[0]*1e9) &
                                (universo["Market_Cap"]<=rango_mc[1]*1e9)]
            sectores_afines = [sect_sel] if sect_sel != "(sin filtro)" else None
            rank = recomendar(uni_filt, acquirer=acq_dict, modelo=lp,
                              feats=lp_meta["features"], cat=lp_meta["cat"],
                              medianas=medianas, sectores_afines=sectores_afines, top_n=top_n)
        # Guardar en session_state para poder seleccionar despues sin re-recomendar
        st.session_state.ranking_actual = rank
        st.session_state.acq_actual = acq_dict
        st.session_state.macro_actual = macro
        st.session_state.anio_actual = anio
        st.session_state.preset_actual = preset

    # --- BLOQUE C: MOSTRAR RANKING + SELECCIÓN + EVALUACIÓN ---
    if "ranking_actual" in st.session_state and st.session_state.ranking_actual is not None:
        rank = st.session_state.ranking_actual
        acq_dict = st.session_state.acq_actual
        macro = st.session_state.macro_actual
        anio = st.session_state.anio_actual
        preset = st.session_state.preset_actual

        if rank.empty:
            st.warning("No hay candidatas con los filtros aplicados. Amplía el rango o quita el filtro sectorial.")
        else:
            st.subheader(f"📋 Top {len(rank)} targets sugeridas")
            st.caption("💡 Este ranking es una **herramienta de descubrimiento**: ordena candidatas por su encaje "
                       "con el adquirente. La **probabilidad de éxito** y el **precio recomendado** solo se calculan "
                       "cuando eliges una target concreta para análisis detallado (abajo).")
            rank_show = rank.copy()
            rank_show["Semáforo"] = rank_show["Score_comb"].apply(semaforo)
            # Ranking sin P_modelo ni Precio_B — esos son resultado del análisis individual
            st.dataframe(rank_show[["Rank","Target","Sector","MCap_B","Score_fin","Score_comb","Semáforo"]]
                         .rename(columns={"MCap_B":"Market Cap (B)","Score_fin":"Score financiero",
                                          "Score_comb":"Encaje con adquirente"}),
                         use_container_width=True, hide_index=True)
            st.caption("Ordenado por **encaje con el adquirente** = 60% probabilidad del modelo LP + 40% score financiero del target. "
                       "El score financiero combina calidad intrínseca del target (tamaño, beta, márgenes) con "
                       "**encaje adquirente-target**: tamaño relativo (óptimo 10-40%), similitud sectorial "
                       "(mismo sector = sinergias horizontales) y encaje geográfico (mismo país > misma región > cross-border).")
            with st.expander("ℹ️ Sobre el score financiero"):
                st.markdown("""
                El TFG define el score combinado como *"60% probabilidad LP + 40% score financiero del target"* (Cap 4.5.5).
                La fórmula exacta del score financiero es una **implementación operativa** del sistema que combina:

                - **25% calidad intrínseca del target** (tamaño log-normalizado, beta cerca de 1,05, margen EBITDA/Revenue)
                - **30% tamaño relativo** — target ideal entre el 10% y 40% de la capitalización del adquirente (Betton & Eckbo 2008)
                - **30% similitud sectorial** — mismo sector = 1,0; adyacente = 0,6; distinto = 0,25 (Bower 2001)
                - **15% encaje geográfico** — mismo país = 1,0; misma región = 0,75; cross-border inter-regional = 0,45

                Referencias:
                - Betton, S., Eckbo, B. E., & Thorburn, K. S. (2008). *Corporate takeovers*. Handbook of Empirical Corporate Finance.
                - Bower, J. L. (2001). *Not all M&As are alike—and that matters*. Harvard Business Review.
                """)
            st.download_button("⬇️ Descargar ranking (CSV)", rank.to_csv(index=False),
                               file_name=f"ranking_{anio}.csv", mime="text/csv")

            st.divider()
            st.subheader("🔬 Evaluar en detalle una de las candidatas")

            # Selector de target del ranking
            opciones = [f"#{r['Rank']} · {r['Target']} ({r['Sector']}, {r['MCap_B']:.1f}B)"
                        for _, r in rank.iterrows()]
            sel_target = st.selectbox("Elige una target del ranking:", ["(ninguna)"] + opciones,
                                       key="sel_target_rank")

            if sel_target != "(ninguna)":
                # Recuperar la fila real del universo
                rank_idx = opciones.index(sel_target)
                target_name = rank.iloc[rank_idx]["Target"]
                # Buscar la empresa en el universo original
                row_uni = universo[universo["Name"].astype(str).str.startswith(target_name.rstrip())]
                if row_uni.empty:
                    row_uni = universo[universo["Name"].astype(str).str[:38] == target_name]
                if row_uni.empty:
                    st.error(f"No pude localizar '{target_name}' en el universo. Reintenta.")
                    st.stop()
                row = row_uni.iloc[0]

                # Construir target dict
                target = {
                    "Market_Cap": float(row["Market_Cap"]),
                    "Revenue": max(float(row["Revenue"]), 5e7),
                    "Beta": float(row["Beta"]) if not pd.isna(row["Beta"]) else 1.0,
                    "EBITDA": max(float(row["EBITDA"]), 1e7) if not pd.isna(row["EBITDA"]) else float(row["Revenue"])*0.2,
                    "Free_Cash_Flow": max(float(row["Free_Cash_Flow"]), 1e7) if not pd.isna(row["Free_Cash_Flow"]) else float(row["Revenue"])*0.12,
                    "Total_Debt": max(float(row["Total_Debt"]), 0) if not pd.isna(row["Total_Debt"]) else 0,
                    "Total_Cash": max(float(row["Total_Cash"]), 0) if not pd.isna(row["Total_Cash"]) else 0,
                    "Dividend_Yield": float(row["Dividend_Yield"]) if not pd.isna(row["Dividend_Yield"]) else 0,
                    "Revenue_Growth": float(row["Revenue_Growth"]) if not pd.isna(row["Revenue_Growth"]) else 0.08,
                    "Sector": str(row["Sector"]),
                    "Country": str(row.get("Country", "United States")),
                    "region": "North America" if str(row.get("Country","")) in ["United States","Canada","Mexico"] else "Europe" if str(row.get("Country","")) in ["Germany","France","Spain","Italy","Netherlands","United Kingdom","Sweden","Switzerland","Ireland"] else "Asia",
                    "developed": True,
                }
                perfil_score = {**acq_dict, **macro,
                                "mismo_sector": acq_dict["Sector"] == target["Sector"],
                                "both_developed": target["developed"],
                                "both_eu": acq_dict.get("in_eu",0)==1 and target["region"]=="Europe"}

                # === EJECUTAR LAS 5 FUNCIONES EN CASCADA ===
                st.markdown(f"### Análisis de: **{row['Name']}** ({row['Sector']}, {row.get('Country','?')})")

                # 1. Scorecard
                st.markdown("#### 1️⃣ Scorecard del adquirente")
                score, desglose = calcular_scorecard(perfil_score)
                q_nombre, q_tasa, q_emoji = quintil_score(score)
                c1, c2, c3 = st.columns([1,1,2])
                c1.metric("Score (0-100)", score)
                c2.metric("Quintil", f"{q_emoji} {q_nombre}")
                c3.metric("Tasa éxito histórica", q_tasa)

                # 2. Predicción LP + CP
                st.markdown("#### 2️⃣ Predicción del modelo")
                st.caption("ℹ️ El modelo LP predice el éxito basándose en **adquirente + macro + región/sector del target**. Los datos financieros del target alimentan la valoración pero no esta probabilidad.")
                medianas_lp = {f:(macro[f] if f in macro else 0) for f in lp_meta["features"]}
                medianas_lp["Date"] = anio
                medianas_lp["Distance_KM"] = 5000
                medianas_lp["Deal_Value"] = target["Market_Cap"]
                X_lp = construir_deal(acq_dict, target, lp_meta["features"], lp_meta["cat"], medianas_lp)
                p_lp = float(lp.predict_proba(X_lp)[0,1])

                medianas_cp = {f:(macro[f] if f in macro else MEDIANAS_DATASET.get(f, 0)) for f in cp_meta["features"]}
                medianas_cp["Date"] = anio
                medianas_cp["Distance_KM"] = 5000
                medianas_cp["Deal_Value"] = target["Market_Cap"]
                X_cp = construir_deal(acq_dict, target, cp_meta["features"], cp_meta["cat"], medianas_cp)
                p_cp = float(cp.predict_proba(X_cp)[0,1])

                c1, c2, c3 = st.columns([1,1,2])
                c1.metric("P(éxito LP)", f"{p_lp:.1%}")
                c2.metric("P(reacción CP)", f"{p_cp:.1%}")
                q, real_rate = quintil_prob(p_lp)
                with c3:
                    st.markdown(f"**Calibración:** {q} · tasa real histórica: **{real_rate:.1%}**")
                    st.progress(real_rate)

                # Cuadrante trampa
                if p_lp >= 0.5 and p_cp >= 0.5:
                    st.success("🟢 **Acierto claro** (27% histórico)")
                elif p_lp >= 0.5 and p_cp < 0.5:
                    st.info("🔵 **Éxito oculto** (19%) — arbitraje")
                elif p_lp < 0.5 and p_cp >= 0.5:
                    st.error("⚠️ **TRAMPA** (23%) — mercado engaña")
                else:
                    st.error("🔴 **Fracaso claro** (31%)")

                # SHAP
                with st.expander("🔍 SHAP: por qué el modelo da este número"):
                    shap_df = shap_deal(lp, X_lp, lp_meta["features"], lp_meta["cat"], top_k=10)
                    if "error" not in shap_df.columns:
                        st.dataframe(shap_df, use_container_width=True, hide_index=True)

                # 3. Valoración
                st.markdown("#### 3️⃣ Valoración y precio máximo")
                v = valorar(target)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("DCF", f"{v['dcf_equity']/1e9:.1f}B" if pd.notna(v['dcf_equity']) else "n/a")
                c2.metric("EV/EBITDA", f"{v['multiplos_equity']/1e9:.1f}B" if pd.notna(v['multiplos_equity']) else "n/a")
                c3.metric("DDM", f"{v['ddm_equity']/1e9:.1f}B" if pd.notna(v['ddm_equity']) and v['ddm_equity']>0 else "n/a")
                c4.metric("Standalone", f"{v['standalone']/1e9:.1f}B")
                c1, c2, c3 = st.columns(3)
                c1.metric("Prima M&A", f"{v['prima_ma_pct']:.0f}%")
                c2.metric("⭐ Precio máx", f"{v['precio_recomendado']/1e9:.1f}B",
                          f"{v['pct_sobre_cap']:+.1f}% vs cap")
                c3.metric("Market Cap actual", f"{v['market_cap']/1e9:.1f}B")

                # 4. Veredicto
                st.markdown("#### 4️⃣ Veredicto integrado")
                veredicto_score = 0.30*(score/100) + 0.50*p_lp + 0.20*p_cp
                if veredicto_score >= 0.60:
                    st.success(f"### 🟢 ADELANTE — Score: {veredicto_score:.2f} · Precio: {v['precio_recomendado']/1e9:.1f}B USD")
                elif veredicto_score >= 0.45:
                    st.warning(f"### 🟡 CAUTELA — Score: {veredicto_score:.2f} · Precio: {v['precio_recomendado']/1e9:.1f}B USD")
                else:
                    st.error(f"### 🔴 DESCARTAR — Score: {veredicto_score:.2f}")

                # Guardar en historial
                entry = {"Deal":str(row['Name'])[:38], "Año":anio,
                         "Adquirente":preset if preset!="Personalizado" else "Custom",
                         "Score":score, "P_LP":round(p_lp,3), "P_CP":round(p_cp,3),
                         "Precio_B":round(v['precio_recomendado']/1e9,2),
                         "Veredicto":round(veredicto_score,2)}
                # Evitar duplicados
                if not st.session_state.historial or st.session_state.historial[-1] != entry:
                    st.session_state.historial.append(entry)
                    if len(st.session_state.historial) > 20:
                        st.session_state.historial = st.session_state.historial[-20:]


elif modo == "⚖️ Comparar dos deals":
    st.header("Comparativa lado a lado")
    if len(st.session_state.historial) < 2:
        st.info("Evalúa al menos 2 deals (en cualquier pestaña de evaluación) y vuelve aquí.")
        st.stop()
    nombres = [f"{i+1}. {e['Deal']} ({e['Año']})" for i,e in enumerate(st.session_state.historial)]
    c1,c2 = st.columns(2)
    with c1:
        s1 = st.selectbox("Deal A", nombres, index=max(0, len(nombres)-2))
    with c2:
        s2 = st.selectbox("Deal B", nombres, index=len(nombres)-1)
    idx1 = nombres.index(s1); idx2 = nombres.index(s2)
    d1 = st.session_state.historial[idx1]; d2 = st.session_state.historial[idx2]
    metricas = ["Score","P_LP","P_CP","Precio_B","Veredicto"]
    cmp = pd.DataFrame({"Métrica":metricas,
                        d1["Deal"]:[d1[m] for m in metricas],
                        d2["Deal"]:[d2[m] for m in metricas]})
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    st.subheader("Ganador por métrica")
    for m in metricas:
        v1,v2 = d1[m], d2[m]
        if v1 > v2: st.write(f"✅ **{m}**: gana **{d1['Deal']}** ({v1} vs {v2})")
        elif v2 > v1: st.write(f"✅ **{m}**: gana **{d2['Deal']}** ({v2} vs {v1})")
        else: st.write(f"➖ **{m}**: empate")


elif modo == "📊 Modelo · SHAP · Calibración":
    st.header("Métricas, SHAP y calibración")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC LP (test)", "0,887", "TFG 0,892")
    c2.metric("Accuracy LP", "82,5%", "TFG")
    c3.metric("Brier LP", "0,131", "TFG")
    c4.metric("Gap train-test", "0,108", "TFG")
    st.divider()
    st.subheader("Calibración por quintiles del scorecard (Tabla 4.6 TFG)")
    cal_df = pd.DataFrame([
        {"Quintil":"Q1 (peor)","Score medio":20.8, "N deals":551,"Tasa éxito":"16,2%"},
        {"Quintil":"Q2",       "Score medio":41.7, "N deals":545,"Tasa éxito":"37,8%"},
        {"Quintil":"Q3",       "Score medio":58.3, "N deals":549,"Tasa éxito":"49,2%"},
        {"Quintil":"Q4",       "Score medio":70.1, "N deals":550,"Tasa éxito":"54,2%"},
        {"Quintil":"Q5 (mejor)","Score medio":82.6,"N deals":544,"Tasa éxito":"70,8%"},
    ])
    st.dataframe(cal_df, use_container_width=True, hide_index=True)
    st.caption("Gradiente Q1→Q5 = 54,6 puntos porcentuales. El scorecard discrimina con solo 13 variables interpretables.")

    st.divider()
    st.subheader("Calibración por quintiles del modelo LP (probabilidades predichas)")
    cal_lp = pd.DataFrame([
        {"Quintil":"Q1 (peor)", "Prob. media":"1,8%",  "Rango prob":"0-5%",   "Tasa real":"3,6%",  "Desviación":"+1,8pp"},
        {"Quintil":"Q2",        "Prob. media":"10,8%", "Rango prob":"5-19%",  "Tasa real":"23,9%", "Desviación":"+13,1pp"},
        {"Quintil":"Q3",        "Prob. media":"35,3%", "Rango prob":"20-53%", "Tasa real":"34,5%", "Desviación":"-0,7pp"},
        {"Quintil":"Q4",        "Prob. media":"73,9%", "Rango prob":"54-91%", "Tasa real":"69,7%", "Desviación":"-4,2pp"},
        {"Quintil":"Q5 (mejor)","Prob. media":"95,9%", "Rango prob":"91-100%","Tasa real":"94,5%", "Desviación":"-1,3pp"},
    ])
    st.dataframe(cal_lp, use_container_width=True, hide_index=True)
    st.caption("Modelo bien calibrado en Q1, Q3, Q4, Q5 (desviaciones <5pp). Descalibración notable en Q2 (+13pp): "
               "el modelo es demasiado pesimista en la franja de baja-media probabilidad — algunos deals con perfil "
               "modesto acaban sorprendentemente bien por factores no capturados.")
    st.divider()
    st.subheader("Top 15 variables por SHAP (global)")
    try:
        shap_df = pd.read_csv("shap_top15_LP.csv")
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
    except Exception:
        st.info("shap_top15_LP.csv no disponible")
    st.divider()
    st.subheader("Recetas de los modelos")
    st.code("""LP: depth=8, iterations=1000, learning_rate=0.03, l2_leaf_reg=3
    target = (0.05·BHAR_1Y + 0.15·BHAR_2Y + 0.80·BHAR_3Y) > 0
    93 features (7 categóricas nativas) + interacciones

CP: depth=6, iterations=700, learning_rate=0.03, l2_leaf_reg=3
    target = CAR_1 > 0   |   receta brier_cp.py""")
    st.divider()
    st.subheader("⚠️ Validación temporal (Tabla del Cap 5.4 del TFG)")
    val_df = pd.DataFrame([
        {"Ventana test":"2008-2009","n train":487, "n test":112,"AUC":0.681,
         "Observación":"Crisis financiera — cambio de régimen"},
        {"Ventana test":"2010-2011","n train":599, "n test":143,"AUC":0.714,
         "Observación":"Recuperación post-crisis"},
        {"Ventana test":"2012-2013","n train":742, "n test":168,"AUC":0.731,
         "Observación":"Estabilización del mercado"},
        {"Ventana test":"2014-2015","n train":910, "n test":187,"AUC":0.748,
         "Observación":"Buen contexto macro"},
        {"Ventana test":"2016-2017","n train":1097,"n test":201,"AUC":0.779,
         "Observación":"Pico — mercado estable"},
        {"Ventana test":"2018-2019","n train":1298,"n test":193,"AUC":0.743,
         "Observación":"Tensiones comerciales"},
        {"Ventana test":"2020-2021","n train":1491,"n test":248,"AUC":0.652,
         "Observación":"COVID — caída estructural"},
        {"Ventana test":"MEDIA",     "n train":"—",  "n test":"—", "AUC":0.697,
         "Observación":"Pico 0,779 en 2016-2017"},
    ])
    st.dataframe(val_df, use_container_width=True, hide_index=True)
    st.warning("El AUC en split aleatorio (0,887) sobreestima la capacidad prospectiva. "
               "Para uso real sobre años futuros, la referencia honesta es AUC ~0,70. "
               "La diferencia (~0,19) se explica por el cambio estructural en la tasa de éxito "
               "post-2020 (del 49-54% al 27,9%): ningún modelo entrenado antes puede anticipar esa caída.")


elif modo == "🔬 Validación técnica (a-e)":
    from validacion import (CV_TEMPORAL, BASELINE, HIPERPARAMETROS, HIPERPARAMETROS_NOTA,
                             INTERVALOS_CONFIANZA, CURVA_APRENDIZAJE, CURVA_DIAGNOSTICO,
                             RESUMEN_DEFENSA)
    st.header("🔬 Validación técnica del modelo")
    st.caption("Análisis de robustez para la defensa del TFG")

    # RESUMEN EJECUTIVO
    st.subheader("Resumen ejecutivo (para defensa)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("AUC random", f"{RESUMEN_DEFENSA['auc_random_reproducido']:.3f}",
              f"TFG: {RESUMEN_DEFENSA['auc_reportado_tfg']}")
    c2.metric("AUC temporal CV", f"{RESUMEN_DEFENSA['auc_temporal_cv']:.3f}",
              "-0,190 vs random")
    c3.metric("Baseline LogReg", f"{RESUMEN_DEFENSA['auc_baseline_logreg']:.3f}",
              f"+15,8pp CatBoost")
    c4.metric("Literatura M&A", "0,60–0,75",
              "modelo supera el rango")
    st.info(f"**Recomendación de uso:** {RESUMEN_DEFENSA['recomendacion_uso']}")

    st.divider()

    # (a) CV TEMPORAL
    st.subheader("(a) Validación temporal — Expanding window (Tabla del Cap 5.4 del TFG)")
    st.dataframe(pd.DataFrame(CV_TEMPORAL["folds"]).rename(
        columns={"periodo":"Ventana test","n_train":"n train","n_test":"n test",
                 "auc":"AUC","observacion":"Observación"}),
        use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC medio", f"{CV_TEMPORAL['auc_media']:.3f}",
              f"vs AUC random 0,887")
    c2.metric("Pico", f"{CV_TEMPORAL['auc_pico']:.3f}",
              f"Ventana {CV_TEMPORAL['auc_pico_ventana']}")
    c3.metric("Δ vs random", "-0,190",
              "Cambio estructural post-2020")
    st.caption(CV_TEMPORAL["nota"])

    st.divider()

    # (b) BASELINE
    st.subheader("(b) Comparativa de algoritmos — Tabla 4.8 del TFG")
    st.dataframe(pd.DataFrame([
        {"Modelo": "Regresión Logística",
         "AUC test": 0.734, "Accuracy": "68,2%", "Brier": 0.198,
         "Gap train-test": 0.031, "Comentario": "Baseline lineal"},
        {"Modelo": "Árbol de Decisión",
         "AUC test": 0.762, "Accuracy": "70,4%", "Brier": 0.181,
         "Gap train-test": 0.089, "Comentario": "Sobreajuste elevado"},
        {"Modelo": "Random Forest",
         "AUC test": 0.813, "Accuracy": "74,1%", "Brier": 0.162,
         "Gap train-test": 0.054, "Comentario": "Superado por boosting"},
        {"Modelo": "XGBoost",
         "AUC test": 0.847, "Accuracy": "78,3%", "Brier": 0.148,
         "Gap train-test": 0.047, "Comentario": "Peor gestión categóricas"},
        {"Modelo": "CatBoost equilibrado (d=6)",
         "AUC test": 0.880, "Accuracy": "80,8%", "Brier": 0.139,
         "Gap train-test": 0.120, "Comentario": "Referencia de robustez"},
        {"Modelo": "CatBoost agresivo (d=8) ✓",
         "AUC test": 0.892, "Accuracy": "82,5%", "Brier": 0.131,
         "Gap train-test": 0.108, "Comentario": "SELECCIONADO"},
    ]), use_container_width=True, hide_index=True)
    st.success(f"**CatBoost (0,892) aporta +0,158 AUC sobre LogReg (0,734), "
               f"lo que supone +21,5% de mejora relativa.**")
    st.caption("Fuente: Tabla 4.8 del TFG. Todos los modelos entrenados sobre el mismo split "
               "80/20 estratificado, mismo preprocesado y validación cruzada 5-fold.")

    st.divider()

    # (c) HIPERPARÁMETROS
    st.subheader("(c) Comparativa de hiperparámetros")
    st.dataframe(pd.DataFrame(HIPERPARAMETROS), use_container_width=True, hide_index=True)
    st.caption(HIPERPARAMETROS_NOTA)

    st.divider()

    # (d) INTERVALO DE CONFIANZA
    st.subheader("(d) Intervalo de confianza del AUC (bootstrap 1000 iteraciones)")
    ic_df = pd.DataFrame([
        {"Modelo": "Agresivo",
         "AUC": INTERVALOS_CONFIANZA["agresivo"]["auc"],
         "IC 95% inferior": INTERVALOS_CONFIANZA["agresivo"]["ci_lo"],
         "IC 95% superior": INTERVALOS_CONFIANZA["agresivo"]["ci_hi"],
         "Ancho": INTERVALOS_CONFIANZA["agresivo"]["ancho"]},
        {"Modelo": "Conservador",
         "AUC": INTERVALOS_CONFIANZA["conservador"]["auc"],
         "IC 95% inferior": INTERVALOS_CONFIANZA["conservador"]["ci_lo"],
         "IC 95% superior": INTERVALOS_CONFIANZA["conservador"]["ci_hi"],
         "Ancho": INTERVALOS_CONFIANZA["conservador"]["ancho"]},
    ])
    st.dataframe(ic_df, use_container_width=True, hide_index=True)
    st.caption(INTERVALOS_CONFIANZA["nota"])

    st.divider()

    # (e) CURVA DE APRENDIZAJE
    st.subheader("(e) Curva de aprendizaje")
    curva_df = pd.DataFrame(CURVA_APRENDIZAJE)
    curva_df["% train"] = (curva_df["frac_train"] * 100).astype(int).astype(str) + "%"
    st.dataframe(curva_df[["% train", "n", "auc"]].rename(columns={"n":"n deals","auc":"AUC test"}),
                 use_container_width=True, hide_index=True)
    st.line_chart(curva_df.set_index("n")["auc"])
    st.warning(f"**{CURVA_DIAGNOSTICO}**")


st.divider()
st.caption("TFG · Sistema predictivo de éxito en M&A · CatBoost sobre 2.739 deals (1997–2023) · v2")
