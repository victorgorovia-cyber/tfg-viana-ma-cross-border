"""
TFG M&A — Adaptador del universo real
======================================
Toma `index_companies.csv` (generado por download_index_companies_v2.py con
datos reales de Yahoo Finance) y lo convierte al formato que espera el
recomendador (`universo_empresas.csv`).

Hace tres cosas:
  1. Convierte todas las monedas a USD.
  2. Filtra a rango de market cap razonable (0.5B–500B).
  3. Renombra columnas y añade defaults sensatos donde falten datos.

Uso:
    python3 preparar_universo.py

Genera:
    universo_empresas.csv  (listo para copiar a modelos_reales/)
"""

import pandas as pd
import numpy as np


INPUT_FILE  = "index_companies.csv"
OUTPUT_FILE = "universo_empresas.csv"

# Tipos de cambio aproximados a USD (medias 2024)
FX_TO_USD = {
    "USD": 1.00, "EUR": 1.08, "GBP": 1.27, "GBp": 0.0127,
    "JPY": 0.0067, "CAD": 0.74, "AUD": 0.66, "CHF": 1.12,
    "SEK": 0.095, "NOK": 0.094, "DKK": 0.145, "SGD": 0.74,
    "HKD": 0.128, "CNY": 0.14, "TWD": 0.031, "KRW": 0.00075,
    "INR": 0.012, "BRL": 0.20, "MXN": 0.058, "ZAR": 0.054, "ILS": 0.27,
}

# Rango sano de market cap (USD): large cap para M&A defendible
MIN_MCAP = 10_000_000_000    # 10B$ (large cap)
MAX_MCAP = 3_000_000_000_000 # 3T$ (para incluir Apple, Microsoft, etc.)


def detectar_moneda(row) -> str:
    """
    Intenta detectar la moneda por el sufijo del ticker.
    (index_companies.csv no tiene columna Currency explícita.)
    """
    ticker = str(row.get("Ticker", ""))
    if "." not in ticker:
        return "USD"                    # tickers sin sufijo = USA
    suffix = ticker.split(".")[-1]
    return {
        "DE": "EUR", "PA": "EUR", "AS": "EUR", "MC": "EUR", "MI": "EUR",
        "BR": "EUR", "HE": "EUR", "CO": "EUR",
        "L":  "GBP", "SW": "CHF",
        "ST": "SEK", "OL": "NOK",
        "T":  "JPY", "HK": "HKD", "KS": "KRW", "SI": "SGD",
        "AX": "AUD", "TO": "CAD",
        "SA": "BRL", "JO": "ZAR",
    }.get(suffix, "USD")


def main():
    print("=" * 60)
    print("ADAPTADOR DEL UNIVERSO — index_companies.csv → universo_empresas.csv")
    print("=" * 60)

    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"[!] {INPUT_FILE} no encontrado.")
        print("    Ejecuta primero: python3 download_index_companies_v2.py")
        return

    print(f"\n[1] Cargadas {len(df)} empresas desde {INPUT_FILE}")

    # ============ 1) CONVERSIÓN A USD ============
    df["_currency"] = df.apply(detectar_moneda, axis=1)
    df["_fx"] = df["_currency"].map(FX_TO_USD).fillna(1.0)
    for col in ["Market_Cap", "Revenue", "Enterprise_Value", "EBITDA",
                "Free_Cash_Flow", "Total_Debt", "Total_Cash"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * df["_fx"]

    print(f"[2] Convertidas monedas a USD "
          f"({df['_currency'].value_counts().head(5).to_dict()}…)")

    # ============ 2) FILTRO DE RANGO SANO ============
    n0 = len(df)
    df = df[(df["Market_Cap"] >= MIN_MCAP) & (df["Market_Cap"] <= MAX_MCAP)]
    print(f"[3] Filtrado por market cap {MIN_MCAP/1e9:.0f}B–{MAX_MCAP/1e12:.0f}T USD (large cap): {n0} → {len(df)}")

    # ============ 3) ADAPTACIÓN AL FORMATO DEL RECOMENDADOR ============
    df = df.rename(columns={"Country": "Country"})   # ya está bien
    if "Name" not in df.columns and "shortName" in df.columns:
        df = df.rename(columns={"shortName": "Name"})

    # Rellenos de seguridad para el recomendador y la valoración
    df["Beta"] = pd.to_numeric(df.get("Beta"), errors="coerce").fillna(1.0)
    for c in ["Revenue", "EBITDA", "Free_Cash_Flow", "Total_Debt", "Total_Cash",
              "Dividend_Yield", "Revenue_Growth", "ROE", "Operating_Margin",
              "Profit_Margin", "Debt_to_Equity"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Estimaciones por sector solo si el campo está vacío
    if "EBITDA" not in df.columns or df["EBITDA"].isna().all():
        df["EBITDA"] = df["Revenue"] * 0.20
    if "Free_Cash_Flow" not in df.columns:
        df["Free_Cash_Flow"] = df["Revenue"] * 0.12
    if "Total_Debt" not in df.columns:
        df["Total_Debt"] = df["Market_Cap"] * 0.25
    if "Total_Cash" not in df.columns:
        df["Total_Cash"] = df["Market_Cap"] * 0.10

    df["EBITDA"]           = df["EBITDA"].fillna(df["Revenue"] * 0.20)
    df["Free_Cash_Flow"]   = df["Free_Cash_Flow"].fillna(df["Revenue"] * 0.12)
    df["Total_Debt"]       = df["Total_Debt"].fillna(df["Market_Cap"] * 0.25)
    df["Total_Cash"]       = df["Total_Cash"].fillna(df["Market_Cap"] * 0.10)
    df["Dividend_Yield"]   = df.get("Dividend_Yield", 0).fillna(0)
    df["Revenue_Growth"]   = df.get("Revenue_Growth", 0.08).fillna(0.08)
    df["ROE"]              = df.get("ROE", 0.12).fillna(0.12)
    df["Operating_Margin"] = df.get("Operating_Margin", 0.15).fillna(0.15)
    df["Profit_Margin"]    = df.get("Profit_Margin", 0.10).fillna(0.10)
    df["Debt_to_Equity"]   = df.get("Debt_to_Equity", 50).fillna(50)

    # Columnas finales
    cols = ["Name", "Ticker", "Sector", "Country", "Market_Cap", "Revenue",
            "Beta", "EBITDA", "Free_Cash_Flow", "Total_Debt", "Total_Cash",
            "Dividend_Yield", "Revenue_Growth", "ROE", "Operating_Margin",
            "Profit_Margin", "Debt_to_Equity"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(OUTPUT_FILE, index=False)

    print(f"\n[4] Guardado {OUTPUT_FILE} con {len(df)} empresas.")
    print(f"\nTop 10 por market cap (USD):")
    for _, r in df.nlargest(10, "Market_Cap").iterrows():
        name = str(r.get("Name", r.get("Ticker","?")))[:35]
        print(f"   {name:<37} {r['Market_Cap']/1e9:>8.0f}B  "
              f"{str(r.get('Sector','?'))[:20]:<22} ROE={r.get('ROE',0):.1%}")

    print(f"\nDistribución por sector:")
    print(df["Sector"].value_counts().head(10).to_string())

    print(f"\n[✓] LISTO. Copia {OUTPUT_FILE} a la carpeta modelos_reales/")
    print(f"    y ejecuta uso_rapido.py para probarlo.")


if __name__ == "__main__":
    main()
