"""
=============================================================================
TFG M&A — Descarga UNIVERSO GLOBAL de empresas cotizadas (v3)
=============================================================================
Extrae componentes de índices bursátiles del mundo entero (US + Europa + Asia
+ emergentes), deduplica y descarga financieros de Yahoo Finance.

Diferencias vs v2:
  - v2: lista hardcoded de ~700 tickers
  - v3: extracción DINÁMICA de índices → ~5000-8000 tickers globales
  - v3: filtro final a large cap (≥ 10B USD) para universo M&A defendible

Uso:
    pip3 install yfinance pandas requests lxml beautifulsoup4
    python3 download_index_companies_v3.py

Salida:
    index_companies.csv  — todas las empresas descargadas con datos
    universo_global_report.txt — resumen por región / sector / país

Tiempo estimado: 2-4 horas (batch, se puede pausar/reanudar con caché)
=============================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import json
import os
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
OUTPUT_DIR    = "bhar_output_v3"
CACHE_FILE    = f"{OUTPUT_DIR}/download_cache_v3.json"
OUTPUT_CSV    = "index_companies.csv"
REPORT_FILE   = "universo_global_report.txt"
BATCH_SIZE    = 20
DELAY         = 0.4   # segundos entre tickers
MIN_MARKET_CAP_USD = 10_000_000_000   # ≥ 10B USD (large cap)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# EXTRACCIÓN DINÁMICA DE COMPONENTES DE ÍNDICES (fuentes públicas)
# =============================================================================
def sp500_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        return df['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"   [!] S&P 500: {e}")
        return []


def nasdaq100_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'Symbol' in str(c)][0]
        return df[col].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"   [!] Nasdaq 100: {e}")
        return []


def dow30_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]
        col = [c for c in df.columns if 'Symbol' in str(c)][0]
        return df[col].tolist()
    except Exception:
        return []


def ftse100_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index')[4]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'EPIC' in str(c)][0]
        return [f"{t}.L" for t in df[col].astype(str).tolist()]
    except Exception as e:
        print(f"   [!] FTSE 100: {e}")
        return []


def ftse250_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/FTSE_250_Index')[3]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'EPIC' in str(c)][0]
        return [f"{t}.L" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def dax_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/DAX')[4]
        col = [c for c in df.columns if 'Ticker' in str(c)][0]
        return [f"{t}.DE" for t in df[col].astype(str).tolist()]
    except Exception as e:
        print(f"   [!] DAX: {e}")
        return []


def mdax_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/MDAX')[1]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'Symbol' in str(c)][0]
        return [f"{t}.DE" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def cac40_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/CAC_40')[4]
        col = [c for c in df.columns if 'Ticker' in str(c)][0]
        return [f"{t}.PA" for t in df[col].astype(str).tolist()]
    except Exception as e:
        print(f"   [!] CAC 40: {e}")
        return []


def ibex35_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/IBEX_35')[2]
        col = [c for c in df.columns if 'Ticker' in str(c)][0]
        return [f"{t}.MC" for t in df[col].astype(str).tolist()]
    except Exception as e:
        print(f"   [!] IBEX 35: {e}")
        return []


def ftse_mib_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/FTSE_MIB')[1]
        col = [c for c in df.columns if 'Ticker' in str(c)][0]
        return [f"{t}.MI" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def aex_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/AEX_index')[2]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'Symbol' in str(c)][0]
        return [f"{t}.AS" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def smi_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/Swiss_Market_Index')[1]
        col = [c for c in df.columns if 'Ticker' in str(c)][0]
        return [f"{t}.SW" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def omx_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/OMX_Stockholm_30')[1]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'Symbol' in str(c)][0]
        return [f"{t}.ST" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def stoxx600_tickers():
    """STOXX 600 = las 600 principales de Europa. Cubre lo de arriba y más."""
    # Wikipedia tiene la lista, pero es larga; damos la de STOXX Europe 50 como núcleo
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/EURO_STOXX_50')[2]
        col = [c for c in df.columns if 'Ticker' in str(c)][0]
        return df[col].astype(str).tolist()
    except Exception:
        return []


def nikkei225_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/Nikkei_225')[1]
        col = [c for c in df.columns if 'Code' in str(c) or 'Ticker' in str(c)][0]
        return [f"{t}.T" for t in df[col].astype(str).tolist()]
    except Exception as e:
        print(f"   [!] Nikkei 225: {e}")
        return []


def hangseng_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/Hang_Seng_Index')[6]
        col = [c for c in df.columns if 'Ticker' in str(c)][0]
        return [f"{str(t).zfill(4)}.HK" for t in df[col].astype(str).tolist()]
    except Exception as e:
        print(f"   [!] Hang Seng: {e}")
        return []


def kospi_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/KOSPI_100')[1]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'Symbol' in str(c)][0]
        return [f"{str(t).zfill(6)}.KS" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def sti_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/Straits_Times_Index')[1]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'Symbol' in str(c)][0]
        return [f"{t}.SI" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def asx200_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/S%26P/ASX_200')[2]
        col = [c for c in df.columns if 'Code' in str(c) or 'Ticker' in str(c)][0]
        return [f"{t}.AX" for t in df[col].astype(str).tolist()]
    except Exception as e:
        print(f"   [!] ASX 200: {e}")
        return []


def tsx60_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/S%26P/TSX_60')[0]
        col = [c for c in df.columns if 'Symbol' in str(c) or 'Ticker' in str(c)][0]
        return [f"{t}.TO" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def nifty50_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/NIFTY_50')[1]
        col = [c for c in df.columns if 'Symbol' in str(c) or 'Ticker' in str(c)][0]
        return [f"{t}.NS" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def bovespa_tickers():
    try:
        df = pd.read_html('https://en.wikipedia.org/wiki/List_of_companies_listed_on_B3')[0]
        col = [c for c in df.columns if 'Ticker' in str(c) or 'Symbol' in str(c)][0]
        return [f"{t}.SA" for t in df[col].astype(str).tolist()]
    except Exception:
        return []


def ta125_tickers():
    """Israel — TA-125. Muy pocos disponibles vía wikipedia, hardcoded top."""
    return ['CHKP', 'NICE', 'CYBR', 'MNDY', 'WIX', 'GLBE', 'SLM', 'SEDG',
            'ORLY', 'CAMT', 'FIVN', 'ELUXY', 'PLTK', 'JFROG']


def emerging_extras():
    """ADRs y otros grandes emergentes no capturados arriba."""
    return [
        # China (ADRs porque los A-shares no van bien en yfinance)
        'BABA','PDD','BIDU','JD','NIO','LI','XPEV','TCEHY','YUMC',
        # LATAM
        'VALE','PBR','ITUB','BBD','NU','MELI','BAP','GGB','CIB',
        # México
        'AMX','FMX','GRBMF','GFNORTEO.MX',
        # Sudáfrica ADR
        'GOLD','GFI','SBSW',
        # Turquía
        'AKBNK.IS','GARAN.IS','ISCTR.IS','THYAO.IS',
        # Emiratos
        'IHC.AE','EMAAR.AE',
    ]


# =============================================================================
# DESCARGA CON CACHÉ Y RETRIES
# =============================================================================
def load_cache():
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def fetch_ticker_data(ticker):
    """Descarga datos financieros completos de un ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        mcap = info.get('marketCap', 0)
        if not mcap or mcap <= 0:
            return None
        return {
            'Ticker': ticker,
            'Name': info.get('shortName', info.get('longName', ticker)),
            'Country': info.get('country', ''),
            'Sector': info.get('sector', 'Unknown'),
            'Industry': info.get('industry', ''),
            'Currency': info.get('currency', 'USD'),
            'Market_Cap': mcap,
            'Revenue': info.get('totalRevenue', 0) or 0,
            'Beta': info.get('beta', None),
            'PE_Ratio': info.get('trailingPE', None),
            'Forward_PE': info.get('forwardPE', None),
            'Employees': info.get('fullTimeEmployees', None),
            'Dividend_Yield': info.get('dividendYield', None),
            'Debt_to_Equity': info.get('debtToEquity', None),
            'ROE': info.get('returnOnEquity', None),
            'ROA': info.get('returnOnAssets', None),
            'Profit_Margin': info.get('profitMargins', None),
            'Operating_Margin': info.get('operatingMargins', None),
            'Gross_Margin': info.get('grossMargins', None),
            'Revenue_Growth': info.get('revenueGrowth', None),
            'Earnings_Growth': info.get('earningsGrowth', None),
            'Free_Cash_Flow': info.get('freeCashflow', None),
            'Enterprise_Value': info.get('enterpriseValue', None),
            'EV_EBITDA': info.get('enterpriseToEbitda', None),
            'EV_Revenue': info.get('enterpriseToRevenue', None),
            'Price_to_Book': info.get('priceToBook', None),
            'Price_to_Sales': info.get('priceToSalesTrailing12Months', None),
            'EBITDA': info.get('ebitda', None),
            'Total_Debt': info.get('totalDebt', None),
            'Total_Cash': info.get('totalCash', None),
            'Current_Ratio': info.get('currentRatio', None),
            'Quick_Ratio': info.get('quickRatio', None),
        }
    except Exception:
        return None


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("DESCARGA UNIVERSO GLOBAL v3 — Todos los índices del mundo")
    print("=" * 70)

    # === 1. EXTRAER COMPONENTES DE TODOS LOS ÍNDICES ===
    print("\n[1] Extrayendo componentes de índices...")
    fuentes = [
        ("S&P 500",       sp500_tickers),
        ("Nasdaq 100",    nasdaq100_tickers),
        ("Dow Jones 30",  dow30_tickers),
        ("TSX 60 (CA)",   tsx60_tickers),
        ("FTSE 100 (UK)", ftse100_tickers),
        ("FTSE 250 (UK)", ftse250_tickers),
        ("DAX (DE)",      dax_tickers),
        ("MDAX (DE)",     mdax_tickers),
        ("CAC 40 (FR)",   cac40_tickers),
        ("IBEX 35 (ES)",  ibex35_tickers),
        ("FTSE MIB (IT)", ftse_mib_tickers),
        ("AEX (NL)",      aex_tickers),
        ("SMI (CH)",      smi_tickers),
        ("OMX 30 (SE)",   omx_tickers),
        ("EURO STOXX 50", stoxx600_tickers),
        ("Nikkei 225",    nikkei225_tickers),
        ("Hang Seng",     hangseng_tickers),
        ("KOSPI 100",     kospi_tickers),
        ("STI (SG)",      sti_tickers),
        ("ASX 200 (AU)",  asx200_tickers),
        ("NIFTY 50 (IN)", nifty50_tickers),
        ("Bovespa (BR)",  bovespa_tickers),
        ("TA-125 (IL)",   ta125_tickers),
    ]

    todos_tickers = set()
    for nombre, fn in fuentes:
        tickers = fn()
        print(f"   {nombre:20} → {len(tickers)} tickers")
        todos_tickers.update(tickers)

    # Extras emergentes
    extras = emerging_extras()
    todos_tickers.update(extras)
    print(f"   Emergentes (extras)  → {len(extras)} tickers")

    todos_tickers = [t for t in todos_tickers if t and isinstance(t, str)]
    print(f"\n   TOTAL únicos: {len(todos_tickers)} tickers")

    # === 2. DESCARGA CON CACHÉ ===
    print(f"\n[2] Descargando datos financieros de Yahoo Finance...")
    print(f"   (Se guarda caché cada 100 tickers en {CACHE_FILE}. Puedes pausar y reanudar.)")

    cache = load_cache()
    pending = [t for t in todos_tickers if t not in cache]
    print(f"   Ya en caché: {len(cache)}")
    print(f"   Pendientes:  {len(pending)}")

    if pending:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(pending), total=len(pending), desc="Descargando")
        except ImportError:
            iterator = enumerate(pending)

        for i, ticker in iterator:
            data = fetch_ticker_data(ticker)
            cache[ticker] = data  # None si no hay datos
            time.sleep(DELAY)
            if (i + 1) % 100 == 0:
                save_cache(cache)
        save_cache(cache)

    # === 3. FILTRAR LARGE CAP Y GUARDAR ===
    print(f"\n[3] Filtrando large cap (≥ {MIN_MARKET_CAP_USD/1e9:.0f}B USD)...")
    rows = [v for v in cache.values() if v is not None]
    df = pd.DataFrame(rows)
    print(f"   Con datos válidos: {len(df)}")

    # Conversión rápida a USD para el filtro
    fx = {"USD":1.0,"EUR":1.08,"GBP":1.27,"GBp":0.0127,"JPY":0.0067,"CAD":0.74,
          "AUD":0.66,"CHF":1.12,"SEK":0.095,"NOK":0.094,"DKK":0.145,"SGD":0.74,
          "HKD":0.128,"CNY":0.14,"TWD":0.031,"KRW":0.00075,"INR":0.012,"BRL":0.20,
          "MXN":0.058,"ZAR":0.054,"ILS":0.27,"TRY":0.030,"AED":0.27}
    df["_fx"] = df["Currency"].map(fx).fillna(1.0)
    df["Market_Cap_USD"] = df["Market_Cap"] * df["_fx"]

    df_large = df[df["Market_Cap_USD"] >= MIN_MARKET_CAP_USD].copy()
    print(f"   Large cap (≥{MIN_MARKET_CAP_USD/1e9:.0f}B USD): {len(df_large)}")

    # Ordenar y guardar
    df_large = df_large.sort_values("Market_Cap_USD", ascending=False)
    df_large.drop(columns=["_fx"]).to_csv(OUTPUT_CSV, index=False)

    # === 4. REPORTE ===
    with open(REPORT_FILE, 'w') as f:
        f.write("UNIVERSO GLOBAL v3 — REPORTE\n" + "="*60 + "\n\n")
        f.write(f"Total tickers descargados: {len(df)}\n")
        f.write(f"Large cap (≥{MIN_MARKET_CAP_USD/1e9:.0f}B USD): {len(df_large)}\n\n")
        f.write("POR PAÍS (top 20):\n")
        for c, n in df_large["Country"].value_counts().head(20).items():
            f.write(f"   {c:<30} {n:>4}\n")
        f.write("\nPOR SECTOR:\n")
        for s, n in df_large["Sector"].value_counts().items():
            f.write(f"   {s:<30} {n:>4}\n")
        f.write("\nTOP 30 POR MARKET CAP:\n")
        for _, r in df_large.head(30).iterrows():
            f.write(f"   {str(r['Name'])[:35]:<37} {r['Market_Cap_USD']/1e9:>7.0f}B  "
                    f"{str(r['Country'])[:15]:<17} {str(r['Sector'])[:20]}\n")

    print(f"\n[4] Guardado:")
    print(f"   Datos:   {OUTPUT_CSV}")
    print(f"   Reporte: {REPORT_FILE}")
    print(f"\nPor país (top 10):")
    for c, n in df_large["Country"].value_counts().head(10).items():
        print(f"   {c:<25} {n:>4}")
    print(f"\nPor sector:")
    for s, n in df_large["Sector"].value_counts().items():
        print(f"   {s:<25} {n:>4}")


if __name__ == "__main__":
    main()
