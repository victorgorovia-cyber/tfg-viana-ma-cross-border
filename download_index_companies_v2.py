"""
═══════════════════════════════════════════════════════════════════════
DESCARGA DE EMPRESAS — TODOS LOS MERCADOS
TFG: Modelo Predictivo de Éxito en M&A
═══════════════════════════════════════════════════════════════════════
pip3 install yfinance pandas
python3 download_index_companies_v2.py
═══════════════════════════════════════════════════════════════════════
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════
# S&P 500 — HARDCODED (top ~200 by market cap + full coverage)
# ═══════════════════════════════════════════════════════════════════════

SP500 = [
    # Mega caps
    'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','BRK-B','UNH','JNJ',
    'V','XOM','JPM','PG','MA','HD','AVGO','CVX','MRK','ABBV',
    'LLY','PEP','KO','COST','TMO','WMT','MCD','CSCO','ABT','ACN',
    'DHR','CRM','ADBE','AMD','TXN','NEE','PM','UNP','CMCSA','VZ',
    'INTC','HON','RTX','AMGN','IBM','LOW','NKE','BA','CAT','GE',
    # Large caps
    'SPGI','ISRG','INTU','BLK','AXP','DE','MDLZ','GILD','SYK','ADI',
    'BKNG','VRTX','REGN','LRCX','MMC','CI','CB','PGR','ZTS','SCHW',
    'SO','DUK','BDX','ITW','SHW','AON','CME','ICE','CL','MCK',
    'APD','EQIX','PLD','EMR','MO','NSC','GD','PNC','USB','TGT',
    'FCX','F','GM','PSA','ORLY','AZO','MNST','MELI','WDAY','SNOW',
    # Mid-large caps
    'AIG','ALL','AFL','CTSH','PAYX','PCAR','KLAC','SNPS','CDNS','FTNT',
    'DXCM','IDXX','MSCI','RMD','ILMN','ALGN','ANSS','CPRT','MKTX','POOL',
    'ODFL','WST','KEYS','TDY','TRMB','WAT','BR','ZBRA','TECH','JBHT',
    # Financials
    'GS','MS','C','WFC','BAC','AMP','TROW','STT','BK','NTRS',
    # Healthcare
    'PFE','BMY','ELV','HCA','DXCM','A','IQV','MTD','WAT','BIO',
    # Energy
    'COP','SLB','EOG','MPC','PSX','VLO','OXY','HAL','DVN','FANG',
    # Consumer
    'SBUX','YUM','CMG','DG','DLTR','ROST','TJX','EBAY','ETSY','W',
    # Tech/Cloud
    'NOW','PANW','CRWD','ZS','DDOG','NET','MDB','TEAM','HUBS','OKTA',
    # Industrial
    'LMT','NOC','GE','WM','RSG','VRSK','CSGP','ROL','FAST','NDSN',
    # REITs/Real Estate
    'AMT','CCI','O','SPG','WELL','DLR','ARE','VTR','EXR','MAA',
]

# ═══════════════════════════════════════════════════════════════════════
# EURO STOXX 50 + other major European
# ═══════════════════════════════════════════════════════════════════════

EUROPE = [
    # Germany
    'SAP.DE','SIE.DE','ALV.DE','DTE.DE','MBG.DE','BMW.DE','BAS.DE','BAYN.DE',
    'ADS.DE','MUV2.DE','IFX.DE','VOW3.DE','HEN3.DE','DB1.DE','RWE.DE',
    'FRE.DE','HEI.DE','BEI.DE','CON.DE','MTX.DE',
    # France
    'MC.PA','OR.PA','TTE.PA','SAN.PA','AIR.PA','SU.PA','BNP.PA','SAF.PA',
    'CS.PA','EL.PA','KER.PA','RMS.PA','ENGI.PA','ORA.PA','CAP.PA',
    'RI.PA','DSY.PA','SGO.PA','VIV.PA','PUB.PA',
    # Netherlands
    'ASML.AS','INGA.AS','PHIA.AS','UNA.AS','WKL.AS','RAND.AS','AD.AS',
    # Spain
    'IBE.MC','SAN.MC','TEF.MC','ITX.MC','REP.MC','BBVA.MC','FER.MC','AMS.MC','CABK.MC',
    # Italy
    'ENEL.MI','ISP.MI','UCG.MI','ENI.MI','STLAM.MI','G.MI','TIT.MI',
    # Switzerland
    'NESN.SW','NOVN.SW','ROG.SW','UBSG.SW','CSGN.SW','ABB.SW','SREN.SW','GIVN.SW',
    # Nordics
    'NOVO-B.CO','MAERSK-B.CO','CARL-B.CO','VWS.CO', # Denmark
    'ERIC-B.ST','VOLV-B.ST','ATCO-A.ST','SAND.ST','HM-B.ST','AZN.ST', # Sweden
    'EQNR.OL','DNB.OL','TEL.OL','ORK.OL', # Norway
    'NOKIA.HE','NESTE.HE','UPM.HE','SAMPO.HE', # Finland
    # Belgium
    'ABI.BR','UCB.BR','SOLB.BR','KBC.BR',
    # Ireland
    'CRH.L','LSEG.L',
]

# ═══════════════════════════════════════════════════════════════════════
# UK (FTSE 100 top)
# ═══════════════════════════════════════════════════════════════════════

UK = [
    'AZN.L','SHEL.L','HSBA.L','ULVR.L','BP.L','GSK.L','RIO.L','LSEG.L',
    'REL.L','DGE.L','AAL.L','BATS.L','PRU.L','NG.L','VOD.L','LLOY.L',
    'BARC.L','GLEN.L','RR.L','BA.L','CPG.L','ABF.L','IMB.L','WPP.L',
    'IHG.L','SSE.L','NWG.L','STAN.L','ANTO.L','EXPN.L','III.L',
    'AHT.L','SGE.L','SMDS.L','RKT.L','PSON.L',
]

# ═══════════════════════════════════════════════════════════════════════
# JAPAN (Nikkei 225 top)
# ═══════════════════════════════════════════════════════════════════════

JAPAN = [
    '7203.T','6758.T','9984.T','8306.T','6861.T','9433.T','6902.T',
    '4063.T','7267.T','8035.T','6501.T','4502.T','6098.T','9432.T',
    '6367.T','4519.T','6594.T','7741.T','4661.T','8001.T',
    '8058.T','8031.T','6301.T','7751.T','6503.T','4568.T','6273.T',
    '6981.T','3382.T','2802.T',
]

# ═══════════════════════════════════════════════════════════════════════
# ASIA-PACIFIC (ex Japan)
# ═══════════════════════════════════════════════════════════════════════

ASIA_PAC = [
    # Australia (ASX)
    'BHP.AX','CBA.AX','CSL.AX','NAB.AX','WBC.AX','ANZ.AX','WES.AX',
    'MQG.AX','FMG.AX','WOW.AX','TLS.AX','RIO.AX','ALL.AX',
    # South Korea (KOSPI)
    '005930.KS','000660.KS','035420.KS','051910.KS','006400.KS',
    # Hong Kong
    '0700.HK','9988.HK','1299.HK','0005.HK','0941.HK','2318.HK',
    # Singapore
    'D05.SI','O39.SI','U11.SI','Z74.SI',
    # India (top ADRs)
    'INFY','WIT','HDB','IBN','TTM',
]

# ═══════════════════════════════════════════════════════════════════════
# CANADA (TSX top)
# ═══════════════════════════════════════════════════════════════════════

CANADA = [
    'RY.TO','TD.TO','ENB.TO','CNR.TO','CP.TO','BMO.TO','BN.TO',
    'SHOP.TO','BCE.TO','TRI.TO','MFC.TO','SU.TO','ATD.TO','WCN.TO',
    'CSU.TO','GIB-A.TO','FNV.TO','WFG.TO','IFC.TO','L.TO',
]

# ═══════════════════════════════════════════════════════════════════════
# LATAM & EMERGING
# ═══════════════════════════════════════════════════════════════════════

EMERGING = [
    # Brazil
    'VALE','PBR','ITUB','BBD','ABEV','NU',
    # Mexico
    'AMX','CEMEX','FMX','BSMX',
    # Israel
    'CHKP','NICE','CYBR','MNDY','WIX',
    # South Africa
    'NPN.JO','SOL.JO','AGL.JO',
]

# ═══════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════

all_tickers = list(set(SP500 + EUROPE + UK + JAPAN + ASIA_PAC + CANADA + EMERGING))
print(f"Total tickers: {len(all_tickers)}")

results = []
errors = []
batch_size = 15

for i in range(0, len(all_tickers), batch_size):
    batch = all_tickers[i:i+batch_size]
    pct = (i/len(all_tickers))*100
    print(f"  [{pct:.0f}%] Batch {i//batch_size + 1}/{len(all_tickers)//batch_size + 1}...")
    
    for ticker in batch:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            mcap = info.get('marketCap', 0)
            if mcap and mcap > 0:
                country = info.get('country', '')
                
                results.append({
                    'Ticker': ticker,
                    'Name': info.get('shortName', info.get('longName', ticker)),
                    'Country': country,
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', ''),
                    'Market_Cap': mcap,
                    'Revenue': info.get('totalRevenue', 0) or 0,
                    'Beta': info.get('beta', None),
                    'PE_Ratio': info.get('trailingPE', None),
                    'Employees': info.get('fullTimeEmployees', None),
                    'Dividend_Yield': info.get('dividendYield', None),
                    'Debt_to_Equity': info.get('debtToEquity', None),
                    'ROE': info.get('returnOnEquity', None),
                    'Profit_Margin': info.get('profitMargins', None),
                    'Free_Cash_Flow': info.get('freeCashflow', None),
                    'Enterprise_Value': info.get('enterpriseValue', None),
                    'EV_EBITDA': info.get('enterpriseToEbitda', None),
                    'Price_to_Book': info.get('priceToBook', None),
                    'EBITDA': info.get('ebitda', None),
                    'Total_Debt': info.get('totalDebt', None),
                    'Total_Cash': info.get('totalCash', None),
                    'Operating_Margin': info.get('operatingMargins', None),
                    'Revenue_Growth': info.get('revenueGrowth', None),
                })
        except Exception as e:
            errors.append((ticker, str(e)))
    
    time.sleep(0.5)

# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════

df = pd.DataFrame(results)
df = df[df['Market_Cap'] > 0].sort_values('Market_Cap', ascending=False)
df.to_csv('index_companies.csv', index=False)

print(f"\n{'═'*60}")
print(f"RESULTADO")
print(f"{'═'*60}")
print(f"Empresas descargadas: {len(df)}")
print(f"Errores: {len(errors)}")
print(f"Guardado en: index_companies.csv")
print(f"\nPor país (top 15):")
for country, n in df['Country'].value_counts().head(15).items():
    print(f"  {country:<25} {n:>4}")
print(f"\nPor sector:")
for sector, n in df['Sector'].value_counts().items():
    print(f"  {sector:<25} {n:>4}")
print(f"\nTop 20 por Market Cap:")
for _, r in df.head(20).iterrows():
    print(f"  {r['Name'][:35]:<37} ${r['Market_Cap']/1e9:>8.0f}B  {r['Country']}")

if errors:
    print(f"\nTickers con error ({len(errors)}):")
    for t, e in errors[:10]:
        print(f"  {t}: {e[:60]}")
