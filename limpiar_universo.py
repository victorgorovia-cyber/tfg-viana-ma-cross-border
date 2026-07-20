"""
Limpia el universo_empresas.csv convirtiendo Revenue de moneda local a USD.

Yahoo Finance devuelve Market_Cap en USD para ADRs y empresas listadas en NYSE/NASDAQ,
pero Revenue viene en moneda de reporte del emisor (yenes, wons, pesos, euros, etc.).

Este script:
1. Detecta el país de cada empresa
2. Aplica el tipo de cambio a USD medio de 2024
3. Guarda universo_empresas_USD.csv con Revenue en USD
4. Marca la columna original como Revenue_LC (local currency)
"""
import pandas as pd

# Tipos de cambio moneda_local → USD (medias 2024, fuente: FRED/ECB)
FX_TO_USD = {
    "Argentina":     1/900,     # ARS: 1 USD = 900 ARS aprox media 2024
    "Australia":     0.66,      # AUD
    "Austria":       1.09,      # EUR
    "Belgium":       1.09,      # EUR
    "Bermuda":       1.0,       # BMD = USD
    "Brazil":        0.20,      # BRL
    "Canada":        0.73,      # CAD
    "Cayman Islands":1.0,       # KYD ≈ USD
    "Chile":         1/945,     # CLP
    "China":         0.14,      # CNY
    "Denmark":       0.146,     # DKK
    "Finland":       1.09,      # EUR
    "France":        1.09,      # EUR
    "Germany":       1.09,      # EUR
    "Greece":        1.09,      # EUR
    "Hong Kong":     0.128,     # HKD
    "Hungary":       0.0028,    # HUF
    "India":         0.012,     # INR
    "Indonesia":     1/15750,   # IDR
    "Ireland":       1.09,      # EUR
    "Isle of Man":   1.27,      # GBP
    "Israel":        0.27,      # ILS
    "Italy":         1.09,      # EUR
    "Japan":         0.0066,    # JPY (1 yen ~ 0.0066 USD)
    "Jersey":        1.27,      # GBP
    "Kuwait":        3.26,      # KWD
    "Luxembourg":    1.09,      # EUR
    "Malaysia":      0.22,      # MYR
    "Mexico":        0.058,     # MXN
    "Netherlands":   1.09,      # EUR
    "New Zealand":   0.60,      # NZD
    "Norway":        0.093,     # NOK
    "Peru":          0.27,      # PEN
    "Philippines":   0.018,     # PHP
    "Poland":        0.25,      # PLN
    "Portugal":      1.09,      # EUR
    "Romania":       0.22,      # RON
    "Russia":        0.011,     # RUB
    "Singapore":     0.75,      # SGD
    "South Africa":  0.055,     # ZAR
    "South Korea":   0.00075,   # KRW (1 won ~ 0.00075 USD)
    "Spain":         1.09,      # EUR
    "Sweden":        0.096,     # SEK
    "Switzerland":   1.13,      # CHF
    "Taiwan":        0.031,     # TWD
    "Thailand":      0.028,     # THB
    "Turkey":        0.031,     # TRY
    "United Arab Emirates": 0.272,  # AED
    "United Kingdom":1.27,      # GBP
    "United States": 1.0,       # USD
    "Vietnam":       1/25400,   # VND
}

def limpiar(path_entrada="universo_empresas.csv",
            path_salida="universo_empresas.csv"):
    u = pd.read_csv(path_entrada)
    print(f"Universo cargado: {len(u)} empresas")

    # Guardar Revenue original como Revenue_LC (local currency)
    u["Revenue_LC"] = u["Revenue"].copy()

    # Convertir Revenue a USD si el país está en el mapa
    def convertir(row):
        country = row["Country"]
        rev = row["Revenue"]
        if pd.isna(rev) or rev == 0:
            return rev
        fx = FX_TO_USD.get(country)
        if fx is None:
            return rev  # Sin FX conocido, deja el valor original
        # Detectar si necesita conversión: si ratio Revenue/Cap > 3 asumimos moneda local
        # (empresas USA tienen ratios <2 típicamente)
        cap = row["Market_Cap"]
        if cap and cap > 0:
            ratio = rev / cap
            if ratio > 3:
                # Casi seguro está en moneda local
                return rev * fx
        return rev  # Si el ratio es razonable, dejamos el valor tal cual

    u["Revenue"] = u.apply(convertir, axis=1)

    # También aplicar a EBITDA, FCF, Total_Debt, Total_Cash si están mal
    for col in ["EBITDA", "Free_Cash_Flow", "Total_Debt", "Total_Cash"]:
        if col not in u.columns:
            continue
        def convertir_col(row, c=col):
            v = row[c]
            if pd.isna(v) or v == 0:
                return v
            country = row["Country"]
            fx = FX_TO_USD.get(country)
            if fx is None:
                return v
            cap = row["Market_Cap"]
            if cap and cap > 0:
                ratio = abs(v) / cap
                if ratio > 3:
                    return v * fx
            return v
        u[col] = u.apply(convertir_col, axis=1)

    # Comprobación
    print(f"\n=== Después de la conversión ===")
    print("Top 10 por Revenue:")
    top = u.nlargest(10, "Revenue")[["Name","Country","Market_Cap","Revenue"]].copy()
    top["Market_Cap"] = (top["Market_Cap"]/1e9).round(1)
    top["Revenue"] = (top["Revenue"]/1e9).round(1)
    print(top.to_string(index=False))

    # Contar sospechosos remanentes
    ratio = u["Revenue"] / u["Market_Cap"]
    sospechosos = (ratio > 5).sum()
    print(f"\nEmpresas con ratio Revenue/Cap > 5 (posiblemente aún mal): {sospechosos}")

    u.to_csv(path_salida, index=False)
    print(f"\n✓ Guardado en {path_salida}")

if __name__ == "__main__":
    limpiar()
