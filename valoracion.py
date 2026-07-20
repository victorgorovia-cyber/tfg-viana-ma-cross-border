"""
TFG - M&A Deal Success Prediction
=================================
valoracion.py — Valoración de targets y precio recomendado (Función 4)

Reconstruido según la descripción del TFG (TFG_VictorV_Atilano.docx):
"valoración por tres métodos: descuento de flujos de caja (DCF) con WACC por
CAPM, múltiplos sectoriales EV/EBITDA y modelo de descuento de dividendos
(DDM), con prima M&A diferenciada por sector."

Salida: precio recomendado a pagar por acción / equity value, con desglose
por los tres métodos y la prima de control aplicada.

Los tres métodos se promedian para obtener el valor standalone; sobre él se
aplica la prima M&A del sector para obtener el precio máximo defendible.
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# PARÁMETROS DE MERCADO (CAPM) Y PRIMAS SECTORIALES
# ------------------------------------------------------------------ #
RF = 0.04          # Tipo libre de riesgo (bono 10Y EE.UU. aprox.)
ERP = 0.055        # Equity Risk Premium (prima de riesgo de mercado, Damodaran)
TAX_RATE = 0.25    # Tipo impositivo efectivo medio

# Múltiplos EV/EBITDA medianos por sector (Yahoo Finance / Damodaran data)
EV_EBITDA_SECTOR = {
    "Technology":             22.0,   # calibrado a transacciones tech (caso NICE del TFG)
    "Communication Services": 9.5,
    "Healthcare":             14.0,
    "Financial Services":     10.0,
    "Consumer Cyclical":      11.0,
    "Consumer Defensive":     12.0,
    "Industrials":            12.5,
    "Energy":                 6.0,
    "Basic Materials":        7.5,
    "Real Estate":            16.0,
    "Utilities":              10.5,
    "_default":               11.0,
}

# Prima M&A diferenciada por sector (rango Damodaran 20-40%)
PRIMA_MA_SECTOR = {
    "Technology":             0.30,   # mayor competencia por activos estratégicos
    "Healthcare":             0.32,   # pipeline / patentes
    "Communication Services": 0.25,
    "Financial Services":     0.20,   # primas más contenidas (regulación)
    "Consumer Cyclical":      0.25,
    "Consumer Defensive":     0.22,
    "Industrials":            0.25,
    "Energy":                 0.22,
    "Basic Materials":        0.22,
    "Real Estate":            0.20,
    "Utilities":              0.20,
    "_default":               0.25,
}


# ------------------------------------------------------------------ #
# 1) DCF con WACC por CAPM
# ------------------------------------------------------------------ #
def coste_equity_capm(beta: float) -> float:
    """Re = Rf + Beta · ERP."""
    return RF + (beta if pd.notna(beta) else 1.0) * ERP


def wacc(market_cap: float, total_debt: float, beta: float,
         coste_deuda: float = 0.05) -> float:
    """WACC = E/V·Re + D/V·Rd·(1-t)."""
    E = max(market_cap or 0, 1)
    D = max(total_debt or 0, 0)
    V = E + D
    re = coste_equity_capm(beta)
    return (E / V) * re + (D / V) * coste_deuda * (1 - TAX_RATE)


def dcf(fcf_inicial: float, beta: float, market_cap: float, total_debt: float,
        debt_neto: float, crecimiento: float = 0.08, g_terminal: float = 0.025,
        n_anios: int = 10) -> dict:
    """
    DCF a n años con crecimiento decreciente + valor terminal (Gordon).
    WACC calculado por CAPM. Devuelve equity value.
    """
    if fcf_inicial is None or fcf_inicial <= 0:
        return {"metodo": "DCF", "equity_value": np.nan, "valido": False}

    w = wacc(market_cap, total_debt, beta)
    if w <= g_terminal:
        w = g_terminal + 0.02   # salvaguarda

    # Crecimiento que converge linealmente de 'crecimiento' a g_terminal
    fcfs, pv = [], []
    fcf = fcf_inicial
    for t in range(1, n_anios + 1):
        g_t = crecimiento - (crecimiento - g_terminal) * (t - 1) / (n_anios - 1)
        fcf = fcf * (1 + g_t)
        fcfs.append(fcf)
        pv.append(fcf / (1 + w) ** t)

    tv = fcfs[-1] * (1 + g_terminal) / (w - g_terminal)
    pv_tv = tv / (1 + w) ** n_anios
    ev = sum(pv) + pv_tv
    equity = ev - (debt_neto or 0)

    return {"metodo": "DCF", "wacc": w, "enterprise_value": ev,
            "equity_value": max(equity, 0), "pct_terminal": pv_tv / ev, "valido": True}


# ------------------------------------------------------------------ #
# 2) Múltiplos EV/EBITDA sectoriales
# ------------------------------------------------------------------ #
def multiplos(ebitda: float, sector: str, debt_neto: float) -> dict:
    if ebitda is None or ebitda <= 0:
        return {"metodo": "EV/EBITDA", "equity_value": np.nan, "valido": False}
    mult = EV_EBITDA_SECTOR.get(sector, EV_EBITDA_SECTOR["_default"])
    ev = ebitda * mult
    equity = ev - (debt_neto or 0)
    return {"metodo": "EV/EBITDA", "multiplo": mult, "enterprise_value": ev,
            "equity_value": max(equity, 0), "valido": True}


# ------------------------------------------------------------------ #
# 3) DDM (Gordon)
# ------------------------------------------------------------------ #
def ddm(dividend_yield: float, market_cap: float, beta: float,
        revenue_growth: float = 0.08) -> dict:
    """
    DDM: g_div = min(crecimiento_ingresos · 50%, 5%); Re por CAPM.
    Valor = D1 / (Re - g).
    """
    if not dividend_yield or dividend_yield <= 0:
        return {"metodo": "DDM", "equity_value": np.nan, "valido": False}
    re = coste_equity_capm(beta)
    g = min((revenue_growth or 0) * 0.5, 0.05)
    if re <= g:
        re = g + 0.02
    d0 = dividend_yield * market_cap          # dividendo total último año
    equity = d0 * (1 + g) / (re - g)
    return {"metodo": "DDM", "coste_equity": re, "g_dividendo": g,
            "equity_value": max(equity, 0), "valido": True}


# ------------------------------------------------------------------ #
# VALORACIÓN COMPLETA + PRECIO RECOMENDADO
# ------------------------------------------------------------------ #
def valorar(target: dict) -> dict:
    """
    target: dict con Market_Cap, Revenue, Beta, EBITDA, Free_Cash_Flow,
            Total_Debt, Total_Cash, Dividend_Yield, Revenue_Growth, Sector.
    Devuelve valoración por 3 métodos, standalone promedio, prima y precio final.
    """
    mc   = target.get("Market_Cap")
    debt = target.get("Total_Debt") or 0
    cash = target.get("Total_Cash") or 0
    debt_neto = debt - cash
    sector = target.get("Sector", "_default")

    r_dcf  = dcf(target.get("Free_Cash_Flow"), target.get("Beta"), mc, debt, debt_neto,
                 crecimiento=target.get("Revenue_Growth") or 0.08)
    r_mult = multiplos(target.get("EBITDA"), sector, debt_neto)
    r_ddm  = ddm(target.get("Dividend_Yield"), mc, target.get("Beta"),
                 target.get("Revenue_Growth") or 0.08)

    # Standalone = promedio de los métodos válidos
    validos = [r for r in [r_dcf, r_mult, r_ddm] if r.get("valido")]
    if not validos:
        # Fallback: usar market cap como referencia
        standalone = mc
    else:
        standalone = float(np.mean([r["equity_value"] for r in validos]))

    prima = PRIMA_MA_SECTOR.get(sector, PRIMA_MA_SECTOR["_default"])
    precio_recomendado = standalone * (1 + prima)
    pct_sobre_cap = (precio_recomendado / mc - 1) * 100 if mc else np.nan

    return {
        "sector": sector,
        "dcf_equity":       r_dcf.get("equity_value"),
        "multiplos_equity": r_mult.get("equity_value"),
        "ddm_equity":       r_ddm.get("equity_value"),
        "standalone":       standalone,
        "prima_ma_pct":     prima * 100,
        "precio_recomendado": precio_recomendado,
        "market_cap":       mc,
        "pct_sobre_cap":    pct_sobre_cap,
        "metodos_validos":  [r["metodo"] for r in validos],
    }


if __name__ == "__main__":
    # Ejemplo: target tecnológico
    demo = {
        "Market_Cap": 12_000_000_000, "Revenue": 3_000_000_000,
        "Beta": 1.15, "EBITDA": 800_000_000, "Free_Cash_Flow": 500_000_000,
        "Total_Debt": 1_000_000_000, "Total_Cash": 2_000_000_000,
        "Dividend_Yield": 0.01, "Revenue_Growth": 0.12, "Sector": "Technology",
    }
    r = valorar(demo)
    print("=" * 55)
    print("VALORACIÓN DE TARGET — Función 4")
    print("=" * 55)
    print(f"  Sector:              {r['sector']}")
    print(f"  DCF (WACC/CAPM):     {r['dcf_equity']/1e9:>8.2f}B" if pd.notna(r['dcf_equity']) else "  DCF: n/a")
    print(f"  EV/EBITDA sectorial: {r['multiplos_equity']/1e9:>8.2f}B" if pd.notna(r['multiplos_equity']) else "  EV/EBITDA: n/a")
    print(f"  DDM:                 {r['ddm_equity']/1e9:>8.2f}B" if pd.notna(r['ddm_equity']) else "  DDM: n/a")
    print(f"  ── Standalone (avg): {r['standalone']/1e9:>8.2f}B")
    print(f"  + Prima M&A {r['prima_ma_pct']:.0f}%:      ")
    print(f"  ★ PRECIO RECOMENDADO:{r['precio_recomendado']/1e9:>8.2f}B  ({r['pct_sobre_cap']:+.1f}% vs cap)")
    print(f"  Market Cap actual:   {r['market_cap']/1e9:>8.2f}B")
