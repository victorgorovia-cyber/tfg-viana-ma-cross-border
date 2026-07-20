# Universo real de empresas — Instrucciones para Víctor

Vas a generar `index_companies.csv` con datos financieros **reales** de ~700 empresas cotizadas (S&P 500 + Euro Stoxx + FTSE + Nikkei + Asia-Pac + Canadá + emergentes). Sustituye al `universo_empresas.csv` actual (que tenía los ratios financieros aproximados).

## Prerrequisitos

Python 3 y estas librerías:

```bash
pip3 install yfinance pandas
```

## Pasos

### 1. Descarga el universo (10–15 min)

Coloca `download_index_companies_v2.py` en cualquier carpeta y ejecútalo:

```bash
python3 download_index_companies_v2.py
```

Verás algo como:

```
Total tickers: ~700
  [0%] Batch 1/47...
  [5%] Batch 3/47...
  ...
```

Al terminar genera **`index_companies.csv`** en la misma carpeta.

### 2. Adaptación al formato del recomendador (1 min)

Ejecuta `preparar_universo.py` (te lo he preparado) que hace tres cosas:

1. Convierte monedas a USD (por si Yahoo devuelve euros/libras/yenes)
2. Filtra a rango sano (market cap 0,5B–500B)
3. Rellena los campos que use el recomendador con nombres consistentes

```bash
python3 preparar_universo.py
```

Genera **`universo_empresas.csv`** (mismo nombre que el actual, listo para sustituirlo).

### 3. Sustitución

Copia el nuevo `universo_empresas.csv` a la carpeta `modelos_reales/` (encima del anterior). El recomendador lo lee automáticamente:

```bash
cp universo_empresas.csv modelos_reales/
```

### 4. Verificación

```bash
cd modelos_reales
python3 uso_rapido.py
```

Deberías ver el ranking del top 5 Technology, ahora con **ROE, márgenes y deuda reales** en el score financiero — y por tanto con más diferenciación entre candidatas.

---

## Qué cambia respecto al universo anterior

| Campo | Antes (aprox) | Ahora (real) |
|---|---|---|
| Market_Cap, Revenue, Beta | ✓ real | ✓ real |
| ROE, Operating_Margin, Profit_Margin | 🟡 imputado | ✓ **real (Yahoo)** |
| EBITDA, Free_Cash_Flow | 🟡 % Revenue estimado | ✓ **real (Yahoo)** |
| Total_Debt, Total_Cash | 🟡 % Market_Cap estimado | ✓ **real (Yahoo)** |
| Debt_to_Equity | 🟡 fijo 50 | ✓ **real (Yahoo)** |
| Dividend_Yield | 🟡 fijo 0 | ✓ **real (Yahoo)** |

El recomendador y la valoración pasarán a usar datos financieros reales de las 700 empresas, no aproximaciones.

## Si algo falla

- **Rate limit de Yahoo**: si ves muchos errores, aumenta `time.sleep(0.5)` a `time.sleep(1.0)` dentro de `download_index_companies_v2.py`.
- **No genera todos los tickers**: es normal, algunos están delisted o cambiaron de nombre. Con ~500–650 reales es más que suficiente.
- **Errores de red**: reinicia y relanza. El script no tiene caché, pero es tan rápido que da igual.
