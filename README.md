# Sistema predictivo de éxito en operaciones de M&A cross-border

**Trabajo Fin de Grado — Víctor Viana Atilano**
Universidad Pontificia de Comillas (ICAI/ICADE)

---

## Descripción del sistema

Esta aplicación implementa un sistema predictivo con arquitectura de tres
capas para operaciones de M&A cross-border, entrenado sobre 2.739 operaciones
del periodo 1997-2023:

1. **Scorecard** de 100 puntos con 13 variables interpretables
2. **Modelo LP** (largo plazo): CatBoost, AUC test 0,892, predice creación de
   valor a 3 años (Strategic Score > 0)
3. **Modelo CP** (corto plazo): CatBoost, AUC test 0,795, predice reacción del
   mercado en el día del anuncio (CAR_1 > 0)

El sistema integra además un recomendador de targets sobre un universo de 973
empresas cotizadas (filtrado a 394 activas) y una valoración multi-método
(DCF + EV/EBITDA + DDM).

---

## Requisitos

- Python 3.10 o superior
- Windows 10/11, macOS o Linux
- 4 GB RAM (recomendado 8 GB)
- Conexión a internet solo para la instalación inicial

---

## Instalación en Windows

### 1. Descomprimir el ZIP en el Escritorio

Doble click sobre el ZIP recibido. Se creará la carpeta `modelos_reales`.

### 2. Abrir PowerShell dentro de la carpeta

Estando dentro de `modelos_reales`:
- Mantener pulsada la tecla `Shift`
- Click derecho en un espacio vacío de la carpeta
- Seleccionar "Abrir ventana de PowerShell aquí"

### 3. Instalar dependencias (solo la primera vez, 2-3 minutos)

```
pip install -r requirements.txt
```

### 4. Ejecutar la aplicación

```
streamlit run app.py
```

Se abrirá automáticamente el navegador en `http://localhost:8501`. Si no,
copiar esa dirección manualmente en Chrome o Edge.

---

## Instalación en macOS / Linux

Con Terminal en lugar de PowerShell:

```
cd ~/Desktop/modelos_reales
pip3 install -r requirements.txt
streamlit run app.py
```

---

## Uso de la aplicación

La aplicación tiene cinco modos, accesibles desde la barra lateral izquierda:

### Modo 1 — Evaluar deal (target elegido)

Ideal para empezar. Incluye un botón **"Cargar caso del TFG"** que precarga
automáticamente el caso Amadeus IT Group con contexto macro 2018,
reproduciendo la probabilidad reportada en el TFG (~64%).

Entradas: adquirente (Market Cap, Revenue, Beta, deals previos, sector,
país), target (país, sector) y contexto macro (preset por año o
personalizable).

Salidas: probabilidad LP + probabilidad CP + cuadrante trampa/oportunidad +
análisis SHAP local + valoración multi-método + veredicto integrado +
informe descargable.

### Modo 2 — Recomendador de targets

Para cuando se tiene un adquirente pero no un target concreto. Busca en el
universo de 973 empresas cotizadas las candidatas con mejor encaje según el
score combinado (60% probabilidad LP + 40% score financiero de encaje).

### Modo 3 — Comparar dos deals

Evalúa dos operaciones en paralelo.

### Modo 4 — Modelo · SHAP · Calibración

Panel técnico con métricas, calibración por quintiles, SHAP top 15.

### Modo 5 — Validación técnica

Análisis de robustez (validación temporal, comparativa de algoritmos,
intervalos de confianza, curva de aprendizaje).

---

## Cerrar la aplicación

En la terminal donde se arrancó: `Ctrl + C`.

---

## Volver a ejecutar (segunda vez)

Ya no hace falta instalar. Basta con:

```
cd (carpeta modelos_reales)
streamlit run app.py
```

---

## Estructura de archivos

| Archivo | Contenido |
|---|---|
| `app.py` | Aplicación Streamlit principal |
| `recomendador.py` | Recomendador de targets |
| `valoracion.py` | Valoración multi-método (DCF, EV/EBITDA, DDM) |
| `validacion.py` | Constantes de análisis de robustez |
| `uso_rapido.py` | Script de demo end-to-end |
| `modelo_LP_definitivo.cbm` | Modelo LP entrenado (4,5 MB) |
| `modelo_CP_definitivo.cbm` | Modelo CP entrenado (4,5 MB) |
| `lp_def_meta.json` | Configuración del modelo LP |
| `cp_def_meta.json` | Configuración del modelo CP |
| `medianas_dataset.json` | Medianas del dataset |
| `universo_empresas.csv` | Universo de 973 empresas |
| `shap_top15_LP.csv` | SHAP top 15 del modelo LP |
| `limpiar_universo.py` | Script auxiliar |
| `preparar_universo.py` | Script auxiliar |
| `download_index_companies_v3.py` | Script auxiliar |
| `requirements.txt` | Dependencias |

Sistema autocontenido: no requiere conexión externa salvo la instalación
inicial.

---

## Problemas frecuentes

- **"python no se reconoce" (Windows)**: reinstalar Python marcando "Add to
  PATH".
- **"pip no se reconoce"**: usar `python -m pip install -r requirements.txt`.
- **Puerto 8501 ocupado**: `streamlit run app.py --server.port 8502`.
- **Firewall al arrancar**: aceptar el acceso local.
- **No se abre el navegador**: ir manualmente a `http://localhost:8501`.

---

## Contacto

Víctor Viana Atilano
Universidad Pontificia de Comillas
