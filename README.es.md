# FraudShield-RT — Detección de fraude en tiempo real, optimizada por costo financiero

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/carlosedurampz-sketch/FraudShield-RT/actions/workflows/ci.yml/badge.svg)](https://github.com/carlosedurampz-sketch/FraudShield-RT/actions/workflows/ci.yml)

[Leer la versión en inglés](README.md)

> ⚠️ **Resultados ilustrativos con datos sintéticos.** Todas las métricas
> siguientes provienen de transacciones sintéticas realistas generadas por
> este repositorio (semilla 42, tasa de fraude 0.88%). Demuestran el
> pipeline de principio a fin; **no** son afirmaciones de desempeño en el
> mundo real. Véase [Metodología](#metodología) y
> [Limitaciones](#limitaciones).

## Qué hace

FraudShield-RT es un pipeline de detección de fraude para pagos con
tarjeta y transferencias, construido alrededor del problema real: **menos
del 1% de las transacciones son fraude**, así que la métrica que importa
no es la exactitud sino el costo en pesos.

- **Variables de velocidad** (10 min, 1 h, 24 h por cuenta, dispositivo y
  comercio) calculadas **usando solo el pasado** — sin fuga temporal.
- **Modelo LightGBM** (con respaldo a XGBoost / HistGradientBoosting),
  entrenado con **división fuera de tiempo** (60/10/30) y **calibración
  isotónica**, optimizado para **PR-AUC**, no solo ROC-AUC.
- **Umbral por costo**: el umbral que minimiza el costo total (fraude no
  detectado + revisión manual + fricción por falsos rechazos), no el
  arbitrario 0.50.
- **Servicio FastAPI** (`POST /score`) que devuelve el puntaje y los
  **3 motivos principales** en milisegundos, más un **simulador de ahorro
  neto** frente a reglas estáticas.
- **Resultados (sintéticos, ilustrativos)**: PR-AUC **0.573** (azar: 0.008),
  ROC-AUC 0.906, KS 0.696; en el conjunto de prueba, el modelo cuesta
  **$241,710 MXN** frente a $285,865 de la mejor regla estática (**ahorro
  de $44,154**) y $726,700 sin controles (**ahorro de $484,989**).

```
transacciones (cronológicas)
        │
        ▼
┌─────────────────────────┐
│  src/features.py        │  variables de velocidad, solo pasado
│  10 min / 1 h / 24 h    │  por cuenta / dispositivo / comercio
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  src/train.py           │  LightGBM, división fuera de tiempo
│  + calibración isotónica│  60% entrenam. / 10% calib. / 30% prueba
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  src/cost_optimization  │  umbral que minimiza el costo total en MXN
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│  src/api/main.py        │  FastAPI POST /score
│  + src/explain.py       │  puntaje + 3 motivos principales, en ms
└─────────────────────────┘
```

## Resultados (datos sintéticos — ilustrativos)

Conjunto de datos: 60,295 transacciones sintéticas, 1 ene – 1 mar 2026,
tasa de fraude **0.88%** (530 fraudes). Conjunto de prueba (último 30%
del tiempo): 18,089 transacciones, 151 fraudes, volumen de fraude
$726,700 MXN. Cifras completas en
[`outputs/metrics.json`](outputs/metrics.json) y
[`outputs/savings_simulation.json`](outputs/savings_simulation.json).

| Métrica | Valor |
|---|---|
| **PR-AUC (prueba)** | **0.573** (línea base aleatoria: 0.008 — ~69×) |
| ROC-AUC (prueba) | 0.906 |
| Estadístico KS (prueba) | 0.696 |
| Umbral óptimo (elegido en la partición de calibración) | **0.02** |
| Costo en prueba @ umbral óptimo | **$241,710 MXN** (P 0.29, R 0.63, 95/151 fraudes detectados) |
| Costo en prueba @ umbral 0.50 | $419,370 MXN (P 0.96, R 0.47) |
| Mejor regla estática (monto > $5,000 o ≥5 txns/h) | $285,865 MXN |
| **Ahorro neto vs. la mejor regla estática** | **$44,154 MXN** en el conjunto de prueba |
| Ahorro neto vs. regla monto > $5,000 | $61,709 MXN |
| Ahorro neto vs. regla monto > $2,000 | $126,014 MXN |
| Ahorro neto vs. sin controles (aprobar todo) | $484,989 MXN |

**¿Por qué el umbral óptimo es 0.02 y no 0.50?** Por dos razones, ambas
dignas de mencionar en una entrevista: (1) las probabilidades calibradas
son bajas porque la tasa base es 0.83% — un puntaje de 0.02 ya es ~2.4×
la tasa base; (2) la matriz de costos es asimétrica — perder un fraude
de ~$4,800 cuesta tanto como ~40 falsas alarmas de $120 cada una. Un
umbral de 0.50 "se ve" limpio y pierde $177,660 más en este conjunto de
prueba. El umbral se elige en la partición de *calibración*, nunca en la
de prueba.

Variables más importantes según LightGBM: `device_age_days`,
`account_age_days`, `time_since_last_acct_min`, `hour_sin`,
`acct_avg_amt_24h` — es decir, dispositivos y cuentas nuevos, ráfagas en
el tiempo y monto frente al usual dominan, lo que coincide con el
comportamiento real del fraude con tarjeta.

### Latencia (medida en una VM compartida de 1.5 CPU — véase `outputs/latency.json`)

| Modo | media | p50 | p95 | throughput |
|---|---|---|---|---|
| Uvicorn en vivo, 200 req | 10.4 ms | **4.5 ms** | 58.2 ms | 96 req/s |
| TestClient en proceso, 500 req | 6.3–19.7 ms | 4.0–6.0 ms | 34.0–73.7 ms | 51–159 req/s |

El p50 es de un solo dígito en ms en todas las corridas; el p95 varía con
la limitación de CPU en esta VM pequeña. La primera petición tras el
arranque es más lenta (~180 ms, carga del modelo en frío). Verificado en
vivo:

```bash
curl -X POST http://127.0.0.1:8000/score -H 'Content-Type: application/json' \
  -d '{"transaction_id":"TX-LIVE-001","amount_mxn":8500,"channel":"online","hour":2,
       "device_age_days":3,"acct_count_10m":6,"acct_count_1h":8,
       "acct_amt_1h":20000,"acct_amt_24h":30000,"is_new_device":1}'
# {"transaction_id":"TX-LIVE-001","score":1.0,"risk_level":"high","decision":"block",
#  "threshold":0.02,
#  "top_reasons":["$30,000 MXN from this account in the last 24 hours",
#                 "$20,000 MXN from this account in the last hour",
#                 "High amount: $8,500 MXN"],"latency_ms":180.3}
```

## Metodología

1. **Sin fuga temporal, por construcción.** Las variables de velocidad se
   calculan con `np.searchsorted` sobre marcas de tiempo estrictamente
   anteriores (`side="left"` sobre la marca actual).
   `tests/test_features.py` verifica que la primera transacción de una
   cuenta tenga velocidad cero, que los montos de ventana excluyan la fila
   actual y que las filas futuras nunca filtren información hacia atrás.
2. **División fuera de tiempo, no aleatoria.** Entrenamiento = primer 60%
   del tiempo, calibración = siguiente 10%, prueba = último 30%. Una
   división aleatoria permitiría que el modelo memorizara cuentas y
   dispositivos e inflara las métricas.
3. **PR-AUC como métrica principal.** Con <1% de fraude, el ROC-AUC es
   complaciente (la mayoría de los negativos son trivialmente fáciles).
   El área bajo la curva de precisión-recall mide lo que le importa a
   operaciones: de las alertas generadas, cuántas son fraude y cuánto
   fraude se captura.
4. **Calibración isotónica** en la partición de calibración (implementada
   en `src/calibrated_model.py` — sklearn ≥ 1.6 eliminó
   `CalibratedClassifierCV(cv="prefit")`). Un puntaje calibrado puede
   compararse con un umbral derivado del costo; uno sin calibrar, no.
5. **Umbral por costo.** Matriz de costos (MXN, supuestos documentados —
   reestimar con el negocio): FN = monto completo de la transacción,
   FP = $120 (revisión + fricción con el cliente), TP = $60 (revisión
   manual), TN = $0. Véase `src/cost_optimization.py`.
6. **Explicabilidad.** `src/explain.py` da los 3 motivos legibles por
   caso (desviación ponderada por importancia, <1 ms — lo que usa la API)
   y una ruta SHAP de mejor esfuerzo (`shap.TreeExplainer`) para revisión
   fuera de línea cuando SHAP está instalado.

## Referencia de la API

`GET /health` devuelve `{"status": "ok", "model_loaded": bool,
"threshold": float, "features": int}`.

`POST /score` acepta una transacción en JSON y devuelve el puntaje
calibrado, el nivel de riesgo, la decisión y los 3 motivos principales:

```json
{
  "transaction_id": "TX-001",
  "amount_mxn": 8500.0,
  "channel": "online",
  "hour": 2,
  "device_age_days": 3,
  "acct_count_10m": 6,
  "acct_count_1h": 8,
  "acct_amt_1h": 20000,
  "acct_amt_24h": 30000,
  "is_new_device": 1
}
```

La validación es estricta (Pydantic): `amount_mxn` debe ser positivo y
como máximo $10,000,000 MXN; `hour` debe estar entre 0 y 23; `channel`
debe ser uno de `online`, `pos`, `ecommerce`, `mobile_app`; los conteos
de velocidad deben ser enteros no negativos; los montos de ventana, las
edades y `time_since_last_acct_min` deben ser no negativos;
`is_new_device` debe ser 0 o 1. La entrada inválida se rechaza con HTTP
422. Si el modelo aún no se ha entrenado, la API devuelve HTTP 503; un
fallo de scoring devuelve un HTTP 500 genérico sin exponer detalles
internos.

## Cómo ejecutar

```bash
# 1. Clonar e instalar (Python >= 3.10)
git clone https://github.com/carlosedurampz-sketch/FraudShield-RT.git
cd FraudShield-RT
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Todo, de principio a fin (datos -> entrenamiento -> ahorro -> latencia -> pruebas)
bash run_all.sh
# ... o paso a paso:
python -m src.generate_synthetic --n 60000 --out data/synthetic_transactions.csv
python -m src.train                        # escribe models/model.pkl, outputs/metrics.json
python scripts/simulate_savings.py         # escribe outputs/savings_simulation.json
python scripts/latency_test.py --n 500
pytest -q                                  # 16 pruebas

# 3. API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
curl http://127.0.0.1:8000/health
# Docs: http://127.0.0.1:8000/docs
```

`make run-all` hace lo mismo con el Makefile. Se incluye un `Dockerfile`
(`docker build -t fraudshield-rt . && docker run -p 8000:8000 fraudshield-rt`).

### Uso con datos reales

El proyecto funciona con **cero descargas** (datos sintéticos). Para usar
IEEE-CIS Fraud Detection o PaySim, véase
[`docs/real_data.md`](docs/real_data.md) con el mapeo de columnas, y luego:

```bash
python -m src.train --data data/ieee_train.csv
```

## Estructura del proyecto

```
FraudShield-RT/
├── README.md  README.es.md
├── requirements.txt  Dockerfile  Makefile  run_all.sh  LICENSE (MIT)
├── .github/workflows/ci.yml    # pytest en cada push (Python 3.11)
├── src/
│   ├── generate_synthetic.py   # txns sintéticos realistas, <1% fraude, semilla 42
│   ├── features.py             # variables de velocidad (10m/1h/24h), solo pasado
│   ├── train.py                # LightGBM/XGBoost/HGB, fuera de tiempo, calib. isotónica
│   ├── calibrated_model.py     # envoltorio isotónico (compatible sklearn >= 1.6)
│   ├── cost_optimization.py    # matriz de costos + búsqueda de umbral óptimo
│   ├── explain.py              # 3 motivos principales por caso (+ SHAP fuera de línea)
│   └── api/main.py             # FastAPI: GET /health, POST /score
├── scripts/
│   ├── simulate_savings.py     # modelo vs reglas estáticas, ahorro neto en MXN
│   └── latency_test.py         # latencia (en proceso o --url contra servidor en vivo)
├── tests/                      # test_features (sin fuga), test_api, test_cost
├── docs/
│   ├── real_data.md            # mapeo de columnas IEEE-CIS / PaySim
│   └── images/                 # figuras para el README (aún no hay ninguna)
├── data/                       # synthetic_transactions.csv (generado, ignorado por git)
├── models/                     # model.pkl, feature_stats.json (generados)
└── outputs/                    # metrics.json, test_predictions.csv,
                                # savings_simulation.json, latency.json
```

## Limitaciones

- **Datos sintéticos.** Los patrones de fraude aquí (noche + dispositivo
  nuevo + ráfagas) son realistas pero simplificados; una cartera real
  tiene huellas de dispositivo, riesgo de comercio, resultados de 3-D
  Secure, contracargos con retraso en la etiqueta y deriva de concepto.
  Trate cada número como ilustrativo.
- **Supuestos de costo.** FP = $120 / TP = $60 son estimaciones
  documentadas, no costos medidos de un negocio. Cámbielos en
  `src/cost_optimization.py` y el umbral óptimo se mueve — esa
  sensibilidad es el punto.
- **Las variables de velocidad en la API las provee quien llama.** En
  producción, un feature store / procesador de streams (p. ej. Flink,
  Redis) es dueño de las ventanas deslizantes; la API las acepta ya
  calculadas para que el scoring se mantenga en ms. `src/features.py`
  es la implementación batch de referencia de la misma lógica.
- **El conjunto de calibración es pequeño** (6,029 filas, 46 fraudes) —
  los pasos isotónicos son gruesos, lo que en parte explica por qué el
  umbral óptimo es bajo. Con volumen real, use una ventana de calibración
  mayor y monitoree la deriva de calibración.
- **Sin monitoreo de deriva en este repo** (PSI / manejo de retraso de
  etiquetas se cubren en el proyecto complementario CreditRisk-IFRS9).

## Autor y licencia

Creado por **Carlos Eduardo Ramírez Pérez** — Físico (UAM) enfocado en modelado cuantitativo, analítica de riesgo y ciencia de datos. Licenciado bajo la Licencia MIT.
