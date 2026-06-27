# QuisMotion · TEKNOFEST 2026 — Modelo de IA (Etapa 2 / FDR)

Sistema de **seguridad vial con IA**: a partir de video, detecta el vehículo, lo
sigue, estima proximidad por crecimiento del bounding box, evalúa señales de
conducta del conductor y calcula una puntuación de riesgo. La lectura de placa
OCR y el trigger QoD se documentan como módulos experimentales / etapa posterior,
no como claims cerrados de producción.

Este repo es el **modelo de IA + pipeline de inferencia** (lo que pide el FDR).
La app móvil, Turkcell Number Verification y QoD real son **Etapa posterior**: aquí
hay una **app web de pruebas** para ejercitar el pipeline desde el navegador.

---

## 1. Los 3 casos de riesgo (taxonomía del proyecto)

Los videos de competición definen **3 casos**, cada uno tipificado por un clip:

| Clip | Caso de riesgo | Cómo se detecta |
|---|---|---|
| **tekno-01** | **Fumar** (`smoking`) | modelo de conducta en la cabina (recorte del conductor) |
| **tekno-02** | **Llamada / móvil** (`phone`) | modelo de conducta en la cabina |
| **tekno-03** | **Conducción imprudente / zigzag** (`swerving`) | trayectoria del vehículo (centro del bbox en el tiempo), no objeto |

Además: detección de vehículo, **ocupantes** (conductor/pasajeros), **matrícula** y
**velocidad**. Todo se combina en una **puntuación de riesgo** ponderada.

## 2. El proceso, de principio a fin

```
Video → [frames muestreados] → Detección vehículo (YOLOv8) → Seguimiento (DeepSORT)
   → Recorte cabina ─→ Conducta (fumar/móvil)  +  Ocupantes (conductor/pasajeros)
   → Recorte placa  ─→ OCR experimental (fast-plate-ocr / EasyOCR)
   → Trayectoria    ─→ Zigzag (swerving)
   → Crecimiento bbox → proximidad + trigger QoD / solicitud de mayor calidad
   → Riesgo ponderado (JSON) → se dibuja en la app + se guarda incidente (DuckDB+VSS)
```

**Datos → Modelo → Prueba** (lo que documentan los notebooks):
1. **Dataset**: dataset público *Distracted Driving* (open-source) re-mapeado a las
   clases del proyecto + frames de los clips de competición (dominio nocturno).
2. **Entrenamiento**: fine-tune de YOLOv8 con augmentación nocturna; GPU recomendada pero se verifica por entorno.
3. **Prueba**: métricas en test (Precision/Recall/F1/mAP/FPS) + comportamiento real
   sobre los clips + límites honestos.

## 3. Qué hace y qué concluye cada notebook (`notebooks/`)

Todos están **ejecutados** (traen sus salidas y gráficas). Abrir con el kernel
**"QuisMotion (.venv)"**.

**6 notebooks**, uno por sección del FDR. Todos **ejecutados** (traen salidas y gráficas).
Abrir con el kernel **"QuisMotion (.venv)"**.

| # | Notebook | FDR | Qué hace | Conclusión clave |
|---|---|---|---|---|
| 00 | `environment_setup` | prerequisito | versiones + verificación GPU/CUDA | la corrida actual reporta si CUDA está disponible |
| 01 | `dataset_preparation` | **Sec. 2** (20 pts) | clips, clases, augmentación, declaración de fuga de datos | 3 clips reales; fuga declarada honestamente |
| 02 | `problem_analysis` | **Sec. 3.1** | 3 casos de riesgo; comparación OCR; señal de trayectoria | OCR mejora con modelo de placas, pero queda experimental |
| 03 | `ai_architecture` | **Sec. 3.2/3.3** | pipeline completo superpuesto en los 3 clips; QoD evidenciado | todo nace del vehículo detectado; módulos son aguas abajo |
| 04 | `model_training` | **Sec. 3.3** | curvas pérdida + mAP/P/R; matriz de confusión | modelo converge; mAP@0.5 ~93% (fuga declarada) |
| 05 | `solution_testing` | **Sec. 4** (20 pts) | P/R/F1, latencia FPS y test de regresión 3 clips | phone/safe fuerte; full pipeline CPU/ONNX fallback no llega a 10 FPS todavía |

> **Honestidad sobre métricas.** Phone/safe es el claim reportable fuerte en
> dataset separado. Smoking/cigarette y OCR de placa son prototipo/escena
> conocida; requieren footage diverso y más etiquetas para generalizar.

## 4. Inicio rápido

```powershell
# Backend
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --port 8000          # http://localhost:8000/api/health

# Frontend (app de pruebas)
cd frontend
npm install
npm run dev                                # http://localhost:5173
```

GPU (recomendado): el wheel de PyTorch es CPU; para la NVIDIA instala el build CUDA:
```powershell
pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126 torch==2.12.1 torchvision==0.27.1
```
`DEVICE=auto` la detecta. El notebook `05_solution_testing` reporta la medición
real del entorno actual; si CUDA aparece como `False`, el full pipeline corre en
CPU/ONNX fallback y no debe reportarse como benchmark GPU.

## 5. App de pruebas (qué ves en pantalla)

http://localhost:5173 → pestaña **En vivo** → elige *Prueba 01/02/03* → **Iniciar
análisis**. El navegador manda frames JPEG por WebSocket y dibuja en vivo:
caja del vehículo (cian), **conductor/pasajero** (verde), **placa** (naranja),
**conducta fumar/móvil** (rojo) y el riesgo + insignia QoD (480p↔1080p).

Como la detección es intermitente, el backend **captura la mejor foto de
evidencia** por incidente — el frame con el vehículo **más cercano** donde se
detectó algo — y la guarda. En **Incidentes** se ve esa foto + matrícula, conducta,
riesgo y QoD (servida en `GET /api/incidents/{id}/snapshot`). Pestañas
**Diagnóstico** (latencia/FPS) y **Modelos** completan el banco de pruebas.

## 6. Modelos y entrenamiento

- Detección de vehículo: **YOLOv8** (COCO; variantes n/s/x según velocidad/calidad).
- Seguimiento: **DeepSORT**.
- Conducta reportable: **phone/safe** con test set separado.
- Conducta experimental: **phone/cigarette/safe** en escena conocida
  (`runs/behavior_combined/weights/best.pt`, conectado vía `BEHAVIOR_MODEL_PATH`).
- Matrícula/cartel: OCR experimental con `fast-plate-ocr` / EasyOCR. Incidentes: DuckDB + VSS opcional.
- Scripts: `backend/scripts/` (entrenar, fine-tune, exportar ONNX, auto-etiquetar).

## 7. Estado vs. FDR y límites honestos

| Requisito FDR | Estado |
|---|---|
| Modelo de IA + pipeline (no la app/5G) | ✅ |
| Dataset (Sec. 2) | ✅ `01_dataset_preparation` |
| Solución (Sec. 3) | ✅ `02_problem_analysis`, `03_ai_architecture`, `04_model_training` |
| Testing con métricas (Sec. 4) | ✅ `05_solution_testing` |
| Pesos fine-tuneados entregables | ✅ `runs/behavior_*` |

**Límites reales (documentados, no ocultos):**
- **Matrícula**: OCR mejora con `fast-plate-ocr` en ROIs válidos, pero a 464p
  sigue siendo experimental; necesita tabla frame-a-frame y más videos.
- **Fumar**: funciona sobre el clip de demo pero **no generaliza** sin footage
  diverso (fuga de datos en la métrica del cigarrillo).
- **Conducción imprudente (tekno-03)**: vía trayectoria; mejora con más clips.
- **FPS full pipeline**: en la corrida actual cae a CPU/ONNX fallback y no llega
  a 10 FPS; requiere CUDA/ONNX Runtime bien configurado, optimización o muestreo asíncrono.
- **SAM 3 / YOLO-World**: herramientas de auto-etiquetado off-line, no para el vivo.

## 8. Conceptos clave y referencias (para entender el proyecto)

Glosario corto de todo lo que se usa, para que cualquiera del equipo entienda el
README sin investigar por su cuenta:

| Término | Qué es / por qué se usa aquí | Referencia |
|---|---|---|
| **YOLOv8** (n/s/x) | Red que **detecta objetos** (vehículo, persona) en una imagen; x = más precisa, n = más rápida. | https://docs.ultralytics.com |
| **DeepSORT** | **Seguimiento**: le da un ID estable al vehículo entre frames (para medir trayectoria/zigzag). | https://github.com/levan92/deep_sort_realtime |
| **EasyOCR** | **Lee texto** de imágenes (matrícula, cartel de velocidad). | https://github.com/JaidedAI/EasyOCR |
| **YOLO-World** | Detección **open-vocabulary**: detecta por *texto* ("cigarette") sin entrenar. La usamos como auto-etiquetador. | https://docs.ultralytics.com/models/yolo-world |
| **SAM 2 / SAM 3** | **Segment Anything**: genera *máscaras* de píxeles; SAM 3 segmenta por concepto de texto. Herramienta de etiquetado. | https://docs.ultralytics.com/models/sam-3 |
| **QoD** (Quality on Demand) | API 5G de Turkcell que **sube el ancho de banda** bajo demanda; aquí la dispara la IA al acercarse el vehículo. | API Turkcell (Etapa posterior) |
| **DuckDB + VSS** | Base de datos analítica + extensión de **búsqueda vectorial** (incidentes similares por embedding). | https://duckdb.org/docs/extensions/vss |
| **Embedding** | Vector numérico (512-d) que resume una imagen; permite comparar/buscar por similitud. | — |
| **mAP / Precision / Recall / F1** | Métricas de calidad de un detector. **Precision**=de lo detectado, cuánto es correcto; **Recall**=de lo real, cuánto detecta; **F1**=balance; **mAP**=precisión media. | https://docs.ultralytics.com/guides/yolo-performance-metrics |
| **Fuga de datos** (data leakage) | Cuando train y test comparten datos casi idénticos → la métrica sale alta pero **engaña** (memoriza, no generaliza). Pasa con `cigarette` (1 solo clip). | — |
| **Brecha de dominio** (domain gap) | Un modelo entrenado en un dominio (cabina diurna) **falla** en otro (vigilancia nocturna). Por eso hay que adaptar/anotar el dominio real. | — |
| **Augmentación** | Variar artificialmente las imágenes de entrenamiento (brillo, blur, ruido…) para que el modelo **generalice**. Ver `01_dataset_preparation`. | https://docs.ultralytics.com/usage/cfg/#augmentation-settings |
| **Fine-tuning** | Re-entrenar un modelo preexistente sobre tus datos para adaptarlo a tu problema. | — |
| **ONNX / FP16 / INT8** | Formato portable + cuantización para correr el modelo **más rápido/ligero**. | https://onnx.ai |

> **Cómo leer los hallazgos del proyecto.** Muchos "límites" (placa, fumar) **no
> son fallos de programación**: son límites de **datos** (pocos/no diversos) o de
> **resolución** (464p). Cada uno está demostrado con su notebook. La forma de
> cerrarlos es conseguir **mejor footage** (1080p, diverso), no más código.
