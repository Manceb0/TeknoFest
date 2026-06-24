# QuisMotion · TEKNOFEST 2026 — Modelo de IA (Etapa 2 / FDR)

Sistema de **seguridad vial con IA**: a partir de video, detecta el vehículo, lo
sigue, lee su matrícula, estima velocidad vs. límite, **clasifica conductas de
riesgo del conductor** y calcula una puntuación de riesgo — mientras una señal de
**QoD** sube el ancho de banda (480p→1080p) justo cuando el vehículo se acerca.

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
Video → [frames 10 FPS] → Detección vehículo (YOLOv8x) → Seguimiento (DeepSORT)
   → Recorte cabina ─→ Conducta (fumar/móvil)  +  Ocupantes (conductor/pasajeros)
   → Recorte placa  ─→ OCR (EasyOCR)
   → Trayectoria    ─→ Zigzag (swerving)
   → Crecimiento bbox → Velocidad + disparo QoD (480p→1080p al superar 15%)
   → Riesgo ponderado (JSON) → se dibuja en la app + se guarda incidente (DuckDB+VSS)
```

**Datos → Modelo → Prueba** (lo que documentan los notebooks):
1. **Dataset**: dataset público *Distracted Driving* (open-source) re-mapeado a las
   clases del proyecto + frames de los clips de competición (dominio nocturno).
2. **Entrenamiento**: fine-tune de YOLOv8 con augmentación nocturna fuerte, en GPU.
3. **Prueba**: métricas en test (Precision/Recall/F1/mAP/FPS) + comportamiento real
   sobre los clips + límites honestos.

## 3. Qué hace y qué concluye cada notebook (`notebooks/`)

Todos están **ejecutados** (traen sus salidas y gráficas). Abrir con el kernel
**"QuisMotion (.venv)"**.

| # | Notebook | Qué hace | Conclusión |
|---|---|---|---|
| 00 | `environment_and_gpu` | versiones + GPU/CUDA | corre en RTX 4060 (CUDA) |
| 01 | `vehicle_detection_qod_and_latency` | detección de vehículo, área→QoD, latencia CPU vs GPU | detección sólida; **YOLOv8x GPU ~24 ms (40 FPS)**, tiempo real |
| 02 | `license_plate_resolution_limit` | por qué falla el OCR de placa | **muro de resolución (464p)**: a esa calidad la placa es ilegible; necesita 1080p |
| 03 | `driver_behavior_and_occupants` | conducta (fumar/móvil) + ocupantes | modelos genéricos **no transfieren** al dominio; ocupantes (conductor) sí detectable |
| 04 | `duckdb_vss_similarity` | embeddings YOLO + búsqueda por similitud | recuperación semántica de incidentes funciona |
| 05 | `behavior_training_curves` | curvas train vs val por época | **sin sobreajuste** (val en su mínimo, mAP en meseta) |
| 06 | `dataset_preparation` | fuentes, distribución de clases, split 70/15/15, referencias | dataset balanceado y justificado (FDR Sec. 2) |
| 07 | `solution_testing` | P/R/F1/mAP por clase + matriz de confusión + FPS | evidencia de testing (FDR Sec. 4) — **ver nota de fuga del cigarrillo** |
| 08 | `operational_analysis` | comportamiento real sobre los clips (timeline, tasa) | la detección es **event-based** (puntual), no continua |
| 09 | `robustification_world_sam` | YOLO-World + SAM2 para **localizar/auto-etiquetar** fumar/móvil | herramienta off-line de auto-etiquetado; pesada para el vivo |
| 10 | `sam3_concept_segmentation` | **SAM 3** (complementario), segmentación por concepto | listo; se activa al añadir `sam3.pt` (peso *gated* de Meta) |
| 11 | `preprocessing_augmentation` | pipeline de preprocesamiento + galería de augmentación | recortes correctos desde la detección + augmentación que cubre el dominio |

> **Nota de honestidad (cigarrillo).** En el notebook 07 la clase `cigarette`
> obtiene métricas muy altas, pero están **infladas por fuga de datos**: sus frames
> de train y test salen del mismo clip (tekno-01). `phone` y `safe` sí son fiables
> (miles de imágenes diversas). Detectar fumar de forma **generalizable** requiere
> más footage diverso.

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
`DEVICE=auto` la detecta. Latencia: YOLOv8x GPU ~24 ms · YOLOv8s CPU ~101 ms.

## 5. App de pruebas (qué ves en pantalla)

http://localhost:5173 → pestaña **En vivo** → elige *Prueba 01/02/03* → **Iniciar
análisis**. El navegador manda frames JPEG a 10 FPS por WebSocket y dibuja en vivo:
caja del vehículo (cian), **conductor/pasajero** (verde), **placa** (naranja),
**conducta fumar/móvil** (rojo) y el riesgo + insignia QoD (480p↔1080p).

Como la detección es intermitente, el backend **captura la mejor foto de
evidencia** por incidente — el frame con el vehículo **más cercano** donde se
detectó algo — y la guarda. En **Incidentes** se ve esa foto + matrícula, conducta,
riesgo y QoD (servida en `GET /api/incidents/{id}/snapshot`). Pestañas
**Diagnóstico** (latencia/FPS) y **Modelos** completan el banco de pruebas.

## 6. Modelos y entrenamiento

- Detección de vehículo: **YOLOv8x** (COCO).
- Seguimiento: **DeepSORT**.
- Conducta: **YOLOv8** fine-tuneado, clases `[phone, cigarette, safe]`
  (`runs/behavior_combined/weights/best.pt`, conectado vía `BEHAVIOR_MODEL_PATH`).
- Matrícula/cartel: **EasyOCR**. Incidentes: **DuckDB + VSS** (embeddings 512-d).
- Scripts: `backend/scripts/` (entrenar, fine-tune, exportar ONNX, auto-etiquetar).

## 7. Estado vs. FDR y límites honestos

| Requisito FDR | Estado |
|---|---|
| Modelo de IA + pipeline (no la app/5G) | ✅ |
| Dataset (Sec. 2) | ✅ nb 06/11 |
| Solución (Sec. 3) | ✅ README + nb 01–04 |
| Testing con métricas (Sec. 4) | ✅ nb 07/08 |
| Pesos fine-tuneados entregables | ✅ `runs/behavior_combined` |

**Límites reales (documentados, no ocultos):**
- **Matrícula**: capada por 464p → necesita footage 1080p (lo que captura el QoD).
- **Fumar**: funciona sobre el clip de demo pero **no generaliza** sin footage
  diverso (fuga de datos en la métrica del cigarrillo).
- **Conducción imprudente (tekno-03)**: vía trayectoria; mejora con más clips.
- **SAM 3 / YOLO-World**: herramientas de auto-etiquetado off-line, no para el vivo.
