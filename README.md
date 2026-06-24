# QuisMotion · TEKNOFEST 2026

Analítica de seguridad vial con IA en el borde. Un flujo de video en vivo se analiza para detectar el vehículo, seguirlo, leer su matrícula, estimar la velocidad respecto al límite publicado, marcar conductas peligrosas del conductor (fumar / uso del móvil), contar ocupantes y calcular una puntuación de riesgo ponderada — mientras una señal de **QoD** impulsada por IA aumenta el ancho de banda de red (480p→1080p) exactamente cuando el vehículo se acerca.

Este repositorio es el **modelo de IA de Etapa 1 + pipeline de inferencia** descrito en el PDR (`QuisMotion PDR Submission`). La verificación de número Turkcell y QoD están simulados a propósito (pertenecen a la Etapa 2). La app web React aquí es un **banco de pruebas** para ejercitar el pipeline desde el navegador; la app operativa final es React Native (Etapa 2).

---

## 1. Arquitectura

```
┌─────────────────────┐   frames JPEG @10 FPS por WSS    ┌──────────────────────────┐
│  Frontend (React)   │ ───────────────────────────────► │  Backend (FastAPI)       │
│  banco de pruebas   │ ◄─────────────────────────────── │  + pipeline IA  :8000    │
│  :5173              │      JSON de detección por frame │  inferencia GPU (CUDA)   │
│  dibuja cajas/riesgo│                                  │                          │
└─────────────────────┘                                   └────────────┬─────────────┘
                                                                        │
                                              ┌─────────────────────────┼───────────────────────┐
                                              ▼                         ▼                       ▼
                                       YOLOv8x + DeepSORT        EasyOCR (matrícula/señal)  DuckDB + VSS
                                       (detección + seguimiento) conducta / ocupantes     (vectores de incidentes)
```

Una cola acotada por sesión (máx. 2) descarta frames obsoletos bajo carga para mantener el flujo ágil; las subtareas pesadas (OCR, conducta) se limitan en frecuencia y no se ejecutan en cada frame.

## 2. Pipeline de IA — etapa PDR vs. estado actual

| Etapa PDR | Estado | Notas |
|---|---|---|
| Detección + seguimiento de vehículo (YOLOv8x + DeepSORT) | ✅ Funciona en GPU | tiempo real, IDs de seguimiento persistentes |
| Disparador QoD (área del bbox > 15% del frame) | ✅ Funciona | bucle cerrado impulsado por IA, sube de 480p a 1080p |
| Velocidad vs. límite publicado (crecimiento del bbox + OCR de señal) | ✅ Funciona | límite leído con EasyOCR; si no, valor por defecto |
| Matrícula (detección + EasyOCR, votación multi-frame) | ⚠️ Limitado por resolución | clips demo a 464p → matrícula ≈30 px; ver notebook 02 |
| Conducta del conductor: fumar / móvil (YOLOv8s-cls/detect) | ⚙️ Requiere entrenamiento de dominio | modelos genéricos no transfieren; ver notebook 03 |
| Detección de ocupantes (conductor + pasajeros) | ✅ Conductor fiable | recorte de cabina + upscale; separación de pasajeros limitada a 464p |
| Agregación de riesgo (JSON ponderado) | ✅ Funciona | zigzag 30 / fumar 25 / exceso velocidad 25 / móvil 20 |
| Almacén de incidentes + similitud (DuckDB + VSS) | ✅ Funciona | embeddings YOLO 512-d, búsqueda coseno |
| Exportación ONNX + FP16/INT8 | ✅ La exportación funciona | aceleración GPU vía PyTorch `.pt`; ver nota ONNX abajo |

## 3. Hallazgos clave (reproducidos en `notebooks/`)

Cada afirmación está respaldada por un notebook **ejecutado** con su salida y gráficos:

- **La GPU lo hace en tiempo real.** YOLOv8x en la RTX 4060 corre ~24 ms/frame (~40 FPS), muy por debajo del objetivo de 100 ms del PDR. YOLOv8s en CPU ~101 ms. → `01_…`
- **La matrícula es un muro de resolución, no un fallo del modelo.** A 832×464 la matrícula mide ~30 px; EasyOCR es inconsistente y un detector público dedicado de matrículas devuelve cero. El upscale es interpolación — no aporta detalle nuevo. La solución es metraje genuinamente de mayor resolución (el flujo 1080p de QoD está diseñado para capturarlo). → `02_…`
- **Fumar/móvil requieren un modelo entrenado en este dominio.** La conducta es visible para un humano, pero COCO no tiene clase "cigarrillo" y el dataset público citado (primeros planos brillantes en cabina) devuelve 0 detecciones en metraje nocturno/lateral. → `03_…`
- **Los ocupantes son detectables** recortando y escalando el vehículo (el conductor sentado es invisible en el frame completo). → `03_…`
- **DuckDB + VSS** ofrece recuperación semántica real de incidentes sobre embeddings YOLO. → `04_…`

## 4. Inicio rápido

Requisitos: Python 3.11+, Node 20+, GPU NVIDIA opcional.

### Backend
```powershell
cd backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env       # luego editar (ver §7)
uvicorn app.main:app --reload --port 8000
```
Docs: http://localhost:8000/docs · Health (muestra dispositivo/modelo activo): http://localhost:8000/api/health

### Frontend (banco de pruebas)
```powershell
cd frontend
npm install
npm run dev        # http://localhost:5173
```

### GPU (opcional pero recomendado)
El wheel estándar de PyTorch es solo CPU. Para usar una GPU NVIDIA (verificado en RTX 4060, CUDA 12.6), instala un build CUDA que coincida con la versión; luego `DEVICE=auto` lo detecta:
```powershell
pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126 torch==2.12.1 torchvision==0.27.1
pip uninstall -y onnxruntime; pip install onnxruntime-gpu nvidia-cudnn-cu12 nvidia-cublas-cu12
```

| Config (512px) | Latencia/frame | FPS |
|---|---:|---:|
| YOLOv8s — CPU | ~101 ms | ~10 |
| YOLOv8s — GPU | ~18 ms | ~55 |
| YOLOv8x — GPU (detector PDR) | ~24 ms | ~40 |
| YOLOv8x + DeepSORT — GPU | ~52 ms | ~19 |

## 5. Banco de pruebas del frontend

Abre http://localhost:5173 (el backend debe estar en marcha). Navegación superior: **En vivo**, **Incidentes**, **Modelos**, **Diagnóstico**.

1. **En vivo** — elige *Prueba 01/02/03* (o **Cargar video** / **Cámara**), pulsa **Iniciar análisis**. El navegador captura frames JPEG a 10 FPS y envía sus píxeles reales por un WebSocket. Superposiciones en vivo: caja del vehículo, ID de seguimiento, matrícula, conducta, ocupantes, puntuación de riesgo y la insignia QoD (480p ↔ 1080p).
2. **Incidentes** — incidentes persistidos en DuckDB cuando se dispara QoD o el riesgo ≥ 70.
3. **Diagnóstico** — latencia (media/p95), frames descartados, conteo QoD, proveedor.
4. **Modelos** — modelo/proveedor activo y registro de datasets.

Añade un clip demo permanente en `frontend/public/demo-videos/` y una entrada en `DEMO_VIDEOS` (`frontend/src/components/VideoPanel.tsx`); los videos ad hoc se cargan desde la UI.

## 6. Notebooks (`notebooks/`)

Cada notebook documenta y reproduce **una** conclusión, y se commitea con sus salidas y gráficos para que los informes sean visibles sin re-ejecutar. Ábrelos en Jupyter / VS Code, o regenera:

```powershell
cd backend; .\.venv\Scripts\python.exe -m ipykernel install --user --name quismotion
cd ..\notebooks
$env:ROBOFLOW_API_KEY="<tu clave>"     # solo necesaria para refrescar las celdas de inferencia hospedada
..\backend\.venv\Scripts\python.exe _build_notebooks.py
```

| Notebook | Muestra |
|---|---|
| `00_environment_and_gpu` | versiones + confirmación GPU / CUDA |
| `01_vehicle_detection_qod_and_latency` | detección, área bbox→disparador QoD, latencia CPU vs GPU |
| `02_license_plate_resolution_limit` | por qué el OCR de matrícula está limitado por la fuente 464p |
| `03_driver_behavior_and_occupants` | desajuste de dominio en conducta + detección de ocupantes funcional |
| `04_duckdb_vss_similarity` | embeddings YOLO + búsqueda de similitud DuckDB VSS |
| `05_behavior_training_curves` | curvas train vs val por época + diagnóstico de sobreajuste del modelo de conducta |
| `06_dataset_preparation` | **FDR Sec. 2** — fuentes, distribución de clases, split 70/15/15 + justificación, augmentación, referencias |
| `07_solution_testing` | **FDR Sec. 4** — Precision/Recall/F1/mAP por clase + matriz de confusión + curvas PR/F1 + FPS ("por qué confiamos") |

Las celdas de Roboflow leen `ROBOFLOW_API_KEY` del entorno — la clave nunca se guarda en un notebook. Sin ella, esas celdas imprimen el resultado documentado.

## 7. Configuración (`backend/.env`)

Ajustes clave (ver `.env.example` para todos):

```env
YOLO_MODEL_PATH=yolov8x.pt     # .pt o .onnx exportado; detector PDR = yolov8x
DEVICE=auto                    # auto -> CUDA si existe, si no CPU; o cuda:0 / cpu
EMBED_MODEL_PATH=yolov8s.pt    # embeddings 512-d para VSS (se mantiene como .pt)
BEHAVIOR_MODEL_PATH=           # pesos YOLOv8 de conducta locales; vacío = "not_observable" honesto
DEFAULT_SPEED_LIMIT=20
ROBOFLOW_API_KEY=              # clave gratuita; habilita acceso a datasets/modelos incluso en proyectos públicos
ROBOFLOW_PROJECT_BEHAVIOR=quismotion-driver-behavior
```

**Proveedores reemplazables** (patrón adaptador): `NumberVerificationService` y `QoDService` son mocks que preservan el contrato de Etapa 2; `AIProvider` es `LocalYOLOProvider` (YOLOv8x + DeepSORT + EasyOCR + ocupantes + embeddings), con `RoboflowHostedProvider` como frontera hospedada.

## 8. Conducta del conductor y pasajeros — la ruta de entrenamiento

Fumar/móvil y un conteo fiable de pasajeros requieren un modelo entrenado en este dominio. La base ya está hecha:

1. Frames extraídos y subidos al proyecto Roboflow `afrikaans/quismotion-driver-behavior`.
2. 65 frames **pre-etiquetados** con cajas de ocupantes en `tmp/cabin_dataset/` (clases: `driver, passenger, phone, cigarette`, con `data.yaml`).
3. Etiqueta las cajas restantes de `phone` / `cigarette`, luego entrena en la GPU:
   ```powershell
   cd backend
   .\.venv\Scripts\yolo detect train data=..\tmp\cabin_dataset\data.yaml model=yolov8s.pt epochs=50 imgsz=512 device=0
   # luego define BEHAVIOR_MODEL_PATH a runs/detect/train/weights/best.pt
   ```
   O entrena/sirve vía Roboflow y define `ROBOFLOW_PROJECT_BEHAVIOR` + versión.

`scripts/export_models.py` exporta cualquier modelo a ONNX (FP32 / `--half` FP16 / `--int8`). `scripts/train_behavior.py` y `scripts/upload_frames_roboflow.py` automatizan el ciclo con Roboflow.

> **Nota ONNX:** la inferencia ONNX está validada en CPU. ONNX **en GPU** vía Ultralytics + onnxruntime-gpu choca con una limitación de IO-binding, así que en GPU usa la ruta PyTorch `.pt` (ya ~24 ms). ONNX es el formato de portabilidad.

## 9. API

`GET /api/health` · `GET /api/diagnostics` · `POST /api/auth/number-verification` ·
`POST /api/qod/request` · `POST /api/qod/release` · `GET /api/incidents` ·
`GET /api/incidents/{id}` · `GET /api/incidents/{id}/similar` (DuckDB+VSS) ·
`GET /api/datasets` · `GET /api/roboflow/status` · `POST /api/roboflow/test-inference` ·
`WS /ws/session/{session_id}`

Frame WebSocket de entrada: `{"type":"frame","image":"data:image/jpeg;base64,...","source":"tekno-01"}`

## 10. Tests

```powershell
cd backend
.\.venv\Scripts\python.exe tests\test_real_videos.py
```
Ejecuta YOLOv8 sobre los tres clips y comprueba detección consistente, confianza y que el área del bbox crece a medida que el coche se acerca.

## 11. Limitaciones honestas

- **OCR de matrícula** limitado por el metraje demo a 464p; necesita 1080p / cámaras reales.
- **Fumar / móvil / conteo de pasajeros** requieren el modelo entrenado en dominio (§8).
- **Verificación de número Turkcell + QoD** son mocks (Etapa 2).
- **Identidad TOGG** — el detector etiqueta un `car`; pesos específicos TOGG son Etapa 2.
