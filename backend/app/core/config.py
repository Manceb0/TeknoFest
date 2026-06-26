from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "QuisMotion API"
    database_path: str = "quismotion.duckdb"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    ai_provider: str = "local_yolo"
    # Primary vehicle detector. PDR target is YOLOv8x; YOLOv8s is the CPU-friendly
    # default. May point to a .pt or an exported .onnx file (see scripts/export_models.py).
    yolo_model_path: str = "yolov8s.pt"
    yolo_confidence: float = 0.18
    yolo_imgsz: int = 512
    # Compute device: "auto" picks CUDA when available (e.g. RTX 4060) else CPU.
    # Force with "cuda:0" or "cpu".
    device: str = "auto"
    embedding_dim: int = 512
    # Dedicated .pt model for VSS frame embeddings (kept as PyTorch so embeddings
    # work even when the detector runs as ONNX).
    embed_model_path: str = "yolov8s.pt"
    # Dedicated license-plate detector (Koushim/yolov8-license-plate-detection).
    # When set, replaces the fixed ROI heuristic with a real YOLO-detected plate bbox,
    # giving EasyOCR a tighter crop and better text at low resolutions.
    plate_model_path: str | None = None
    # Driver-behavior classifier source, resolved in this order:
    #   1. behavior_model_path set        -> local YOLOv8s-cls .pt
    #   2. else Roboflow behavior project -> hosted inference (detect.roboflow.com)
    #   3. else                           -> honest "not_observable"
    # Path to a fine-tuned YOLOv8s-cls weights file for driver behavior.
    behavior_model_path: str | None = None
    # Default posted speed limit used until a speed sign is read via OCR.
    default_speed_limit: int = 20
    enable_deepsort: bool = True
    roboflow_api_key: str | None = None
    roboflow_workspace: str | None = None
    roboflow_project_vehicle: str | None = None
    roboflow_version_vehicle: str | None = None
    roboflow_project_plate: str | None = None
    roboflow_version_plate: str | None = None
    roboflow_project_behavior: str | None = None
    roboflow_version_behavior: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def resolved_device(self) -> str:
        """Resolve "auto" to "cuda:0" when a CUDA GPU is present, else "cpu"."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @property
    def use_gpu(self) -> bool:
        return self.resolved_device.startswith("cuda")

    @property
    def roboflow_configured(self) -> bool:
        return bool(self.roboflow_api_key and self.roboflow_workspace)

    @property
    def origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",")]


settings = Settings()
