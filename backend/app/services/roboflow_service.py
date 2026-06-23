import httpx
from ..core.config import settings


class RoboflowService:
    def status(self, active_provider="LocalYOLOProvider"):
        if not settings.roboflow_configured:
            return {
                "configured": False, "workspace": None, "available_projects": [],
                "model_versions": [], "active_provider": active_provider,
                "message": "Roboflow credentials not found. Live inference is running locally with YOLOv8.",
            }
        projects = []
        versions = []
        for kind in ("vehicle", "plate", "behavior"):
            project = getattr(settings, f"roboflow_project_{kind}")
            version = getattr(settings, f"roboflow_version_{kind}")
            if project:
                projects.append(kind)
                versions.append({"model_type": kind, "project": project, "version": version or "not set"})
        return {
            "configured": True, "workspace": settings.roboflow_workspace,
            "available_projects": projects, "model_versions": versions,
            "active_provider": active_provider,
            "message": "Roboflow is configured; live inference is currently using local YOLOv8",
        }

    def _endpoint(self, model_type: str):
        project = getattr(settings, f"roboflow_project_{model_type}")
        version = getattr(settings, f"roboflow_version_{model_type}")
        if not all((settings.roboflow_configured, project, version)):
            return None
        return f"https://detect.roboflow.com/{project}/{version}"

    async def infer(self, image: bytes, model_type: str):
        url = self._endpoint(model_type)
        if not url:
            return None
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(url, params={"api_key": settings.roboflow_api_key}, content=image)
            response.raise_for_status()
            return response.json()

    def infer_sync(self, image: bytes, model_type: str):
        """Blocking hosted inference for use inside the provider worker thread."""
        url = self._endpoint(model_type)
        if not url:
            return None
        with httpx.Client(timeout=20) as client:
            response = client.post(url, params={"api_key": settings.roboflow_api_key}, content=image)
            response.raise_for_status()
            return response.json()


roboflow_service = RoboflowService()
