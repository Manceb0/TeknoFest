from .ai_provider import AIProvider, SessionState
from .roboflow_service import roboflow_service


class RoboflowHostedProvider(AIProvider):
    name = "RoboflowHostedProvider"

    async def process_frame(self, payload: dict, state: SessionState) -> dict:
        image = payload.get("image_bytes")
        if not image:
            raise ValueError("Roboflow provider requires image_bytes")
        return await roboflow_service.infer(image, payload.get("model_type", "vehicle"))
