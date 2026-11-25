"""Run the actual self-hosted Dockerfile and common API on a Modal T4.

Deploy explicitly: modal deploy deploy/modal_container_api.py
Credentials come from MODAL_API_KEY in the local .env or caller environment;
they become API_KEY only in the running container, never in an image layer.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
if modal.is_local():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

image = modal.Image.from_dockerfile(
    ROOT / "configs" / "Dockerfile.selfhosted", context_dir=ROOT,
)
cache = modal.Volume.from_name("receipt-intelligence-model-cache", create_if_missing=True)
app = modal.App("receipt-intelligence-selfhosted-api")
api_secret = modal.Secret.from_dict({"API_KEY": os.environ.get("MODAL_API_KEY", "")})


@app.cls(
    image=image, gpu="T4", cpu=2, memory=8192,
    volumes={"/model-cache": cache}, secrets=[api_secret],
    min_containers=0, max_containers=1, scaledown_window=2,
    timeout=600, startup_timeout=600, region="us",
)
class SelfHostedAPI:
    @modal.enter()
    def load(self):
        if not os.environ.get("API_KEY", "").strip():
            raise RuntimeError("Set MODAL_API_KEY before deploying the public API")
        from app.providers.selfhosted import warmup
        warmup()
        cache.commit()

    @modal.asgi_app()
    def web(self):
        # The same app, routes, validation, auth, and response contract as Docker.
        from app.main import app as api
        return api
