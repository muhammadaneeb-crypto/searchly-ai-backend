"""Start the backend:  python run.py"""
import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    s = get_settings()
    uvicorn.run("app.main:app", host=s.backend_host, port=s.backend_port)
