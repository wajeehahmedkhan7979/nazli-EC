"""
Uvicorn entrypoint.
Run: python main.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1,
    )
