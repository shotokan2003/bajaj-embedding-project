from fastapi import FastAPI
from app.api import router, alt_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Real-Time QA with Embeddings and Pinecone")
app.include_router(router, prefix="/api/v1/hackrx")
app.include_router(alt_router)  # No prefix, so /hackrx/run is available

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML at /frontend
@app.get("/frontend")
def serve_frontend():
    return FileResponse(os.path.join("static", "frontend.html"))

@app.get("/")
def root():
    return {"message": "Welcome! Use /api/v1/hackrx/upload to upload documents or /docs for API documentation."}
