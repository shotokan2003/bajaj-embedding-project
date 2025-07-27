from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Real-Time QA with Embeddings and Pinecone")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(router, prefix="/hackrx")

# Mount static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the HTML at /frontend
@app.get("/frontend")
def serve_frontend():
    return FileResponse(os.path.join("static", "frontend.html"))

@app.get("/")
def root():
    return {"message": "Welcome! Use /hackrx/run to process documents and answer questions or /docs for API documentation."}

@app.get("/test")
def test_endpoint():
    return {"status": "API is working", "message": "Server is running correctly"}
