from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import query, results, tools

load_dotenv()

app = FastAPI(title="PeptiScout AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router, prefix="/api")
app.include_router(tools.router, prefix="/api")
app.include_router(results.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
