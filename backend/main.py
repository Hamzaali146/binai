from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from backend.config import settings
from backend.db.database import init_db
from backend.api.routes import models, datasets, evaluations, scenarios, reports
from backend.schemas.schemas import HealthResponse

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-grade platform to test, evaluate, and monitor "
        "computer vision models across multiple real-world scenarios."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(models.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(evaluations.router, prefix="/api/v1")
app.include_router(scenarios.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")


@app.on_event("startup")
async def startup():
    init_db()


@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({"message": f"Welcome to {settings.APP_NAME}", "docs": "/docs"})


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
