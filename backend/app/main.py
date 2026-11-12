from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.core.config import get_settings
from app.db.database import Base, engine

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0",
    description="Predictive maintenance API. Demo scoring is not validated for operational use.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"])
app.include_router(router)

@app.get("/health")
def health():
    from app.api.routes import predictor
    return {"status": "ok", "model_source": predictor.source}
