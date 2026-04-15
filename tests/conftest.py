import pytest
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.db.database import Base, get_db
from backend.main import app

# StaticPool ensures all connections share the same in-memory DB instance
TEST_DB_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_db():
    from backend.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_image():
    """Return a 128x128 random RGB image as numpy array."""
    return np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)


@pytest.fixture
def sample_classification_data():
    n = 50
    y_true = np.random.randint(0, 3, n).tolist()
    y_pred = np.random.randint(0, 3, n).tolist()
    y_prob = np.random.dirichlet(np.ones(3), n).tolist()
    ids = [f"img_{i}" for i in range(n)]
    return ids, y_true, y_pred, y_prob


@pytest.fixture
def sample_detection_data():
    """Generate GT and pred boxes for 10 images."""
    n = 10
    gt_boxes = []
    pred_boxes = []
    for _ in range(n):
        gt_boxes.append([{"class_id": 0, "bbox": [10, 10, 60, 60]},
                         {"class_id": 1, "bbox": [70, 70, 120, 120]}])
        pred_boxes.append([{"class_id": 0, "bbox": [12, 12, 62, 62], "score": 0.9},
                           {"class_id": 1, "bbox": [68, 68, 118, 118], "score": 0.8}])
    ids = [f"img_{i}" for i in range(n)]
    return ids, gt_boxes, pred_boxes
