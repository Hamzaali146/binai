import pytest


class TestHealth:
    def test_health_check(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestModelRegistry:
    MODEL_PAYLOAD = {
        "name": "TestModel",
        "version": "1.0.0",
        "task_type": "classification",
        "framework": "pytorch",
        "description": "Test model",
        "config": {},
    }

    def test_list_models_empty(self, client):
        r = client.get("/api/v1/models/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_create_model(self, client):
        r = client.post("/api/v1/models/", json=self.MODEL_PAYLOAD)
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "TestModel"
        assert data["version"] == "1.0.0"
        assert "id" in data

    def test_get_model(self, client):
        create = client.post("/api/v1/models/", json=self.MODEL_PAYLOAD)
        model_id = create.json()["id"]
        r = client.get(f"/api/v1/models/{model_id}")
        assert r.status_code == 200
        assert r.json()["id"] == model_id

    def test_get_model_not_found(self, client):
        r = client.get("/api/v1/models/nonexistent-id")
        assert r.status_code == 404

    def test_delete_model(self, client):
        create = client.post("/api/v1/models/", json=self.MODEL_PAYLOAD)
        model_id = create.json()["id"]
        r = client.delete(f"/api/v1/models/{model_id}")
        assert r.status_code == 204
        r2 = client.get(f"/api/v1/models/{model_id}")
        assert r2.status_code == 404


class TestDatasets:
    def test_list_datasets_empty(self, client):
        r = client.get("/api/v1/datasets/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_register_dataset_bad_path(self, client):
        payload = {
            "name": "BadDataset",
            "task_type": "classification",
            "path": "/nonexistent/path/to/dataset",
            "num_samples": 0, "num_classes": 2, "classes": [], "metadata": {},
        }
        r = client.post("/api/v1/datasets/", json=payload)
        assert r.status_code == 400


class TestScenarios:
    def test_list_scenarios(self, client):
        r = client.get("/api/v1/scenarios/")
        assert r.status_code == 200
        scenarios = r.json()
        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        for sc in scenarios:
            assert "name" in sc
            assert "description" in sc

    def test_scenario_names_include_blur(self, client):
        r = client.get("/api/v1/scenarios/")
        names = [s["name"] for s in r.json()]
        assert "gaussian_blur" in names
        assert "low_light" in names
        assert "random_occlusion" in names

    def test_unknown_scenario_params(self, client):
        r = client.get("/api/v1/scenarios/nonexistent/params")
        assert r.status_code == 404


class TestEvaluations:
    def test_list_evaluations_empty(self, client):
        r = client.get("/api/v1/evaluations/")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_create_evaluation_missing_model(self, client):
        payload = {
            "name": "Test Eval",
            "model_id": "nonexistent-model-id",
            "dataset_id": "nonexistent-dataset-id",
            "config": {},
        }
        r = client.post("/api/v1/evaluations/", json=payload)
        assert r.status_code == 404

    def test_get_evaluation_not_found(self, client):
        r = client.get("/api/v1/evaluations/nonexistent-id")
        assert r.status_code == 404
