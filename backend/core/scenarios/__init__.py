from backend.core.scenarios.base import SCENARIO_REGISTRY, BaseScenario
# Import all scenario modules to trigger registration
from backend.core.scenarios import blur, noise, lighting, occlusion, rotation  # noqa: F401


def get_scenario(name: str, **params) -> BaseScenario:
    if name not in SCENARIO_REGISTRY:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(SCENARIO_REGISTRY)}")
    return SCENARIO_REGISTRY[name](**params)


def list_scenarios() -> list[dict]:
    return [
        {"name": cls.name, "description": cls.description}
        for cls in SCENARIO_REGISTRY.values()
    ]


__all__ = ["SCENARIO_REGISTRY", "get_scenario", "list_scenarios", "BaseScenario"]
