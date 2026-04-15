from abc import ABC, abstractmethod
import numpy as np


class BaseScenario(ABC):
    """Base class for all robustness scenario transforms."""

    name: str = "base"
    description: str = ""

    def __init__(self, **params):
        self.params = params
        self.validate_params()

    def validate_params(self):
        """Override to validate scenario-specific params."""
        pass

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the transformation to an image (H, W, C) uint8."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.params})"


SCENARIO_REGISTRY: dict[str, type[BaseScenario]] = {}


def register_scenario(cls: type[BaseScenario]) -> type[BaseScenario]:
    SCENARIO_REGISTRY[cls.name] = cls
    return cls
