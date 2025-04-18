from abc import ABC, abstractmethod
from typing import Dict, List, Union, Any

class Parameter(ABC):
    """
    Abstract base class for all parameter types.
    """

    @abstractmethod
    def __init__(self, default):
        self.default = default
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Abstract method to convert the parameter to a dictionary for
        serialization.
        """
        pass

class IntParameter(Parameter):
    """
    Represents an integer parameter.
    """

    def __init__(self, min: int, max: int, default: int):
        super().__init__(default)
        self.type = "int"
        self.min = min
        self.max = max
        self.default = default
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "min": self.min,
            "max": self.max,
            "default": self.default,
        }

class FloatParameter(Parameter):
    """
    Represents a float parameter.
    """

    def __init__(self, min: float, max: float, default: float):
        super().__init__(default)
        self.type = "float"
        self.min = min
        self.max = max
        self.default = default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "min": self.min,
            "max": self.max,
            "default": self.default,
        }

class StringParameter(Parameter):
    """
    Represents a string parameter.
    """

    def __init__(self, default: str):
        super().__init__(default)
        self.type = "string"
        self.default = default

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "default": self.default}

class BooleanParameter(Parameter):
    """
    Represents a boolean parameter.
    """

    def __init__(self, default: bool):
        super().__init__(default)
        self.type = "boolean"
        self.default = default
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "default": self.default}


class EnumParameter(Parameter):
    """
    Represents an enum parameter.
    """

    def __init__(self, options: List[str], default: str):
        super().__init__(default)
        self.type = "enum"
        self.options = options
        self.default = default

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "options": self.options, "default": self.default}


Parameters = Dict[str, Parameter]

class Module:
    """
    Represents a module with a name, description, and parameters.
    """

    def __init__(self, name: str, description: str, parameters: Parameters, fn):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.fn = fn

    def run(self, params, **kwargs) -> Any:
        """
        Runs the module function with the given keyword arguments.
        """
        return self.fn(params, **kwargs)
     
    def to_dict(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """
        Converts the module to a dictionary for serialization.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {k: v.to_dict() for k, v in self.parameters.items()},
        }
    
class TransformerModule(Module):
    def __init__(self, name: str, description: str, parameters: Parameters, fn):
        super().__init__(name, description, parameters, fn)
        self.parameters["active"] = BooleanParameter(True)

class Config:
    """
    Represents a configuration containing lists of different module types.
    """

    def __init__(
        self,
        data_generators: List[Module],
        distributions: List[Module],
        partitioners: List[Module],
        plots: List[Module],
    ):
        self.data_generators = data_generators
        self.distributions = distributions
        self.partitioners = partitioners
        self.plots = plots
