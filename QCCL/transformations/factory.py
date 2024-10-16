from .transforms import AddGateTransformation
from networkx import DiGraph

class TransformationFactory:
    @staticmethod
    def create_transformation(transformation_type: str, circuit: DiGraph):
        if transformation_type == "add_gate":
            return AddGateTransformation(circuit)
        elif transformation_type == "remove_gate":
            return RemoveGateTransformation(circuit)
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")
