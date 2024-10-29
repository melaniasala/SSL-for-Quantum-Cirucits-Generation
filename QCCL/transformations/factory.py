from .transforms import AddIdentityGatesTransformation, RemoveIdentityGatesTransformation, SwapControlTargetTransformation, CNOTDecompositionTransformation, ChangeOfBasisTransformation, ParallelXTransformation, ParallelZTransformation, CommuteCNOTRotationTransformation, CommuteCNOTsTransformation, SwapCNOTsTransformation
from Data.QuantumCircuitGraph import QuantumCircuitGraph

class TransformationFactory:
    @staticmethod
    def create_transformation(transformation_type: str, circuit: QuantumCircuitGraph):
        if transformation_type == "add_identity":
            return AddIdentityGatesTransformation(circuit)
        elif transformation_type == "remove_identity":
            return RemoveIdentityGatesTransformation(circuit)
        elif transformation_type == "swap_ctrl_trgt":
            return SwapControlTargetTransformation(circuit)
        elif transformation_type == "cnot_decomp":
            return CNOTDecompositionTransformation(circuit)
        elif transformation_type == "change_basis":
            return ChangeOfBasisTransformation(circuit)
        elif transformation_type == "parallel_x":
            return ParallelXTransformation(circuit)
        elif transformation_type == "parallel_z":
            return ParallelZTransformation(circuit)
        elif transformation_type == "commute_cnot_rot":
            return CommuteCNOTRotationTransformation(circuit)
        elif transformation_type == "commute_cnots":
            return CommuteCNOTsTransformation(circuit)
        elif transformation_type == "swap_cnots":
            return SwapCNOTsTransformation(circuit)
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")
