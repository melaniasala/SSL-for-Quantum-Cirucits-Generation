import random
from .transforms import CircuitTransformation, NoMatchingSubgraphsError, TransformationError
from Data.QuantumCircuitGraph import QuantumCircuitGraph
from .factory import TransformationFactory

class CompositeTransformation(CircuitTransformation):
    """Applies a sequence of transformations to a quantum circuit graph."""

    def __init__(self, qcg: QuantumCircuitGraph, transformations: list[str]):
        """
        Initialize with a QuantumCircuitGraph and a list of transformation types.
        
        Parameters:
        - qcg (QuantumCircuitGraph): The quantum circuit graph to transform.
        - transformations (list of str): List of transformation types to apply sequentially.
        """
        super().__init__(qcg)
        self.qcg = qcg # maybe this could be stored in the parent class... instead of separate circuit and graph
        self.transformations = transformations

    def apply(self):
        """Apply all transformations in sequence."""
        transformed_qcg = self.qcg
        for transformation_type in self.transformations:
            # Get the transformation instance using the factory
            transformation = TransformationFactory.create_transformation(transformation_type, transformed_qcg)
            # Apply the transformation to the graph
            try:
                transformed_qcg = transformation.apply()
            except TransformationError as e:
                raise e 
            except NoMatchingSubgraphsError as e:
                raise e

        # Return the modified quantum circuit graph
        return transformed_qcg


class RandomCompositeTransformation(CircuitTransformation):
    """Applies a specified number of random transformations to a quantum circuit graph."""

    def __init__(self, qcg: QuantumCircuitGraph, transformation_pool=None, num_transformations=2, max_trials=10):
        """
        Initialize with a QuantumCircuitGraph, a pool of transformation types, the number of transformations to apply, and a maximum number of trials.
        
        Parameters:
        - qcg (QuantumCircuitGraph): The quantum circuit graph to transform.
        - transformation_pool (list of str): List of available transformation types to choose from.
        - num_transformations (int): Number of successful transformations to apply. Default is 2.
        - max_trials (int): Maximum number of attempts to apply transformations. Default is 10.
        """
        super().__init__(qcg)
        self.qcg = qcg
        if transformation_pool is None:
            transformation_pool = [
                "add_identity", 
                "remove_identity", 
                "swap_ctrl_trgt", 
                "cnot_decomp", 
                "change_basis", 
                "parallel_x", 
                "parallel_z", 
                "commute_cnot_rot", 
                "commute_cnots", 
                "swap_cnots"
                ]
        self.transformation_pool = transformation_pool
        self.num_transformations = num_transformations
        self.max_trials = max_trials

    def apply(self):
        """Apply a fixed number of random transformations in sequence."""
        transformed_qcg = self.qcg
        successful_transformations = 0
        trials = 0

        while successful_transformations < self.num_transformations and trials < self.max_trials:
            # Randomly select a transformation type from the pool
            transformation_type = random.choice(self.transformation_pool)
            transformation = TransformationFactory.create_transformation(transformation_type, transformed_qcg)
            # print(f"Trying transformation '{transformation_type}'...")
            try:
                # Apply the transformation
                transformed_qcg = transformation.apply()
                successful_transformations += 1
                self.transformation_pool.remove(transformation_type)  # Remove the transformation from the pool
            # except NoMatchingSubgraphsError:
            #     # Print message and continue to the next trial
            #     print(f"No matching subgraph found for transformation '{transformation_type}'. Trying with another transformation...")
            # except TransformationError as e:
            #     # Print detailed error information and stop computation
            #     print(f"Failed to apply transformation '{transformation_type}' due to error: {e}")
            #     raise  # Re-raise the exception to halt the execution
            except (NoMatchingSubgraphsError, TransformationError): #as e:
                # print(f"Skipping transformation '{transformation_type}' due to error: {e}")
                continue

            trials += 1  # Increment the number of attempts

        # if successful_transformations < self.num_transformations:
        #     raise TransformationError(f"Only {successful_transformations} transformations were applied after {trials} trials.")

        # Return the modified quantum circuit graph
        return transformed_qcg


