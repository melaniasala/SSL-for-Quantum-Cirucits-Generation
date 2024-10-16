import networkx as nx
from .transforms import CircuitTransformation

class CompositeTransformation(CircuitTransformation):
    """A composite transformation that applies multiple transformations in sequence."""

    def __init__(self, graph: nx.Graph):
        super().__init__(graph)
        self.transformations = []

    def add_transformation(self, transformation: CircuitTransformation):
        """Add a transformation to the composite."""
        if isinstance(transformation, CircuitTransformation):
            self.transformations.append(transformation)
        else:
            raise ValueError("Transformation must be an instance of CircuitTransformation")

    def apply(self):
        """Apply all transformations in sequence."""
        for transformation in self.transformations:
            self.graph = transformation.apply()

        # After applying all, return the modified graph.
        return self.graph

# Usage Example:
# if __name__ == "__main__":
#     import networkx as nx
#     from transformations.add_gate import AddGateTransformation
#     from transformations.remove_gate import RemoveGateTransformation

#     # Create a NetworkX graph representing a quantum circuit
#     graph = nx.Graph()
#     # Add nodes (representing qubits) and edges (representing gates) to the graph
#     graph.add_node('q0')
#     graph.add_node('q1')
#     graph.add_edge('q0', 'H', qubit='q0')  # H gate on q0
#     graph.add_edge('q1', 'X', qubit='q1')  # X gate on q1
    
#     # Create individual transformations
#     add_gate_transformation = AddGateTransformation(graph)
#     remove_gate_transformation = RemoveGateTransformation(graph)
    
#     # Create composite transformation
#     composite = CompositeTransformation(graph)
    
#     # Add transformations to composite
#     composite.add_transformation(add_gate_transformation)
#     composite.add_transformation(remove_gate_transformation)
    
#     # Apply all transformations
#     modified_graph = composite.apply()
    
#     print(modified_graph.nodes(data=True))
#     print(modified_graph.edges(data=True))

