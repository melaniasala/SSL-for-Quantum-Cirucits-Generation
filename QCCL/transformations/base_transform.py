import random
from qiskit.circuit.library.standard_gates import XGate, HGate, CXGate, ZGate, TGate
from networkx.algorithms.isomorphism import DiGraphMatcher
from Data.QuantumCircuitGraph import QuantumCircuitGraph


class NoMatchingSubgraphsError(Exception):
    """Exception raised when no matching subgraphs are found for the transformation."""
    pass

class TransformationError(Exception):
    """Exception raised when there is an error during the transformation process."""
    pass


class CircuitTransformation:
    """Base class for all quantum circuit transformations represented as graphs."""

    def __init__(self, qcg: QuantumCircuitGraph):
        self.circuit, self.graph = qcg.quantum_circuit, qcg.graph
        self.num_qubits = self.circuit.num_qubits
        self.gate_type_map = qcg.GATE_TYPE_MAP

        self.matching_subgraph = None  # Stores found subgraph matches
        self.matching_key = None  # Key of the matching subgraph (not used for all transformations)
        self.pattern_subgraph = None  # Pattern subgraph to be matched
        self.replacement = None  # Replacement subgraph

        self.gate_pool = [XGate(), HGate(), CXGate(), ZGate(), TGate()] # Pool of available gates

        self.graph_to_circuit_mapping = None  # Mapping between circuit's gate index and graph nodes

    def apply(self):
        """Apply the transformation to the circuit graph."""
        self.create_pattern()  # Step 1: Generate pattern subgraph
        if self.pattern_subgraph:  # Step 2: If there is a pattern, search for matches
            self.find_matching_subgraph()
            if not self.matching_subgraph:  # No matches found, return None
                raise NoMatchingSubgraphsError("No matching subgraphs found for the given pattern.")

        try:
            self.create_replacement() # Step 3: Generate replacement subgraph
            return self.apply_transformation()  # Step 4: Attempt to apply the transformation
        except Exception as e:
            raise TransformationError(f"An error occurred during the transformation: {e}")

    def create_pattern(self):
        """Generate the pattern to be found."""
        raise NotImplementedError("Subclasses must implement this method.")

    def find_matching_subgraph(self):
        """Find matching subgraphs (pattern) in the graph."""
        if self.pattern_subgraph is None:
            # No pattern means transformation applies everywhere, no need to search
            return
        else:
            # shuffle the pattern subgraphs to avoid bias
            random.shuffle(self.pattern_subgraph)
            # iterate over all possible pattern subgraphs until a match is found; if not, raise NoMatchingSubgraphsError
            print(f"Pattern subgraphs: {self.pattern_subgraph}")
            for key, subgraph in self.pattern_subgraph:
                matcher = DiGraphMatcher(self.graph, subgraph, 
                                         node_match=lambda n1, n2: n1['type'] == n2['type'] and n1['ctrl_trgt'] == n2['ctrl_trgt'])
                matching_subgraphs = list(matcher.subgraph_isomorphisms_iter()) 
                matching_key = key
                # matching_subgraphs is a list of dictionaries, where each dictionary maps nodes in the pattern to nodes in the graph (by node labels, graph_label:pattern_label)
                
                if matching_subgraphs:
                    break
            
            print(f"Matching subgraphs: {matching_subgraphs}")
            print(f"Matching key: {matching_key}")

            if not matching_subgraphs:
                raise NoMatchingSubgraphsError("No matching subgraphs found for the given pattern.")
            
            # Shuffle the matching subgraphs to avoid bias and store a single match
            random.shuffle(matching_subgraphs)
            print(f"Matching subgraph selected: {matching_subgraphs[0]}")
            self.matching_subgraph = matching_subgraphs[0] 
            self.matching_key = matching_key
            
    def create_replacement(self):
        """Generate the replacement subgraph to be applied."""
        raise NotImplementedError("Subclasses must implement this method.")

    def apply_transformation(self):
        """Apply the transformation by replacing matched patterns in the graph."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_matching_indices(self):
    # Retrieve indices of gates in the matching subgraph
        if not self.matching_subgraph:
            raise TransformationError("No matching subgraphs found for the identity removal transformation.")
        
        if not self.graph_to_circuit_mapping:
            self.compute_graph_to_circuit_mapping()
        matching_idxs = [self.graph_to_circuit_mapping[node] for node in self.matching_subgraph.keys()]
        # retrieve unique indices
        return list(set(matching_idxs))
    
    def compute_graph_to_circuit_mapping(self):
        graph_to_circuit_mapping = {}

        # Initialize a graph node index tracker
        circuit_index = 0

        print(f"Graph nodes: {self.graph.nodes}")

        for node in self.graph.nodes:
            if 'cx' in node:  # Handle CNOT nodes
                if 'control' in node or 'target' in node:
                    # Map the control and target nodes to the corresponding circuit index
                    graph_to_circuit_mapping[node] = circuit_index
                    # Only move to the next circuit instruction after handling both control and target
                    if 'target' in node:
                        circuit_index += 1  # Move to the next CNOT instruction after the target node
            else:  # Single-qubit gates
                # Map the single-qubit node to the corresponding circuit index
                graph_to_circuit_mapping[node] = circuit_index
                circuit_index += 1  # Move to the next circuit instruction

        self.graph_to_circuit_mapping = graph_to_circuit_mapping