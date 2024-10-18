import networkx as nx
import random
from ...Data.QuantumCircuitGraph import QuantumCircuitGraph
from ...Data.data_preprocessing import build_graph_from_circuit
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, HGate, CXGate, ZGate, TGate
from nx.algorithmds.isomorphism.vf2 import DiGraphMatcher

class NoMatchingSubgraphsError(Exception):
    """Exception raised when no matching subgraphs are found for the transformation."""
    pass

class TransformationError(Exception):
    """Exception raised when there is an error during the transformation process."""
    pass


class CircuitTransformation:
    """Base class for all quantum circuit transformations represented as graphs."""

    """Base class for all quantum circuit transformations represented as graphs."""

    def __init__(self, qcg: QuantumCircuitGraph):
        self.circuit, self.graph = qcg.quantum_circuit, qcg.graph
        self.num_qubits = self.circuit.num_qubits

        self.matching_subgraphs = None  # Stores found subgraph matches
        self.pattern_subgraph = None  # Pattern subgraph to be matched
        self.replacement_subgraph = None  # Replacement subgraph

        self.gate_pool = [XGate(), HGate(), CXGate(), ZGate(), TGate()] # Pool of available gates

        self.graph_to_circuit_mapping = None  # Mapping between circuit's gate index and graph nodes

    def apply(self):
        """Apply the transformation to the circuit graph."""
        self.create_pattern_and_replacement()  # Step 1: Generate pattern and replacement subgraphs
        if self.pattern_subgraph:  # Step 2: If there is a pattern, search for matches
            self.find_matching_subgraphs()
            if not self.matching_subgraphs:  # No matches found, return None
                raise NoMatchingSubgraphsError("No matching subgraphs found for the given pattern.")

        try:
            return self.execute_transformation()  # Step 3: Attempt to apply the transformation
        except Exception as e:
            raise TransformationError(f"An error occurred during the transformation: {e}")

    def create_pattern_and_replacement(self):
        """Generate the pattern to be found and the corresponding replacement pattern."""
        raise NotImplementedError("Subclasses must implement this method.")

    def find_matching_subgraphs(self):
        """Find matching subgraphs (pattern) in the graph."""
        if self.pattern_subgraph is None:
            # No pattern means transformation applies everywhere, no need to search
            return
        else:
            # shuffle the pattern subgraphs to avoid bias
            random.shuffle(self.pattern_subgraph)
            # iterate over all possible pattern subgraphs until a match is found; if not, raise NoMatchingSubgraphsError
            for subgraph in self.pattern_subgraph:
                matcher = DiGraphMatcher(self.graph, subgraph, node_match=lambda n1, n2: n1['type'] == n2['type'])
                self.matching_subgraphs = list(matcher.subgraph_isomorphisms_iter())
                # self.matching_subgraphs is a list of dictionaries, where each dictionary maps nodes in the pattern to nodes in the graph (by node labels, graph_label:pattern_label)
                
                if self.matching_subgraphs:
                    break

            if not self.matching_subgraphs:
                raise NoMatchingSubgraphsError("No matching subgraphs found for the given pattern.")

    def execute_transformation(self):
        """Apply the transformation by replacing matched patterns in the graph."""
        raise NotImplementedError("Subclasses must implement this method.")
    

class AddIdentityGatesTransformation(CircuitTransformation):
    """Transformation to add gates equivalent to the identity operation."""

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern_and_replacement(self):
        """
        Create the replacement gates that are equivalent to identity.
        Since there's no specific pattern to match, only create the replacement.
        """
        # Choose a random gate type from the available options
        selected_gate = random.choice(self.gate_pool)

        # Select qubits based on the gate's requirements
        if selected_gate.num_qubits == 1:
            selected_qubit = [random.choice(range(self.num_qubits))]
        elif selected_gate.num_qubits == 2:
            selected_qubit = random.sample(range(self.num_qubits), 2)
        
        # Create two instances of the gate (to form an identity transformation)
        self.replacement = 2*[(selected_gate, [self.circuit.qubits[q] for q in selected_qubit])]

        # Since there's no specific pattern to match, we leave the pattern as None
        self.pattern = None

    def apply_transformation(self):
        """Apply the identity gate transformation to the circuit."""
        # Get the list of current operations in the circuit
        operations = list(self.circuit.data)

        # Choose a random position to insert the identity gate sequence
        random_position = random.randint(0, len(operations))

        # Insert the replacement gates at the chosen position
        if not self.replacement:
            raise TransformationError("No replacement gates found for the identity transformation.")
        transformed_operations = operations[:random_position] + self.replacement + operations[random_position:]

        # Create a new circuit with the transformed operations
        transformed_qc = QuantumCircuit(self.num_qubits)
        for inst, qargs, cargs in transformed_operations:
            transformed_qc.append(inst, qargs, cargs)

        # Return the new circuit graph representation
        return QuantumCircuitGraph(transformed_qc)


class RemoveIdentityGatesTransformation(CircuitTransformation):
    """Transformation to remove gates equivalent to the identity operation."""

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern_and_replacement(self):
        """
        Create the pattern to match identity gates and the corresponding replacement.
        Since we're only removing gates, the replacement is an empty list.
        """
        self.pattern_subgraph = []

        # Iterate over the available gates to create all possible identity patterns
        for gate in self.gate_pool:
            if gate.num_qubits == 1:
                # Define the pattern to match the identity gates
                pattern_subcircuit = QuantumCircuit(1)
                pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0]])
                pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0]])

                self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, construct_by_layer=False, data=False))

            if gate.num_qubits == 2:
                # Define the pattern to match the identity gates
                pattern_subcircuit = QuantumCircuit(2)
                pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
                pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

                self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, construct_by_layer=False, data=False))

        # Since we're only removing gates, the replacement is an empty list
        self.replacement = []

    def execute_transformation(self):
        """Apply the transformation by removing identity gates from the circuit."""
        # Get the list of current operations in the circuit
        operations = list(self.circuit.data)

        # Retrieve indices of gates in the matching subgraph
        matching_idxs = self.get_matching_indices()

        # Remove the identity gates from the circuit
        transformed_operations = []
        for idx, op in enumerate(operations):
            if idx not in matching_idxs:
                transformed_operations.append(op)

        # Create a new circuit with the transformed operations
        transformed_qc = QuantumCircuit(self.num_qubits)
        for inst, qargs, cargs in transformed_operations:
            transformed_qc.append(inst, qargs, cargs)

        # Return the new circuit graph representation
        return QuantumCircuitGraph(transformed_qc)
   

class CommuteCNOTsTransformation(CircuitTransformation):
    """Transformation to commute two CNOT gates in the circuit."""

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern_and_replacement(self):
        """
        Create the pattern to match two CNOT gates (with the same control or same target qubits)
        The replacement is the same as the pattern, but with the gates commuted: no need to define it,
        gates will be just switched in the circuit.
        """
        # Define the pattern to match two CNOT gates with the same control
        pattern_subcircuit = QuantumCircuit(3)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[2]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, construct_by_layer=False, data=False))

        # Define the pattern to match two CNOT gates with the same target
        pattern_subcircuit = QuantumCircuit(3)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[2], pattern_subcircuit.qubits[1]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, construct_by_layer=False, data=False))

        # Since the replacement is the same as the pattern, no need to define it
        self.replacement = None

    def execute_transformation(self):
        """Apply the transformation by removing identity gates from the circuit."""
        # Retrieve indices of gates in the matching subgraph
        matching_idxs = self.get_matching_indices()

        # Get the list of current operations in the circuit
        operations = list(self.circuit.data)
        transformed_operations = operations.copy() 

        # Check if matching_idxs contains exactly two elements
        if len(matching_idxs) == 2:
            idx1, idx2 = matching_idxs

            # Swap the operations at idx1 and idx2
            transformed_operations[idx1], transformed_operations[idx2] = transformed_operations[idx2], transformed_operations[idx1]
        else:
            raise TransformationError(f"Expected exactly two matching indices for commutations, but got {len(matching_idxs)}: {matching_idxs}")

        # Create a new circuit with the transformed operations
        transformed_qc = QuantumCircuit(self.num_qubits)
        for inst, qargs, cargs in transformed_operations:
            transformed_qc.append(inst, qargs, cargs)

        # Return the new circuit graph representation
        return QuantumCircuitGraph(transformed_qc)
    


# ------------------ Helper functions ------------------ #

def get_matching_indices(self):
    # Retrieve indices of gates in the matching subgraph
    if not self.matching_subgraphs:
        raise TransformationError("No matching subgraphs found for the identity removal transformation.")
    
    # Select randomly a matching subgraph to remove
    matching = random.choice(self.matching_subgraphs)
    
    if not self.graph_to_circuit_mapping:
        self.compute_graph_to_circuit_mapping()
    matching_idxs = [self.graph_to_circuit_mapping[node] for node in matching.keys()]
    # retrieve unique indices
    return list(set(matching_idxs))


#     def apply(self):
#         # randomly select the gates to add
#         selected_gates = random.choice([('X', 'X'), ('H', 'H'), ('Z', 'Z'), ('CNOT', 'CNOT')])

#         # if gate is single qubit
#         if selected_gates[0] in ['X', 'H', 'Z']:
#             # find locations where identity gates can be added, wich are
#             # - wires in the circuit (as tuples of nodes)
#             # - nodes that corresponds to first or last gate for a qubit in a circuit (as nodes)
#             locations = self._find_wire() + self._find_first_gate() + self._find_last_gate()

#         # if gate is two qubit
#         else:
#             # find locations where identity gates can be added, wich are
#             # - wires in the circuit (as tuples of 4 nodes)
#             # - nodes that corresponds to first or last gate for a qubit in a circuit (as nodes)
#             locations = self._find_wire_two_qubit() + self._find_first_gate_two_qubit() + self._find_last_gate_two_qubit()

#         # if no locations are found, return the original circuit
#         if not locations:
#             return self.graph

#         # randomly select a location to add the identity gates
#         selected_location = random.choice(locations)       

#         # if the selected location is a wire
#         if isinstance(selected_location, tuple):
#             self._add_gates_wire(selected_location, selected_gates)

#         # if the selected location is a node
#         else:
#             self._add_gates_begin_end(selected_location, selected_gates)

#         return self.graph
    
#     def _find_wire(self):
#         # select a random edge in the circuit
#         edge = random.choice(list(self.graph.edges))

#         # ckeck if the edge is a wire (not a edge going from a control node to a target node or vice versa)
#         if not is_same_cnot_gate(self.graph.nodes[edge[0]]['label'], self.graph.nodes[edge[1]]['label']):
#             return edge
#         else:
#             return self._find_wire()

#     def _find_first_gate(self):
#         # find the first gate for each qubit in the circuit
#         first_gates = [node for node in self.graph.nodes if not list(self.graph.predecessors(node))]

#         # if there are no first gates, choose another node
#         if not first_gates:
#             return self._find_first_gate()

#         return first_gates
    
#     def _find_last_gate(self):
#         # find the last gate for each qubit in the circuit
#         last_gates = [node for node in self.graph.nodes if not list(self.graph.successors(node))]

#         # if there are no last gates, choose another node
#         if not last_gates:
#             return self._find_last_gate()

#         return last_gates
    
#     def _find_wire_two_qubit(self):
#         # select a random edge in the circuit
#         edge = random.choice(list(self.graph.edges))

#         # if the edge is not a wire, choose another edge
#         if is_same_cnot_gate(self.graph.nodes[edge[0]]['label'], self.graph.nodes[edge[1]]['label']):
#             return self._find_wire_two_qubit()
        
#         # find cnot that surrounds the wire, which are the last cnot preceding the wire and the first cnot succeeding the wire
#         # CNOT1 --> ... --> wire --> ... --> CNOT2
#         predecessors = list(self.graph.predecessors(edge[0]))
#         successors = list(self.graph.successors(edge[1]))
#         # pred_cnot = 
#         # succ_cnot =
        


# def is_same_cnot_gate(label1, label2):
#     """
#     Check if two node labels belong to the same CNOT gate.
    
#     Args:
#         label1 (str): The first node label.
#         label2 (str): The second node label.
        
#     Returns:
#         bool: True if both labels represent the same CNOT gate, False otherwise.
#     """
#     label1 = label1.replace('_control_', '_split_').replace('_target_', '_split_').split('_split_')
#     label2 = label2.replace('_control_', '_split_').replace('_target_', '_split_').split('_split_')
    
#     return label1 == label2




