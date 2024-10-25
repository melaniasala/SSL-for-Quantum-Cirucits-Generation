import random

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from Data.data_preprocessing import build_graph_from_circuit, process_gate, insert_node, build_circuit_from_graph
from Data.QuantumCircuitGraph import QuantumCircuitGraph
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, TGate, XGate, ZGate


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
    


    
# ------------------ Identity gate transformations ------------------ #
class AddIdentityGatesTransformation(CircuitTransformation):
    """Transformation to add gates equivalent to the identity operation."""

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        pass

    def create_replacement(self):
        """
        Create the replacement gates to add to the circuit.
        Randomly select a single qubit gate from the gate pool and duplicate it to form an identity.
        """
        try:
            if not hasattr(self, 'gate_pool') or len(self.gate_pool) == 0:
                raise ValueError("The gate pool is empty or not initialized.")
            
            # Choose a random gate type from the available options
            selected_gate = random.choice(self.gate_pool)

            # Select qubits based on the gate's requirements
            if selected_gate.num_qubits == 1:
                selected_qubit = [random.choice(range(self.num_qubits))]
            elif selected_gate.num_qubits == 2:
                selected_qubit = random.sample(range(self.num_qubits), 2)
            else:
                raise ValueError(f"Unexpected number of qubits for the gate: {selected_gate.num_qubits}")
            
            # Create two instances of the gate (to form an identity transformation)
            self.replacement = 2*[(selected_gate, [self.circuit.qubits[q] for q in selected_qubit], [])]

        except Exception as e:
            raise TransformationError(f"Failed to create replacement gates: {e}")

    def apply_transformation(self):
        """Apply the identity gate transformation to the circuit."""
        try:
            # Get the list of current operations in the circuit
            operations = [(op.operation, op.qubits, op.clbits) for op in self.circuit.data]

            # Choose a random position to insert the identity gate sequence
            random_position = random.randint(0, len(operations))

            # Insert the replacement gates at the chosen position
            if not self.replacement:
                raise TransformationError("No replacement gates found for the identity transformation.")
            transformed_operations = operations[:random_position] + self.replacement + operations[random_position:]

            # Create a new circuit with the transformed operations
            transformed_qc = QuantumCircuit(self.num_qubits)
            for instruction in transformed_operations:
                inst, qargs, cargs = instruction
                transformed_qc.append(inst, qargs, cargs)

            # Return the new circuit graph representation
            return QuantumCircuitGraph(transformed_qc)
        
        except Exception as e:
            raise TransformationError(f"Failed to apply the transformation: {e}")


class RemoveIdentityGatesTransformation(CircuitTransformation):
    """Transformation to remove gates equivalent to the identity operation."""

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        """ Create the pattern to match identity gates."""
        try:
            self.pattern_subgraph = []

            # Iterate over the available gates to create all possible identity patterns
            for gate in self.gate_pool:
                if gate.num_qubits == 1:
                    # Define the pattern to match the identity gates
                    pattern_subcircuit = QuantumCircuit(1)
                    pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0]])
                    pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0]])

                    self.pattern_subgraph.append((None, build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

                if gate.num_qubits == 2:
                    # Define the pattern to match the identity gates
                    pattern_subcircuit = QuantumCircuit(2)
                    pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
                    pattern_subcircuit.append(gate, [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

                    self.pattern_subgraph.append((None, build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))
        
        except Exception as e:
            raise TransformationError(f"Error while creating pattern: {e}")


    def create_replacement(self):
        """No replacement is necessary, we are just removing identity gates."""
        pass

    def apply_transformation(self):
        """Apply the transformation by removing identity gates from the circuit."""
        try:
            # Get the list of current operations in the circuit
            operations = [(op.operation, op.qubits, op.clbits) for op in self.circuit.data]
            print(f"Current operations: {operations}")

            # Retrieve indices of gates in the matching subgraph
            matching_idxs = self.get_matching_indices()
            print(f"Graph to circuit mapping: {self.graph_to_circuit_mapping}")
            print(f"Matching indices: {matching_idxs}")

            # Remove the identity gates from the circuit
            transformed_operations = []
            for idx, op in enumerate(operations):
                if idx not in matching_idxs:
                    transformed_operations.append(op)
            print(f"Transformed operations: {transformed_operations}")

            # Create a new circuit with the transformed operations
            transformed_qc = QuantumCircuit(self.num_qubits)
            for instruction in transformed_operations:
                inst, qargs, cargs = instruction    
                transformed_qc.append(inst, qargs, cargs)

            # Return the new circuit graph representation
            return QuantumCircuitGraph(transformed_qc)
        
        except Exception as e:
            raise TransformationError(f"Failed to apply the transformation: {e}")
   



# ------------------ Commutation transformations ------------------ #
class CommuteCNOTsTransformation(CircuitTransformation):
    """Transformation to commute two CNOT gates in the circuit."""

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        """
        Create the pattern to match two CNOT gates (with the same control or same target qubits)
        """
        # Define the pattern to match two CNOT gates with the same control
        pattern_subcircuit = QuantumCircuit(3)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[2]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))

        # Define the pattern to match two CNOT gates with the same target
        pattern_subcircuit = QuantumCircuit(3)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[2], pattern_subcircuit.qubits[1]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))

    def create_replacement(self):
        """Since the replacement is the same as the pattern, no need to define it"""
        pass

    def apply_transformation(self):
        """Apply the transformation by removing identity gates from the circuit."""
        # TODO: Working with .data is not ideal for commutations, as it doesn't preserve the order 
        # of operations in the middle of the two gates to be swapped

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
        for instruction in transformed_operations:
            inst = instruction.operation  
            qargs = instruction.qubits    
            cargs = instruction.clbits    

            transformed_qc.append(inst, qargs, cargs)

        # Return the new circuit graph representation
        return QuantumCircuitGraph(transformed_qc)
    

class CommuteCNOTRotationTransformation(CircuitTransformation):
    """
    Transformation to commute a CNOT gate with a rotation gate in the circuit.
    A CNOT gate immediately preceeded/followed by a rotation gate around x-axis (X) on the target qubit can be commuted.
    The same applies for a CNOT gate immediately preceeded/followed by a rotation gate around z-axis (Z, T) on the control qubit.
    """

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        """
        Create the pattern to match a CNOT gate with a rotation gate and the corresponding replacement.
        The replacement is the same as the pattern, but with the gates commuted: no need to define it,
        gates will be just switched in the circuit.
        """
        # CX -> X pattern (common target qubit)
        pattern_subcircuit = QuantumCircuit(2)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(XGate(), [pattern_subcircuit.qubits[1]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))

        # CX -> Z pattern (common control qubit)
        pattern_subcircuit = QuantumCircuit(2)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(ZGate(), [pattern_subcircuit.qubits[0]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))

        # X -> CX pattern (common target qubit)
        pattern_subcircuit = QuantumCircuit(2)
        pattern_subcircuit.append(XGate(), [pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))

        # Z -> CX pattern (common control qubit)
        pattern_subcircuit = QuantumCircuit(2)
        pattern_subcircuit.append(ZGate(), [pattern_subcircuit.qubits[0]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

        self.pattern_subgraph.append(build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))

        # TODO: Add pattern for T -> CX and CX -> T

    def create_replacement(self):
        pass

    def apply_transformation(self):
        """Apply the transformation by commuting a CNOT gate with a rotation gate in the circuit."""
        # TODO: Working with .data is not ideal for commutations, as it doesn't preserve the order 
        # of operations in the middle of the two gates to be swapped

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
        for instruction in transformed_operations:
            inst = instruction.operation  
            qargs = instruction.qubits    
            cargs = instruction.clbits    

            transformed_qc.append(inst, qargs, cargs)

        # Return the new circuit graph representation
        return QuantumCircuitGraph(transformed_qc)


# ------------------ Equivalence transformations ------------------ #
class SwapControlTargetTransformation(CircuitTransformation):
    """
    Transformation to swap the control and target qubits of a CNOT gate in the circuit.
    Adding 4 Hadamard gates before and after the CNOT's control and target is equivalent to swapping them.
    """

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        """
        Create the pattern to match: it can be either a CNOT gate or a sequence of Hadamard gates + CNOT,
        so that both the transformation and its inverse can be applied.
        In the first element of the pattern subgraph list, the pattern is a CNOT gate, while in the second element
        the pattern is Hadamard gates + CNOT.
        """
        try:
            self.pattern_subgraph = []

            # Define the pattern to match a CNOT gate
            #  ──■──
            #  ┌─┴─┐
            #  ┤ X ├
            #  └───┘
            pattern_subcircuit = QuantumCircuit(2)
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

            self.pattern_subgraph.append(('cx',build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

            # Define the pattern to match 4 Hadamard gates + CNOT
            #  ┌───┐     ┌───┐
            #  ┤ H ├──■──┤ H ├
            #  ├───┤┌─┴─┐├───┤
            #  ┤ H ├┤ X ├┤ H ├
            #  └───┘└───┘└───┘
            pattern_subcircuit = QuantumCircuit(2)
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[0]])
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[0]])
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[1]])

            self.pattern_subgraph.append(('h-h-cx-h-h', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

            # Define the pattern to match 2 Hadamard gates (before) + CNOT
            #  ┌───┐
            #  ┤ H ├──■──
            #  ├───┤┌─┴─┐
            #  ┤ H ├┤ X ├
            #  └───┘└───┘
            pattern_subcircuit = QuantumCircuit(2)
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[0]])
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

            self.pattern_subgraph.append(('h-h-cx', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

            # Define the pattern to match CNOT + 2 Hadamard gates (after)
            #       ┌───┐
            #  ──■──┤ H ├
            #  ┌─┴─┐├───┤
            #  ┤ X ├┤ H ├
            #  └───┘└───┘
            pattern_subcircuit = QuantumCircuit(2)
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[0]])
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[1]])

            self.pattern_subgraph.append(('cx-h-h', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

            # Define the pattern to match Hadamard gate (on control) + CNOT + Hadamard gate (on control)
            #  ┌───┐     ┌───┐
            #  ┤ H ├──■──┤ H ├
            #  └───┘┌─┴─┐└───┘
            #  ─────┤ X ├─────
            #       └───┘       
            pattern_subcircuit = QuantumCircuit(2)
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[0]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(HGate(), [pattern_subcircuit.qubits[0]])

            self.pattern_subgraph.append(('h-cx-h', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

            print(f"Pattern subgraph: {self.pattern_subgraph}")

        except Exception as e:
            raise TransformationError(f"Failed to create pattern: {e}")

    def create_replacement(self):
        """
        Create the replacement subgraph to swap the control and target qubits of a CNOT gate.
        Both the transformation and its reverse can be applied: the first element of the replacement list
        contains the Hadamard gates + CNOT, while the second element contains the CNOT gate.
        """
        try:
            # Get the qubits on which the CNOT gate acts
            if not self.matching_subgraph:
                raise TransformationError("No matching subgraphs found for the control-target swap transformation.")
            
            if not self.matching_key:
                raise TransformationError("No matching key found for the control-target swap transformation.")
            
            # Retrieve the control and target qubits from the matching subgraph
            control_qubit = None
            target_qubit = None

            iter_nodes = iter(self.matching_subgraph.keys())
            # until both control and target qubits are found, iterate over the nodes in the matching subgraph
            for node in iter_nodes:
                if 'control' in node: # get control qubit and assign to target qubit for swap
                    target_qubit = int(node.split('_')[1]) # cx_{control_qubit}_control_{node_id}
                elif 'target' in node: # get target qubit and assign to control qubit for swap
                    control_qubit = int(node.split('_')[1]) # cx_{target_qubit}_target_{node_id}

                if control_qubit is not None and target_qubit is not None:
                    break

            if control_qubit is None or target_qubit is None:
                raise TransformationError("Could not find both control and target qubits in the subgraph.")

            if self.matching_key == 'cx': # CNOT -> H, H, CNOT, H, H
                replacement = []

                replacement.append((HGate(), [self.circuit.qubits[control_qubit]], []))
                replacement.append((HGate(), [self.circuit.qubits[target_qubit]], []))
                replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))
                replacement.append((HGate(), [self.circuit.qubits[control_qubit]], []))
                replacement.append((HGate(), [self.circuit.qubits[target_qubit]], []))

            elif self.matching_key == 'h-h-cx-h-h': # H, H, CNOT, H, H -> CNOT
                # Create replacement subgraph with CNOT gate
                replacement = ((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))

            elif self.matching_key == 'h-h-cx': # H, H, CNOT -> CNOT, H, H
                replacement = []
                replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))
                replacement.append((HGate(), [self.circuit.qubits[control_qubit]], []))
                replacement.append((HGate(), [self.circuit.qubits[target_qubit]], []))

            elif self.matching_key == 'cx-h-h': # CNOT, H, H -> H, H, CNOT
                replacement = []
                replacement.append((HGate(), [self.circuit.qubits[control_qubit]], []))
                replacement.append((HGate(), [self.circuit.qubits[target_qubit]], []))
                replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))

            elif self.matching_key == 'h-cx-h': # H, CNOT, H -> H, CNOT, H
                replacement = []
                replacement.append((HGate(), [self.circuit.qubits[control_qubit]], []))
                replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))
                replacement.append((HGate(), [self.circuit.qubits[control_qubit]], []))

            else:
                raise TransformationError(f"Unexpected matching key for control-target swap: {self.matching_key}")

            self.replacement = replacement

        except Exception as e:
            raise TransformationError(f"Failed to create replacement: {e}")


    def apply_transformation(self):
        """Apply the transformation by swapping the control and target qubits of a CNOT gate in the circuit."""
        try:
            if not self.replacement:
                raise TransformationError("No replacement gates found for the transformation.")
        
            # Retrieve indices of gates in the matching subgraph
            matching_idxs = self.get_matching_indices()

            # Get the list of current operations in the circuit
            operations = [(op.operation, op.qubits, op.clbits) for op in self.circuit.data]
            replacement = self.replacement

            # Check if matching_idxs contains exactly (CNOT)
            if self.matching_key == 'cx': # CNOT -> H, H, CNOT, H, H
                idx = matching_idxs[0]
                # Swap
                transformed_operations = operations[:idx] + replacement + operations[idx+1:]

            elif self.matching_key == 'h-h-cx-h-h': # H, H, CNOT, H, H -> CNOT
                idx_h1, idx_h2, idx_cnot, idx_h3, idx_h4 = matching_idxs
                transformed_operations = []

                for i, op in enumerate(operations):
                    if i == idx_cnot:
                        transformed_operations.append(replacement)  # Substitute the CNOT with a CNOT with swapped control and target
                    elif i not in [idx_h1, idx_h2, idx_h3, idx_h4]:
                        transformed_operations.append(op)  # Keep all elements not in the matching indices
            
            elif self.matching_key == 'h-h-cx': # H, H, CNOT -> CNOT, H, H
                idx_h1, idx_h2, idx_cnot = matching_idxs
                transformed_operations = []

                for i, op in enumerate(operations):
                    if i == idx_cnot:
                        transformed_operations.extend(replacement) 
                    elif i not in [idx_h1, idx_h2]:
                        transformed_operations.append(op)

            elif self.matching_key == 'cx-h-h': # CNOT, H, H -> H, H, CNOT
                idx_cnot, idx_h1, idx_h2 = matching_idxs
                transformed_operations = []

                for i, op in enumerate(operations):
                    if i == idx_cnot:
                        transformed_operations.extend(replacement)
                    elif i not in [idx_h1, idx_h2]:
                        transformed_operations.append(op)

            elif self.matching_key == 'h-cx-h': # H, CNOT, H -> H, CNOT, H
                idx_h1, idx_cnot, idx_h2 = matching_idxs
                transformed_operations = []

                for i, op in enumerate(operations):
                    if i == idx_cnot:
                        transformed_operations.extend(replacement)
                    elif i not in [idx_h1, idx_h2]:
                        transformed_operations.append(op)

            else:
                raise TransformationError(f"Expected exactly one or five matching index for control-target swap, but got {len(matching_idxs)}: {matching_idxs}")

            print(f"Transformed operations: {transformed_operations}")
            # Create a new circuit with the transformed operations
            transformed_qc = QuantumCircuit(self.num_qubits)
            for instruction in transformed_operations:
                inst, qargs, cargs = instruction
                transformed_qc.append(inst, qargs, cargs)

            # Return the new circuit graph representation
            return QuantumCircuitGraph(transformed_qc)
        
        except Exception as e:
            raise TransformationError(f"Failed to apply transformation: {e}")


class CNOTDecompositionTransformation(CircuitTransformation):
    """
    Transformation to implement the equivalence that expresses a multi-controlled CNOT gate as a sequence 
    of CNOT gates using an ancilla qubit.
    """

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        """
        Create the pattern to match a single CNOT gate or a sequence of 4 CNOT gates, as showed below.
                                 
        ──■─────────■───────                ───────■─────────■──
        ┌─┴─┐     ┌─┴─┐                          ┌─┴─┐     ┌─┴─┐
        ┤ X ├──■──┤ X ├──■──        or      ──■──┤ X ├──■──┤ X ├
        └───┘┌─┴─┐└───┘┌─┴─┐                ┌─┴─┐└───┘┌─┴─┐└───┘
        ─────┤ X ├─────┤ X ├                ┤ X ├─────┤ X ├─────
             └───┘     └───┘                └───┘     └───┘ 
        """
        try:
            self.pattern_subgraph = []

            # Define the pattern to match a single CNOT gate
            pattern_subcircuit = QuantumCircuit(2)
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

            self.pattern_subgraph.append(('cx', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

            # Define the pattern to match a sequence of 4 CNOT gates
            # ──■─────────■───────
            # ┌─┴─┐     ┌─┴─┐
            # ┤ X ├──■──┤ X ├──■──
            # └───┘┌─┴─┐└───┘┌─┴─┐
            # ─────┤ X ├─────┤ X ├
            #      └───┘     └───┘
            pattern_subcircuit = QuantumCircuit(3)
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[1], pattern_subcircuit.qubits[2]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[1], pattern_subcircuit.qubits[2]])

            self.pattern_subgraph.append(('t-c-t-c', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

            # ───────■─────────■──
            #      ┌─┴─┐     ┌─┴─┐
            # ──■──┤ X ├──■──┤ X ├
            # ┌─┴─┐└───┘┌─┴─┐└───┘
            # ┤ X ├─────┤ X ├─────
            # └───┘     └───┘
            pattern_subcircuit = QuantumCircuit(3)
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[1], pattern_subcircuit.qubits[2]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[1], pattern_subcircuit.qubits[2]])
            pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

            self.pattern_subgraph.append(('c-t-c-t', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

        except Exception as e:
            raise TransformationError(f"Failed to create pattern: {e}")
        
    def create_replacement(self):
        """
        Create the replacement subgraph to decompose a CNOT gate into a sequence of 4 CNOT gates (or vice versa).
        """
        try:
            # Get the qubits on which the CNOT gate acts
            if not self.matching_subgraph:
                raise TransformationError("No matching subgraphs found for the control-target swap transformation.")
            
            if not self.matching_key:
                raise TransformationError("No matching key found for the control-target swap transformation.")
            
            if self.matching_key == 'cx':
                # Retrieve the control and target qubits from the matching subgraph
                control_qubit = None
                target_qubit = None

                iter_nodes = iter(self.matching_subgraph.keys())
                # until both control and target qubits are found, iterate over the nodes in the matching subgraph
                for node in iter_nodes:
                    if 'control' in node: # get control qubit and assign to target qubit for swap
                        control_qubit = int(node.split('_')[1]) # cx_{control_qubit}_control_{node_id}
                    elif 'target' in node: # get target qubit and assign to control qubit for swap
                        target_qubit = int(node.split('_')[1]) # cx_{target_qubit}_target_{node_id}

                    if control_qubit is not None and target_qubit is not None:
                        break

                if control_qubit is None or target_qubit is None:
                    raise TransformationError("Could not find both control and target qubits in the subgraph.")
                
                # Choose an ancilla qubit (randomly) that is not the control or target qubit
                ancilla_qubit = random.choice([q for q in range(self.num_qubits) if q not in [control_qubit, target_qubit]])

                replacement = []
                # Choose randomly the order of gates in the decomposition
                if random.choice([True, False]):
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))
                else:
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))


            elif self.matching_key == 't-c-t-c':
                # Get the qubits on which the replacing CNOT gates act:
                # - control qubit is the qubit of the first node in the matching subgraph
                # - target qubit is the qubit of the last node in the matching subgraph
                control_qubit = int(list(self.matching_subgraph.keys())[0].split('_')[1])
                ancilla_qubit = int(list(self.matching_subgraph.keys())[1].split('_')[1])
                target_qubit = int(list(self.matching_subgraph.keys())[-1].split('_')[1])

                replacement = []
                # Choose randomly if substitute with a single CNOT or with the other sequence of CNOTs
                if random.choice([True, False]):
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))
                else:
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))

            elif self.matching_key == 'c-t-c-t':
                # Get the qubits on which the replacing CNOT gates act:
                # - target qubit is the qubit of the second node in the matching subgraph
                # - control qubit is the qubit of the second-last node in the matching subgraph
                target_qubit = int(list(self.matching_subgraph.keys())[1].split('_')[1])
                ancilla_qubit = int(list(self.matching_subgraph.keys())[0].split('_')[1])
                control_qubit = int(list(self.matching_subgraph.keys())[-2].split('_')[1])

                replacement = []
                # Choose randomly if substitute with a single CNOT or with the other sequence of CNOTs
                if random.choice([True, False]):
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))
                else:
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[ancilla_qubit]], []))
                    replacement.append((CXGate(), [self.circuit.qubits[ancilla_qubit], self.circuit.qubits[target_qubit]], []))

            else:
                raise TransformationError(f"Unexpected matching key for control-target swap: {self.matching_key}")

            self.replacement = replacement

        except Exception as e:
            raise TransformationError(f"Failed to create replacement: {e}")
        

    def apply_transformation(self):
        """Apply the transformation by decomposing a CNOT gate into a sequence of 4 CNOT gates (or vice versa)."""
        try:
            if not self.replacement:
                raise TransformationError("No replacement gates found for the transformation.")
        
            # Retrieve indices of gates in the matching subgraph
            matching_idxs = self.get_matching_indices()

            # Get the list of current operations in the circuit
            operations = [(op.operation, op.qubits, op.clbits) for op in self.circuit.data]
            replacement = self.replacement

            # Check if matching_idxs contains exactly one or four elements
            if self.matching_key == 'cx':
                idx = matching_idxs[0]
                # Swap
                transformed_operations = operations[:idx] + replacement + operations[idx+1:]

            elif self.matching_key in ['t-c-t-c', 'c-t-c-t']:
                idx1, idx2, idx3, idx4 = matching_idxs
                transformed_operations = []

                for i, op in enumerate(operations):
                    if i not in [idx1, idx2, idx3, idx4]:
                        transformed_operations.append(op)
                    if i == idx2+1:
                        transformed_operations.extend(replacement)

            else:
                raise TransformationError(f"Unexpected matching key for CNOT decomposition: {self.matching_key}")

            print(f"Transformed operations: {transformed_operations}")
            # Create a new circuit with the transformed operations
            transformed_qc = QuantumCircuit(self.num_qubits)
            for instruction in transformed_operations:
                inst, qargs, cargs = instruction
                transformed_qc.append(inst, qargs, cargs)

            # Return the new circuit graph representation
            return QuantumCircuitGraph(transformed_qc)
        
        except Exception as e:
            raise TransformationError(f"Failed to apply transformation: {e}")


class ChangeOfBasisTransformation(CircuitTransformation):
    """
    Transformation to implement a change of basis on a qubit in the circuit, by applying the equivalence
    HZH = X or HXH = Z. Any of the subpatterns derived from this equivalence can be used for the transformation.
    """
    def __init__(self, qcg):
        super().__init__(qcg)

        self.transformation_rules = {
            'H-Z-H': 'X',
            'H-X-H': 'Z',
            'X-H-Z': 'H',
            'Z-H-X': 'H',
            'H-Z': 'X-H',
            'Z-H': 'H-X',
            'X-H': 'H-Z',
            'H-X': 'Z-H',
            'X': 'H-Z-H',
            'Z': 'H-X-H',
            'H': ['X-H-Z','Z-H-X']
        }

    def create_pattern(self):
        """Generate all possible patterns based on transformation rules."""
        self.pattern_subgraph = []
        
        for pattern in self.transformation_rules.keys():
            subcircuit = QuantumCircuit(1)
            
            gates = pattern.split('-')
            for gate in gates:
                if gate == 'H':
                    subcircuit.append(HGate(), [subcircuit.qubits[0]])
                elif gate == 'X':
                    subcircuit.append(XGate(), [subcircuit.qubits[0]])
                elif gate == 'Z':
                    subcircuit.append(ZGate(), [subcircuit.qubits[0]])

            # Build the graph from the pattern subcircuit and append to subgraph list
            self.pattern_subgraph.append((pattern.lower(), build_graph_from_circuit(subcircuit, self.gate_type_map, data=False)))

    def create_replacement(self):
        """Create the replacement subgraph using the transformation rules."""
        try:
            if not self.matching_subgraph:
                raise TransformationError("No matching subgraphs found for the transformation.")
            
            if not self.matching_key:
                raise TransformationError("No matching key found for the transformation.")

            qubit = int(list(self.matching_subgraph.keys())[0].split('_')[1])
            replacement_key = self.transformation_rules.get(self.matching_key.upper())

            if replacement_key:
                replacement = []
                if isinstance(replacement_key, list):
                    replacement_key = random.choice(replacement_key)

                replacement_gates = replacement_key.split('-')
                for gate in replacement_gates:
                    if gate == 'X':
                        replacement.append((XGate(), [self.circuit.qubits[qubit]], []))
                    elif gate == 'H':
                        replacement.append((HGate(), [self.circuit.qubits[qubit]], []))
                    elif gate == 'Z':
                        replacement.append((ZGate(), [self.circuit.qubits[qubit]], []))
                        
                
                print(f"Replacement: {replacement}")
                self.replacement = replacement
            else:
                raise TransformationError(f"Unexpected matching key for change of basis: {self.matching_key}")

        except Exception as e:
            raise TransformationError(f"Failed to create replacement: {e}")
        
    def apply_transformation(self):
        """Apply the transformation by changing the basis of a qubit in the circuit."""
        try:
            if not self.replacement:
                raise TransformationError("No replacement gates found for the transformation.")
        
            # Retrieve indices of gates in the matching subgraph
            matching_idxs = self.get_matching_indices()
            print(f"Matching indices: {matching_idxs}")
            print(f"Number of matching indices: {len(matching_idxs)}")

            # Get the list of current operations in the circuit
            operations = [(op.operation, op.qubits, op.clbits) for op in self.circuit.data]
            replacement = self.replacement

            if len(matching_idxs) == 1:
                idx = matching_idxs[0]
                transformed_operations = operations[:idx] + replacement + operations[idx+1:]

            elif len(matching_idxs) == 2:
                idx1, idx2 = matching_idxs
                transformed_operations = operations.copy()
                transformed_operations[idx1], transformed_operations[idx2] = replacement[0], replacement[1]

            elif len(matching_idxs) == 3:
                idx1, idx2, idx3 = matching_idxs
                transformed_operations = []
                for i, op in enumerate(operations):
                    if i not in [idx1, idx2, idx3]:
                        transformed_operations.append(op)
                    if i == idx2:
                        transformed_operations.extend(replacement)
            else:
                raise TransformationError(f"Expected 1, 2 or 3 matching indices for change of basis, but got {len(matching_idxs)}: {matching_idxs}")

            print(f"Transformed operations: {transformed_operations}")
            # Create a new circuit with the transformed operations
            transformed_qc = QuantumCircuit(self.num_qubits)
            for instruction in transformed_operations:
                inst, qargs, cargs = instruction
                transformed_qc.append(inst, qargs, cargs)

            # Return the new circuit graph representation
            return QuantumCircuitGraph(transformed_qc)
        
        except Exception as e:
            raise TransformationError(f"Failed to apply transformation: {e}")
        

class ParallelXTransformation(CircuitTransformation):
    """
    #TODO
    """

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        """
        Create the pattern to be matched for the reverse transformation. How to match a parallel X gate 
        on two qubits in the circuit will be implemented in matching_subgraph method.
        """
        # Define the pattern to match two CNOTs with a X gate in the middle of the controls
        #       ┌───┐    
        #  ──■──┤ X ├──■──
        #  ┌─┴─┐└───┘┌─┴─┐
        #  ┤ X ├─────┤ X ├
        #  └───┘     └───┘

        pattern_subcircuit = QuantumCircuit(2)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(XGate(), [pattern_subcircuit.qubits[0]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

        self.pattern_subgraph = [('cx-x-cx', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))]

    def find_matching_subgraph(self):
        try:
            super().find_matching_subgraph()
        except NoMatchingSubgraphsError:
            self.matching_subgraph = None

        # Check if there is a parallel X gate in the circuit
        matching_parallel_x = self.find_parallel_x()

        if matching_parallel_x and self.matching_subgraph:
            # Choose randomly which one to apply the transformation to
            if random.choice([True, False]):
                self.matching_subgraph = matching_parallel_x
                self.matching_key = 'parallel-x'

        elif not self.matching_subgraph and matching_parallel_x:
            self.matching_subgraph = matching_parallel_x
            self.matching_key = 'parallel-x'

        # else, keep the original matching subgraph and key (which is the CNOT-X-CNOT pattern)

        if not self.matching_subgraph and not matching_parallel_x:
            raise NoMatchingSubgraphsError("No matching subgraphs found for the given pattern.")

    def create_replacement(self):
        """Create the replacement subgraph for the transformation."""
        try:
            if not self.matching_subgraph:
                raise TransformationError("No matching subgraphs found for the control-target swap transformation.")
            
            if not self.matching_key:
                raise TransformationError("No matching key found for the control-target swap transformation.")
            
            if self.matching_key == 'cx-x-cx':
                control_qubit = None
                target_qubit = None

                iter_nodes = iter(self.matching_subgraph.keys())
                for node in iter_nodes:
                    if 'control' in node:
                        control_qubit = int(node.split('_')[1])
                    elif 'target' in node:
                        target_qubit = int(node.split('_')[1])

                replacement = []
                replacement.append((XGate(), [self.circuit.qubits[control_qubit]], []))
                replacement.append((XGate(), [self.circuit.qubits[target_qubit]], []))

            elif self.matching_key == 'parallel-x':
                # Get the qubits associated with the two parallel X gates and assign randomly to control and target
                qubits = list(set([int(node.split('_')[1]) for node in self.matching_subgraph.keys()]))
                control_qubit = qubits[0]
                target_qubit = qubits[1]

                replacement = []
                replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))
                replacement.append((XGate(), [self.circuit.qubits[control_qubit]], []))
                replacement.append((CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []))

            else:
                raise TransformationError(f"Unexpected matching key for control-target swap: {self.matching_key}")
            
            print(f"Replacement: {replacement}")

            self.replacement = replacement

        except Exception as e:
            raise TransformationError(f"Failed to create replacement: {e}")
        
    def apply_transformation(self):
        """TODO"""
        try:
            if not self.replacement:
                raise TransformationError("No replacement gates found for the transformation.")
        
            # Retrieve indices of gates in the matching subgraph
            matching_idxs = self.get_matching_indices()

            # Get the list of current operations in the circuit
            operations = [(op.operation, op.qubits, op.clbits) for op in self.circuit.data]

            # If the matching key is 'cx-x-cx', the replacement is a list of gates to be inserted in the circuit
            if self.matching_key == 'cx-x-cx':
                idx_cx1, idx_x, idx_cx2 = matching_idxs
                transformed_operations = []

                for i, op in enumerate(operations):
                    if i == idx_x:
                        transformed_operations.extend(self.replacement)
                    elif i not in [idx_cx1, idx_cx2]:
                        transformed_operations.append(op)

                # Create a new circuit with the transformed operations
                transformed_qc = QuantumCircuit(self.num_qubits)
                for instruction in transformed_operations:
                    inst, qargs, cargs = instruction
                    transformed_qc.append(inst, qargs, cargs)

            elif self.matching_key == 'parallel-x':
                x_control, x_target = self.matching_subgraph.keys()
                transformed_operations = []

                # Retrieve predecessors and successors of the X gates
                pred_control = get_predecessor(self.graph, x_control)
                pred_target = get_predecessor(self.graph, x_target)
                succ_control = get_successor(self.graph, x_control)
                succ_target = get_successor(self.graph, x_target)
                print(f"Predecessors and successors of control X gate: {pred_control}, {succ_control}")
                print(f"Predecessors and successors of target X gate: {pred_target}, {succ_target}")

                # Remove the X gates from the circuit (and their edges)
                # while keeping remaining gates on the qubit connected (add edges between predecessors and successors)
                self.graph.remove_nodes_from([x_control, x_target])
                self.graph.remove_edges_from([(pred_control, x_control), (x_control, succ_control), (pred_target, x_target), (x_target, succ_target)])
                self.graph.add_edges_from([(pred_control, succ_control), (pred_target, succ_target)])

                # Add gates to the circuit
                for i, gate in enumerate(self.replacement):
                    print(f"Adding gate: {gate}")
                    gate_data = process_gate(gate, self.gate_type_map, node_id='add'+str(i))
                    print(f"Gate data: {gate_data}")
                    if isinstance(gate_data, list) and len(gate_data) == 2:
                        added_node_label = insert_node(self.graph, gate_data, [(pred_control, succ_control), (pred_target, succ_target)])
                        pred_control = added_node_label[0]
                        pred_target = added_node_label[1]
                    elif isinstance(gate_data, tuple) and len(gate_data) == 2:
                        added_node_label = insert_node(self.graph, gate_data, (pred_control, succ_control))
                        pred_control = added_node_label[0]
                    else:
                        raise TransformationError(f"Unexpected gate data format: {gate_data} for gate {gate}")

                # Transform the graph back to a circuit
                transformed_qc = build_circuit_from_graph(self.graph)

            else:
                raise TransformationError(f"Unexpected matching key for CNOT decomposition: {self.matching_key}")
            
            return QuantumCircuitGraph(transformed_qc)
        
        except Exception as e:
            raise TransformationError(f"Failed to apply transformation: {e}")
        

    def find_parallel_x(self):
        # Identify all X gates in the circuit
        x_gates = [n for n, d in self.graph.nodes(data=True) if d['type'] == 'x']
        random.shuffle(x_gates)

        # Check for parallelism (no path through CNOT)
        for i in range(len(x_gates)):
            for j in range(i + 1, len(x_gates)):
                x1, x2 = x_gates[i], x_gates[j]
                
                # Check if the gates are on different qubits
                if self.graph.nodes[x1]['qubit'] != self.graph.nodes[x2]['qubit']:
                    
                    # Check if there's no directed path between x1 and x2 (passing through a CNOT)
                    if not nx.has_path(self.graph, x1, x2) and not nx.has_path(self.graph, x2, x1):
                        print(f"Found parallel X gates: {x1}, {x2}")
                        return {x1: x1, x2: x2}

        # If no parallel X gates are found, return None
        return None


# ------------- Helper functions -------------

def get_predecessor(graph, node):
    """
        Retrieves the predecessor of a given node in a the graph representation of a circuit.
        
        A node `n1` is considered a predecessor of `node` (`n2`) if:
        - There is a directed edge from `n1` to `node` (i.e., `n1 -> node`)
        - There is no directed edge from `node` back to `n1` (i.e., `node -> n1` does not exist)

        If no predecessor is found, an empty list is returned.
    """
    for pred in graph.predecessors(node):
        if not graph.has_edge(node, pred):
            return pred
    return None


def get_successor(graph, node):
    """
        Retrieves the successor of a given node in a the graph representation of a circuit.
        
        A node `n1` is considered a successor of `node` (`n2`) if:
        - There is a directed edge from `node` to `n1` (i.e., `node -> n1`)
        - There is no directed edge from `n1` back to `node` (i.e., `n1 -> node` does not exist)

        If no successor is found, an empty list is returned.
    """
    for succ in graph.successors(node):
        if not graph.has_edge(succ, node):
            return succ
    return None
