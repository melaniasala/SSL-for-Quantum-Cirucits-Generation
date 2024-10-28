import random

from QCCL.transformations.base_transform import CircuitTransformation, TransformationError, NoMatchingSubgraphsError, get_qubit
from QCCL.transformations.mixins import ParallelGatesMixin, CommutationMixin
from Data.data_preprocessing import build_graph_from_circuit, build_circuit_from_graph
from Data.QuantumCircuitGraph import QuantumCircuitGraph
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, TGate, XGate, ZGate 


    
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
class CommuteCNOTsTransformation(CircuitTransformation, CommutationMixin):
    """
    Transformation to commute two CNOT gates in the circuit.
    Two CNOT gates can be commuted if they have the same control or the same target qubits.
    """

    def __init__(self, qcg):
        super().__init__(qcg)

    def create_pattern(self):
        """
        Create the pattern to match two CNOT gates (with the same control or same target qubits)
        """
        self.pattern_subgraph = []

        # Define the pattern to match two CNOT gates with the same control
        pattern_subcircuit = QuantumCircuit(3)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[2]])

        self.pattern_subgraph.append(('control', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

        # Define the pattern to match two CNOT gates with the same target
        pattern_subcircuit = QuantumCircuit(3)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[2], pattern_subcircuit.qubits[1]])

        self.pattern_subgraph.append(('target', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False)))

    def create_replacement(self):
        pass

    def apply_transformation(self):
        """Apply the transformation by commuting two CNOT gates in the circuit."""
        if not self.matching_subgraph:
            raise TransformationError("No matching subgraphs found for the CNOTs commutation transformation.")
        
        if not self.matching_key:
            raise TransformationError("No matching key found for the CNOTs commutation transformation.")

        transformed_graph = self.graph.copy()
        check_graph = self.graph.copy()
        self._commute_gates(transformed_graph, self.matching_subgraph)

        if transformed_graph == check_graph:
            raise TransformationError("Failed to apply the transformation: the graph was not modified.")

        return QuantumCircuitGraph(build_circuit_from_graph(transformed_graph))
    

class CommuteCNOTRotationTransformation(CircuitTransformation, CommutationMixin):
    """
    Transformation to commute a CNOT gate with a rotation gate in the circuit.
    A CNOT gate immediately preceeded/followed by a rotation gate around x-axis (X) on the target qubit can be commuted.
    The same applies for a CNOT gate immediately preceeded/followed by a rotation gate around z-axis (Z, T) on the control qubit.
    """

    def __init__(self, qcg):
        super().__init__(qcg)

        self.patterns = ['cx-x-target', 'x-cx-target', 'cx-z-control', 'z-cx-control', 't-cx-control', 'cx-t-control']

    def create_pattern(self):
        """
        Create the pattern to match a CNOT gate adjacent to a rotation gate on the target or control qubit.
        """
        self.pattern_subgraph = []

        for pattern in self.patterns:
            subcircuit = QuantumCircuit(2)

            gates, qubit_role = pattern.split('-')[:-1], pattern.split('-')[-1]
            common_qubit = 0 if qubit_role == 'control' else 1

            for gate in gates:
                if gate == 'cx':
                    subcircuit.append(CXGate(), [subcircuit.qubits[0], subcircuit.qubits[1]])
                # for single-qubit gates, they should be added on qubit 0 (control) or 1 (target) based on value of common_qubit
                elif gate == 'x':
                    subcircuit.append(XGate(), [subcircuit.qubits[common_qubit]])
                elif gate == 'z':
                    subcircuit.append(ZGate(), [subcircuit.qubits[common_qubit]])
                elif gate == 't':
                    subcircuit.append(TGate(), [subcircuit.qubits[common_qubit]])

            self.pattern_subgraph.append((pattern, build_graph_from_circuit(subcircuit, self.gate_type_map, data=False)))                

    def create_replacement(self):
        pass

    def apply_transformation(self):
        """Apply the transformation by commuting a CNOT gate with a rotation gate in the circuit."""

        if not self.matching_subgraph:
            raise TransformationError("No matching subgraphs found for the CNOT-rotation commutation transformation.")
        
        if not self.matching_key:
            raise TransformationError("No matching key found for the CNOT-rotation commutation transformation.")

        transformed_graph = self.graph.copy()
        check_graph = self.graph.copy()
        self._commute_gates(transformed_graph, self.matching_subgraph)

        if transformed_graph == check_graph:
            raise TransformationError("Failed to apply the transformation: the graph was not modified.")

        return QuantumCircuitGraph(build_circuit_from_graph(transformed_graph))
    


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
                    target_qubit = get_qubit(node)
                elif 'target' in node: # get target qubit and assign to control qubit for swap
                    control_qubit = get_qubit(node)

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
                        control_qubit = get_qubit(node)
                    elif 'target' in node: # get target qubit and assign to control qubit for swap
                        target_qubit = get_qubit(node)

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
                control_qubit = get_qubit(list(self.matching_subgraph.keys())[0])
                ancilla_qubit = get_qubit(list(self.matching_subgraph.keys())[1])
                target_qubit = get_qubit(list(self.matching_subgraph.keys())[-1])

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
                target_qubit = get_qubit(list(self.matching_subgraph.keys())[1])
                ancilla_qubit = get_qubit(list(self.matching_subgraph.keys())[0])
                control_qubit = get_qubit(list(self.matching_subgraph.keys())[-2])


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

            qubit = get_qubit(list(self.matching_subgraph.keys())[0])
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
        

class ParallelXTransformation(CircuitTransformation, ParallelGatesMixin):
    """
    A transformation class for matching and replacing parallel X gates in a quantum circuit.
    Parallel X gates are defined as two X gates acting on different qubits with no intermediate CNOT gates.
    """

    def __init__(self, qcg):
        super().__init__(qcg)
        ParallelGatesMixin.__init__(self, gate_type='x')


    def create_pattern(self):
        """
        Creates the pattern to match for the transformation.
        The pattern includes two CNOT gates with an X gate in between, 
        acting on the same control qubit, to be transformed if matched.
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
        """
        Finds a subgraph that matches either the cx-x-cx pattern or parallel X gates.
        Chooses randomly between matched patterns if both are found.
        """
        try:
            super().find_matching_subgraph()
        except NoMatchingSubgraphsError:
            self.matching_subgraph = None

        # Check if there is a parallel X gate in the circuit
        matching_parallel_x = self.find_parallel_gates(self.graph)

        if matching_parallel_x:
            # Randomly choose between cx-x-cx pattern and parallel X pattern
            if not self.matching_subgraph or random.choice([True, False]):
                self.matching_subgraph = matching_parallel_x
                self.matching_key = 'parallel-x'
        
        if not self.matching_subgraph:
            raise NoMatchingSubgraphsError("No matching subgraphs found for the given pattern.")

    def create_replacement(self):
        """
        Creates the replacement subgraph based on the matching pattern found.
        For 'cx-x-cx', replaces with two X gates on the control and target qubits.
        For 'parallel-x', replaces with a CNOT-X-CNOT sequence.
        """
        if not self.matching_subgraph:
            raise TransformationError("No matching subgraphs found for transformation.")
        
        control_qubit, target_qubit = self._get_control_target_qubits(self.matching_key, self.matching_subgraph)

        if self.matching_key == 'cx-x-cx':
            self.replacement = [
                (XGate(), [self.circuit.qubits[control_qubit]], []),
                (XGate(), [self.circuit.qubits[target_qubit]], [])
            ]
        elif self.matching_key == 'parallel-x':
            self.replacement = [
                (CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []),
                (XGate(), [self.circuit.qubits[control_qubit]], []),
                (CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], [])
            ]
        else:
            raise TransformationError(f"Unexpected matching key: {self.matching_key}")
        
    def apply_transformation(self):
        """
        Applies the transformation by replacing the matched subgraph with the replacement sequence.
        Converts the graph back to a circuit after modification.
        """
        if not self.replacement:
            raise TransformationError("No replacement gates found for the transformation.")

        # If the matching key is 'cx-x-cx', the replacement is a list of gates to be inserted in the circuit
        if self.matching_key == 'cx-x-cx':
            # Retrieve indices of gates in the matching subgraph
            matching_indices = self.get_matching_indices()

            transformed_qc = self._apply_cx_gate_cx_transformation(self.circuit, 
                                                                    self.replacement, 
                                                                    matching_indices)
        elif self.matching_key == 'parallel-x':
            trasformed_graph = self.graph.copy() 
            transformed_qc = self._apply_parallel_gate_transformation(trasformed_graph, 
                                                                        self.matching_subgraph, 
                                                                        self.replacement, 
                                                                        self.gate_type_map)
        else:
            raise TransformationError(f"Unexpected matching key: {self.matching_key}")
        
        return QuantumCircuitGraph(transformed_qc)
    

class ParallelZTransformation(CircuitTransformation, ParallelGatesMixin):
    """
    A transformation class for matching and replacing parallel Z gates in a quantum circuit.
    Parallel Z gates are defined as two Z gates acting on different qubits with no intermediate CNOT gates.
    """

    def __init__(self, qcg):
        super().__init__(qcg)
        ParallelGatesMixin.__init__(self, gate_type='z')

    def create_pattern(self):
        """
        Creates the pattern to match for the transformation.
        The pattern includes two CNOT gates with a Z gate in between, 
        acting on the target qubit, to be transformed if matched.
        """
        # Define the pattern to match two CNOTs with a Z gate in the middle on the target qubit   
        #  ──■───────────■──
        #  ┌─┴─┐ ┌───┐ ┌─┴─┐
        #  ┤ X ├─┤ Z ├─┤ X ├
        #  └───┘ └───┘ └───┘

        pattern_subcircuit = QuantumCircuit(2)
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(ZGate(), [pattern_subcircuit.qubits[1]])
        pattern_subcircuit.append(CXGate(), [pattern_subcircuit.qubits[0], pattern_subcircuit.qubits[1]])

        self.pattern_subgraph = [('cx-z-cx', build_graph_from_circuit(pattern_subcircuit, self.gate_type_map, data=False))]

    def find_matching_subgraph(self):
        """
        Finds a subgraph that matches either the cx-z-cx pattern or parallel Z gates.
        Chooses randomly between matched patterns if both are found.
        """
        try:
            super().find_matching_subgraph()
        except NoMatchingSubgraphsError:
            self.matching_subgraph = None

        # Check if there is a parallel Z gate in the circuit
        matching_parallel_z = self.find_parallel_gates(self.graph)

        if matching_parallel_z:
            # Randomly choose between cx-z-cx pattern and parallel Z pattern
            if not self.matching_subgraph or random.choice([True, False]):
                self.matching_subgraph = matching_parallel_z
                self.matching_key = 'parallel-z'

        print(f"Matching subgraph: {self.matching_subgraph}")
        print(f"Matching key: {self.matching_key}")
        
        if not self.matching_subgraph:
            raise NoMatchingSubgraphsError("No matching subgraphs found for the given pattern.")

    def create_replacement(self):
        """
        Creates the replacement subgraph based on the matching pattern found.
        For 'cx-z-cx', replaces with two Z gates on the control and target qubits.
        For 'parallel-z', replaces with a CNOT-Z-CNOT sequence.
        """
        if not self.matching_subgraph:
            raise TransformationError("No matching subgraphs found for transformation.")
        
        control_qubit, target_qubit = self._get_control_target_qubits(self.matching_key, self.matching_subgraph)

        if self.matching_key == 'cx-z-cx':
            self.replacement = [
                (ZGate(), [self.circuit.qubits[control_qubit]], []),
                (ZGate(), [self.circuit.qubits[target_qubit]], [])
            ]
        elif self.matching_key == 'parallel-z':
            self.replacement = [
                (CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], []),
                (ZGate(), [self.circuit.qubits[target_qubit]], []),
                (CXGate(), [self.circuit.qubits[control_qubit], self.circuit.qubits[target_qubit]], [])
            ]
        else:
            raise TransformationError(f"Unexpected matching key: {self.matching_key}")
    
    def apply_transformation(self):
        """
        Applies the transformation by replacing the matched subgraph with the replacement sequence.
        Converts the graph back to a circuit after modification.
        """
        if not self.replacement:
            raise TransformationError("No replacement gates found for the transformation.")
    
        # Retrieve indices of gates in the matching subgraph
        matching_indices = self.get_matching_indices()

        # If the matching key is 'cx-z-cx', the replacement is a list of gates to be inserted in the circuit
        if self.matching_key == 'cx-z-cx':
            transformed_qc = self._apply_cx_gate_cx_transformation(self.circuit, 
                                                                    self.replacement, 
                                                                    matching_indices)
            
        elif self.matching_key == 'parallel-z':
            trasformed_graph = self.graph.copy() 
            transformed_qc = self._apply_parallel_gate_transformation(trasformed_graph, 
                                                                        self.matching_subgraph, 
                                                                        self.replacement, 
                                                                        self.gate_type_map)
        else:
            raise TransformationError(f"Unexpected matching key: {self.matching_key}")
        
        return QuantumCircuitGraph(transformed_qc)




# ------------- Helper functions -------------

