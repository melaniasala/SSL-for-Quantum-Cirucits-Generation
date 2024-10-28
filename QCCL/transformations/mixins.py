import random
import networkx as nx
from qiskit import QuantumCircuit

from Data.data_preprocessing import process_gate, insert_node, build_circuit_from_graph
from QCCL.transformations.base_transform import TransformationError, get_qubit

class ParallelGatesMixin:
    """
    A mixin providing helper methods for transformations with parallel gates.
    """

    def __init__(self, gate_type):
        self.gate_type = gate_type

    def find_parallel_gates(self, graph):
        """
        Identifies two parallel gates of the same type acting on different qubits with no directed path
        between them through a CNOT gate.
        """
        # Identify all gate_type gates in the circuit
        gates = [n for n, d in graph.nodes(data=True) if d['type'] == self.gate_type]
        random.shuffle(gates)

        # Check for parallelism (no path through CNOT)
        for i, gate1 in enumerate(gates):
            for gate2 in gates[i + 1:]:
                
                # Check if the gates are on different qubits
                if graph.nodes[gate1]['qubit'] != graph.nodes[gate2]['qubit']:
                    # Check if there's no directed path between gate1 and gate2 (passing through a CNOT)
                    if not nx.has_path(graph, gate1, gate2) and not nx.has_path(graph, gate2, gate1):
                        return {gate1: gate1, gate2: gate2}

        # If no parallel gates are found, return None
        return None
    
    def _get_control_target_qubits(self, matching_key, matching_subgraph):   
        """
        Extracts control and target qubits from the matching subgraph based on the key.
        """
        # If matching key is a string in the format 'cx-' + gate_type + '-cx'
        if matching_key == 'cx-' + self.gate_type + '-cx':
            control_qubit = None
            target_qubit = None
            for node in matching_subgraph.keys():
                if 'control' in node and control_qubit is None:
                    control_qubit = get_qubit(node)
                elif 'target' in node and target_qubit is None:
                    target_qubit = get_qubit(node)
                if control_qubit is not None and target_qubit is not None:
                    break

        # If matching key is a string in the format 'parallel-' + gate_type
        elif matching_key == 'parallel-' + self.gate_type:
            nodes = list(matching_subgraph.keys())
            control_qubit = get_qubit(nodes[0])
            target_qubit = get_qubit(nodes[1])

        if control_qubit is None or target_qubit is None:
            raise TransformationError("Failed to extract control and target qubits from the matching subgraph.")

        return control_qubit, target_qubit
    
    def _apply_cx_gate_cx_transformation(self, circuit, replacement, matching_indices):
        """
        Applies the transformation for the cx_gate_cx pattern.
        """
        idx_cx1, idx_gate, idx_cx2 = matching_indices
        operations = [(op.operation, op.qubits, op.clbits) for op in circuit.data]
        
        transformed_operations = []
        for i, op in enumerate(operations):
            if i == idx_gate:
                transformed_operations.extend(replacement)
            elif i not in [idx_cx1, idx_cx2]:
                transformed_operations.append(op)
        
        transformed_qc = QuantumCircuit(circuit.num_qubits)
        for inst, qargs, cargs in transformed_operations:
            transformed_qc.append(inst, qargs, cargs)
        
        return transformed_qc
    
    def _apply_parallel_gate_transformation(self, graph, matching_subgraph, replacement, gate_type_map):
        """
        Applies the transformation for parallel gates.
        """
        gate_control, gate_target = matching_subgraph.keys()
         # Remove the parallel gates and set up predecessors/successors
        pred_control, succ_control, pred_target, succ_target = self._remove_parallel_gates_and_update_graph(graph, gate_control, gate_target)
        
        for i, gate in enumerate(replacement):
            gate_data = process_gate(gate, gate_type_map, node_id=f'add{i}')
            pred_control, pred_target = self._insert_and_update_predecessors(graph, gate_data, pred_control, succ_control, pred_target, succ_target)

        return build_circuit_from_graph(graph)
    
    def _remove_parallel_gates_and_update_graph(self, graph, gate_control, gate_target):
        """
        Removes parallel gates from the circuit graph and updates edges between their predecessors and successors.
        """
        pred_control, succ_control = get_predecessor(graph, gate_control), get_successor(graph, gate_control)
        pred_target, succ_target = get_predecessor(graph, gate_target), get_successor(graph, gate_target)

        graph.remove_nodes_from([gate_control, gate_target])
        # Add edges between predecessors and successors only if both are not None (i.e., not a input/output parallel gate)
        if pred_control is not None and succ_control is not None:
            graph.add_edge(pred_control, succ_control)
        if pred_target is not None and succ_target is not None:
            graph.add_edge(pred_target, succ_target)

        return pred_control, succ_control, pred_target, succ_target

    def _insert_and_update_predecessors(self, graph, gate_data, pred_control, succ_control, pred_target, succ_target):
        """
        Inserts a gate node into the graph and updates predecessors based on gate type.
        """
        # Two-qubit gate
        if isinstance(gate_data, list) and len(gate_data) == 2:  
            added_node_label = insert_node(
                graph, gate_data, [(pred_control, succ_control), (pred_target, succ_target)]
            )
            return added_node_label  # Update both control and target predecessors

        # Single-qubit gate
        elif isinstance(gate_data, tuple) and len(gate_data) == 2 and self.gate_type == 'x':  
            added_node_label = insert_node(graph, gate_data, (pred_control, succ_control))
            return added_node_label[0], pred_target  # Update control predecessor only

        elif isinstance(gate_data, tuple) and len(gate_data) == 2 and self.gate_type == 'z':
            added_node_label = insert_node(graph, gate_data, (pred_target, succ_target))
            return pred_control, added_node_label[0]  # Update target predecessor only
        
        else:
            raise TransformationError(f"Unexpected gate data format: {gate_data}")
    

class CommutationMixin:
    """
    A mixin providing helper methods for transformations with commutation.
    """

    def _commute(self, graph, matching_subgraph):
        """
        Commutes the gates in the matching subgraph.
        """
        # Check if there are exactly two gates in the matching subgraph
        if self._count_gates(matching_subgraph) != 2:
            raise TransformationError("Expected exactly two gates in the matching subgraph.")
        
        # Get predecessors and successors of the pattern
        predecessors_successors = {}

        # Iterate over each node in the matching subgraph
        for node_label in matching_subgraph.keys():
            # Get the qubit associated with this node
            qubit = get_qubit(node_label)

            # Initialize lists for predecessors and successors if not already done
            if qubit not in predecessors_successors:
                predecessors_successors[qubit] = {'predecessors': [], 'successors': []}

            # Get the predecessor and successor of the node
            predecessors_successors[qubit]['predecessors'].append(get_predecessor(graph, node_label))
            predecessors_successors[qubit]['successors'].append(get_successor(graph, node_label))
            
        common_qubits = []

        print(f"Predecessors and successors (before): {predecessors_successors}")

        # Now remove the nodes in the matching subgraph from lists of predecessors and successors,
        # retrieving the predecessor and successor of the pattern (should be exactly one for each qubit)
        for qubit, preds_succs in predecessors_successors.items():
            if preds_succs['predecessors'] is None or len(preds_succs['predecessors']) == 0:
                raise TransformationError("Predecessor not found for qubit {qubit}.")
            
            # If there are multiple predecessors, remove the ones that are part of the matching subgraph.
            # At the end, there should be exactly one predecessor.
            if len(set(preds_succs['predecessors'])) > 1:
                try:
                    preds_succs['predecessors'] = [pred for pred in preds_succs['predecessors'] if pred not in matching_subgraph.keys()]
                except ValueError:
                    raise TransformationError("Even if there are multiple predecessors, none of them belong to the matching subgraph.")
                if len(preds_succs['predecessors']) != 1:
                    raise TransformationError("Commuting gates in the matching subgraph are not adjacent.")
                print(f"Predecessors: {preds_succs['predecessors'][0]}")
                predecessors_successors[qubit]['predecessors'] = preds_succs['predecessors'][0]
                # Add the qubit to the list of common qubits, the ones which have 2 commuting nodes on them
                common_qubits.append(qubit)
                
            # Do the same for successors
            if preds_succs['successors'] is None or len(preds_succs['successors']) == 0:
                raise TransformationError("Successor not found for qubit {qubit}.")
            
            if len(set(preds_succs['successors'])) > 1:
                try:
                    preds_succs['successors'] = [succ for succ in preds_succs['successors'] if succ not in matching_subgraph.keys()]
                except ValueError:
                    raise TransformationError("Even if there are multiple successors, none of them belong to the matching subgraph.")
                if len(preds_succs['successors']) != 1:
                    raise TransformationError("Commuting gates in the matching subgraph are not adjacent.")
                print(f"Successors: {preds_succs['successors'][0]}")
                predecessors_successors[qubit]['successors'] = preds_succs['successors'][0]
                if qubit not in common_qubits:
                    raise TransformationError("There is a mismatch in predecessors and successors of the matching subgraph: a non-common qubit has only one successor.")
            else:
                if qubit in common_qubits:
                    raise TransformationError("There is a mismatch in predecessors and successors of the matching subgraph: a common qubit has only one successor.")
   
        print(f"Common qubits: {common_qubits}")
        print(f"Predecessors and successors (after): {predecessors_successors}")

        # Commute the gates, by modifying the edges in the graph
        for qubit in common_qubits:
            print(f"Handling common qubit: {qubit}")
            # Get the gates acting on the qubit
            gates = [node for node in matching_subgraph.keys() if get_qubit(node) == qubit]
            print(f"Gates to be swapped: {gates}")
            if len(gates) != 2:
                raise TransformationError("Expected exactly two gates acting on the qubit.")
            
            # Check they act on the same qubit
            if get_qubit(gates[0]) != get_qubit(gates[1]):
                raise TransformationError("Gates to be commuted do not act on the same qubit.")
            
            predecessor = predecessors_successors[qubit]['predecessors']
            successor = predecessors_successors[qubit]['successors']

            print(f"Predecessor: {predecessor}, Successor: {successor} on the qubit")
            print(f"First gate: {gates[0]}, Second gate: {gates[1]}")
            
            # Swap the gates
            if (gates[0], gates[1]) in graph.edges:
                self._reverse_edge_direction(graph, gates[0], gates[1])
                if predecessor is not None:
                    graph.remove_edge(predecessor, gates[0])
                    graph.add_edge(predecessor, gates[1])

                if successor is not None:
                    graph.remove_edge(gates[1], successor)
                    graph.add_edge(gates[0], successor)

            else:
                self._reverse_edge_direction(graph, gates[0], gates[1])
                if predecessor is not None:
                    graph.remove_edge(predecessor, gates[1])
                    graph.add_edge(predecessor, gates[0])

                if successor is not None:
                    graph.remove_edge(gates[0], successor)
                    graph.add_edge(gates[1], successor)


    def _count_gates(self, matching_subgraph):
        """
        Counts the unique gates in the matching subgraph.
        """
        unique_gates = set()
        
        for gate in matching_subgraph.keys():
            components = gate.split('_')
            if components[0] == 'cx':
                components = components[:1] + components[3:]
            unique_gate = '_'.join(components)
            unique_gates.add(unique_gate)
        
        return len(unique_gates)
                
    def _reverse_edge_direction(self, graph, u, v):
        """
        Changes the direction of an edge in the graph.
        """
        print(f"Swapping edge direction: {u} -> {v} to {v} -> {u}")
        
        graph.remove_edge(u, v)
        graph.add_edge(v, u)
        




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
