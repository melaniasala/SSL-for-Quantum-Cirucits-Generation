import random

import networkx as nx
import numpy as np
from qiskit import Aer, execute
from qiskit.quantum_info import Statevector

from .QuantumCircuitGraph import QuantumCircuitGraph
from .RandomCircuitGenerator import RandomCircuitGenerator


class QuantumCircuitDataset:
    def __init__(self, circuit_generator=None, dataset_size=100, save_format='json', graph_config=None):
        """
        Initialize the QuantumCircuitDataset class with parameters.

        Parameters:
        circuit_generator (RandomCircuitGenerator): Instance of the RandomCircuitGenerator class to generate circuits.
        dataset_size (int): Number of circuits to include in the dataset.
        save_format (str): Format to save the dataset in, default is 'json'.
        """
        self.circuit_generator = circuit_generator if circuit_generator is not None else RandomCircuitGenerator()
        self.dataset_size = dataset_size
        self.save_format = save_format
        self.dataset = []
        self.statevectors = {} 

        if graph_config is not None:
            self.set_graph_config(graph_config)

    def generate_dataset(self, dataset_size, qubit_range=(2, 5), depth_range=(2, 10), max_gates=None):
        """
        Generate a dataset of random quantum circuits.

        Parameters:
        dataset_size (int): Total number of circuits to generate in the dataset.
        qubit_range (tuple): Range of qubits (min, max) for the circuits.
        depth_range (tuple): Range of depth (min, max) for the circuits.
        max_gates (int or str): Maximum number of gates allowed per circuit. 
                                Can be an integer or a string like 'exponential'.
        """
        min_qubits, max_qubits = qubit_range
        min_depth, max_depth = depth_range
        
        # Number of circuits to generate for each qubit value
        num_qubit_values = max_qubits - min_qubits + 1
        circuits_per_qubit = dataset_size // num_qubit_values

        for num_qubits in range(min_qubits, max_qubits + 1):
            generated_count = 0
            while generated_count < circuits_per_qubit:
                # Select a random depth within the provided range
                depth = random.randint(min_depth, max_depth)

                # Handle max_gates based on qubits if it's a string
                if isinstance(max_gates, str):
                    max_gates_value = self.compute_max_gates(num_qubits, max_gates)
                else:
                    max_gates_value = max_gates

                # Generate the circuit
                circuit = self.circuit_generator.generate_circuit(num_qubits=num_qubits, depth=depth, max_gates=max_gates_value)

                graph = QuantumCircuitGraph(circuit).graph
                
                # Perform checks 
                if self.check_equivalence(circuit) and self.check_connectivity(graph):
                    # If the circuit passes checks, add it to the dataset
                    self.dataset.append((circuit, graph))
                    generated_count += 1

        print(f"Dataset generated with {len(self.dataset)} circuits.")


    def check_equivalence(self, circuit):
        """
        Check if the generated circuit is not equivalent to another in the dataset.
        
        Steps:
        1. Get the statevector from the classical simulator.
        2. Check if the statevector has complex entries.
        3. Sort and compare the statevector with others of the same length in the dataset.
        """
        # Simulate the statevector for the circuit
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circuit, backend)
        result = job.result()
        statevector = np.array(result.get_statevector())

        # Check if any complex entries are in the statevector
        if any(np.iscomplex(statevector)):
            print("Warning: Circuit has complex entries in the statevector, failing the equivalence check: equivalence check for complex statevectors not supported.")
            return False

        # Sort the statevector entries
        ordered_statevector = np.sort(statevector)

        # Get the number of qubits in the circuit
        num_qubits = circuit.num_qubits

        # Retrieve the list of statevectors for this qubit count, if any exist
        if num_qubits in self.statevectors:
            dataset_statevectors = self.statevectors[num_qubits]
        else:
            dataset_statevectors = []

        # Compare the sorted statevector with each one in the dataset
        for sv in dataset_statevectors:
            # Compare within a given tolerance
            if not np.allclose(ordered_statevector, sv, atol=1e-5):
                # If at least one entry is different, the circuits are not equivalent: the check goes on
                continue
            else:
                # Circuits are equivalent, fail the check
                return False

        # If no equivalent circuit was found, pass the check and store the statevector
        self.statevectors.setdefault(num_qubits, []).append(ordered_statevector)
        return True

    def check_connectivity(self, graph):
        """
        Check if the graph associated to the generated quantum circuit is connected.
        """
        return nx.is_weakly_connected(graph)

    def compute_max_gates(self, num_qubits, strategy):
        """
        Compute the maximum number of gates based on the number of qubits and a strategy.
        
        Parameters:
        num_qubits (int): Number of qubits in the circuit.
        strategy (str): Strategy for determining max gates (e.g., 'exponential').
        
        Returns:
        int: Maximum number of gates.
        """
        if strategy == 'exponential':
            # Placeholder for exponential function
            return 2 ** num_qubits
        else:
            raise ValueError(f"Unknown max_gates strategy: {strategy}")
        
    def set_graph_config(self, graph_config):
        """
        Set the configuration for the QuantumCircuitGraph class.

        Parameters:
        graph_config (dict): Configuration parameters for the QuantumCircuitGraph class.
        """
        QuantumCircuitGraph.set_gate_type_map(graph_config.gate_type_map)
        QuantumCircuitGraph.set_include_params(graph_config.include_params)
        QuantumCircuitGraph.set_include_identity_gates(graph_config.include_identity_gates)
        QuantumCircuitGraph.set_differentiate_cx(graph_config.differentiate_cx)

