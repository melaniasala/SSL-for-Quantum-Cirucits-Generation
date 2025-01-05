import json
import pickle
import random
import time
from math import ceil
import os
import traceback
from datetime import datetime

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from qiskit.quantum_info import Statevector
from .QuantumCircuitGraph import QuantumCircuitGraph
from .RandomCircuitGenerator import RandomCircuitGenerator


class DatasetGenerator:
    def __init__(self, circuit_generator=None, save_format='pickle', graph_config=None, save_path=None):
        """
        Initialize the DatasetGenerator class with parameters.

        Parameters:
        circuit_generator (RandomCircuitGenerator): Instance of the RandomCircuitGenerator class to generate circuits.
        save_format (str): Format to save the dataset in, default is 'pickle'.
        graph_config (dict): Optional configuration for the graph representation of quantum circuits.
        save_path (str): Path to save the dataset incrementally. Default will be based on save_format.
        """
        self.circuit_generator = circuit_generator if circuit_generator is not None else RandomCircuitGenerator()
        self.save_format = save_format
        self.dataset = []
        self.statevectors = {}
        self.generation_progress = {}  # To track circuits generated for each qubit count
        self.verbose = False

        # Default save path based on the chosen format
        if save_path is None:
            extension = {'json': 'json', 'pickle': 'pkl'}.get(self.save_format, 'txt')
            self.save_path = f"dataset_in_progress.{extension}"
        else:
            self.save_path = save_path

        if graph_config is not None:
            self.set_graph_config(graph_config)
        
        # Load existing dataset if it exists
        self._load_existing_dataset()

    def _load_existing_dataset(self):
        """Load an existing dataset from file if it exists, allowing resumption."""
        if os.path.exists(self.save_path):
            try:
                if self.save_format == 'json':
                    with open(self.save_path, 'r') as file:
                        saved_data = json.load(file)
                elif self.save_format == 'pickle':
                    with open(self.save_path, 'rb') as file:
                        saved_data = pickle.load(file)
                else:
                    print(f"Unsupported format for loading: {self.save_format}")
                    return

                self.dataset = saved_data.get('dataset', [])
                self.generation_progress = saved_data.get('generation_progress', {})
                self.statevectors = saved_data.get('statevectors', {})  
                print(f"Loaded {len(self.dataset)} previously generated circuits from {self.save_path}.")
            except Exception as e:
                print(f"Error loading dataset: {str(e)}")
        else:
            print("No existing dataset found, starting fresh.")


    def generate_dataset(self, dataset_size=None, qubit_range=(2, 5), depth_range=(2, 10), max_gates=None, circuits_per_qubit=None, save_interval=10, verbose=False):
        """
        Generate a dataset of random quantum circuits.

        Parameters:
        dataset_size (int): Total number of circuits to generate in the dataset.
        qubit_range (tuple): Range of qubits (min, max) for the circuits.
        depth_range (tuple): Range of depth (min, max) for the circuits.
        max_gates (int or str): Maximum number of gates allowed per circuit.
                                Can be an integer or a string like 'exponential'.
        circuits_per_qubit (dict): Optional dictionary specifying the number of circuits to generate for each qubit count.
                                   Example: {2: 10, 3: 15, 4: 20}.
        save_interval (int): Interval for incremental saving, in terms of number of circuits generated.
        """
        self.verbose = verbose
        min_qubits, max_qubits = qubit_range if isinstance(qubit_range, tuple) else (qubit_range, qubit_range)

        # Determine the number of circuits per qubit count if not specified
        if circuits_per_qubit is None:
            num_qubit_values = max_qubits - min_qubits + 1
            circuits_per_qubit = {q: dataset_size // num_qubit_values for q in range(min_qubits, max_qubits + 1)}

        self.target_dataset_size = sum(circuits_per_qubit.values())
        
        circuit_data = []

        failed_saves = 0

        for num_qubits, count in circuits_per_qubit.items():
            # Define depth range incrementally if specified
            if depth_range == 'incremental':
                min_depth, max_depth = self._determine_depth_range(num_qubits)
            else:
                min_depth, max_depth = depth_range

            # Load progress or initialize if starting fresh
            generated_count = self.generation_progress.get(num_qubits, 0)
            remaining_count = count - generated_count
            print(f"Generating {remaining_count} more circuits for {num_qubits} qubits (already generated {generated_count}).")
            
            while generated_count < count:
                depth = random.randint(min_depth, max_depth)
                max_gates_value = self.compute_max_gates(num_qubits, max_gates) if isinstance(max_gates, str) else max_gates

                # Generate the circuit
                start_time = time.time()
                circuit, num_gates = self.circuit_generator.generate_circuit(num_qubits=num_qubits, depth=depth, max_gates=max_gates_value, return_num_gates=True, verbose=verbose)
                qcg = QuantumCircuitGraph(circuit)

                # Check connectivity and equivalence before adding
                if self.check_connectivity(qcg.graph) and self.check_equivalence(circuit):
                    generated_count += 1
                    self.dataset.append(qcg)
                    time_taken = time.time() - start_time

                    self.generation_progress[num_qubits] = generated_count

                    circuit_data.append({
                        'generation_time': time_taken,
                        'num_qubits': num_qubits,
                        'num_gates': num_gates,
                        'depth': depth
                    }) # TODO: do not only return, but also save

                    # Save incrementally every `save_interval` circuits
                    if len(self.dataset) % save_interval == 0:
                        save_success = self.save_dataset(in_progress=True)
                        if not save_success:
                            failed_saves += 1
                            if failed_saves > 3:
                                print("Failed to save dataset multiple times, stopping generation.")
                                break

                    if self.verbose:
                        print(f"Added circuit {generated_count}/{count} for {num_qubits} qubits. Time: {time_taken:.4f}s")
                else:
                    if self.verbose:
                        print("Circuit failed checks, retrying generation...")

            print(f"Completed {count} circuits for {num_qubits} qubits. Total circuits: {len(self.dataset)}")

        print(f"Dataset generated with {len(self.dataset)} circuits.")
        self.save_dataset()
        print(f"Dataset saved successfully to {self.save_path}.")
        return circuit_data


    def check_equivalence(self, circuit):
        """
        Check if the generated circuit is not equivalent to another in the dataset.
        
        Steps:
        1. Get the statevector from the classical simulator.
        2. Check if the statevector has complex entries.
        3. Sort and compare the statevector with others of the same length in the dataset.
        """
        # Compute the statevector for the circuit
        statevector = Statevector(circuit).data

        # Set absolute and relative tolerances for the comparison
        atol = 1e-8
        rtol = 1e-5

        # Compute the probabilities for the statevector
        probabilities = np.abs(statevector) ** 2

        # Sort the statevector entries (modulus squared) in ascending order
        ordered_statevector = np.sort(probabilities) 

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
            if not np.allclose(ordered_statevector, sv, atol=atol, rtol=rtol):
                # If at least one entry is different, the circuits are not equivalent: the check goes on
                continue
            else:
                # Circuits are equivalent, fail the check
                if self.verbose:
                    print("Warning: Circuit is equivalent to another in the dataset, failing the equivalence check.")
                return False

        # If no equivalent circuit was found, pass the check and store the statevector
        self.statevectors.setdefault(num_qubits, []).append(ordered_statevector)
        if self.verbose:
            print("Circuit is not equivalent to any other in the dataset, passing the equivalence check.")
        return True

    def check_connectivity(self, graph):
        """
        Check if the graph associated to the generated quantum circuit is connected.
        """
        result = nx.is_weakly_connected(graph)
        if self.verbose:
            if nx.is_weakly_connected(graph):
                print("Circuit graph is connected, passing the connectivity check.")
            else:
                print("Warning: Circuit graph is not connected, failing the connectivity check.")
        
        return result

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
            return 2 ** (num_qubits-1)
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

    def save_dataset(self, file_path=None, include_statevectors=False, in_progress=False):
        """
        Save the dataset, generation progress, and optionally statevectors to a file.

        Parameters:
        file_path (str): Path where the dataset will be saved.
        include_statevectors (bool): Whether to include statevectors in the saved dataset.
        in_progress (bool): Flag for in-progress saving, includes statevectors by default.
        """
        if in_progress:
            include_statevectors = True  # Always include statevectors in in-progress saves

        # Prepare dataset content
        dataset_content = {
            'dataset': self.dataset,
            'generation_progress': self.generation_progress
        }
        if include_statevectors:
            dataset_content['statevectors'] = self.statevectors

        # Define file path with appropriate format extension
        file_path = file_path or self._generate_file_path(in_progress)

        # Use a temporary file for safer incremental saving
        temp_file_path = f"{file_path}.tmp"
        save_successful = False

        try:
            save_successful = self._save_to_file(temp_file_path, dataset_content)
            
            if save_successful:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_file_path = f"{file_path}_{timestamp}"

                try:
                    if os.path.exists(file_path):
                        os.rename(file_path, unique_file_path)
                    os.rename(temp_file_path, file_path)
                    print(f"Dataset successfully saved to {file_path}")
                except Exception as e:
                    print("Error: Failed to rename temporary file.")
                    print(f"Exception details: {str(e)}")
                    print(traceback.format_exc())  # Print full traceback
                    return False
                
                if os.path.exists(unique_file_path):
                    os.remove(unique_file_path)  # Remove old file if renaming was successful
            else:
                print("Error: Failed to save dataset content to the temporary file.")

        except Exception as e:
            print("Error: Exception occurred during dataset saving.")
            print(f"Exception details: {str(e)}")
            print(traceback.format_exc())  # Print full traceback
        
        # Cleanup in case temp file was not renamed
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return True


    def _generate_file_path(self, in_progress=False):
        """Generate a file path with the correct format extension."""
        extension = {'json': 'json', 'pickle': 'pkl'}.get(self.save_format, 'txt')
        if in_progress:
            return f"dataset_{self.target_dataset_size}_in_progress.{extension}"
        else:
            return f"dataset_{self.target_dataset_size}.{extension}"


    def _save_to_file(self, file_path, data):
        """Save data to file based on the format specified."""
        try:
            if self.save_format == 'json':
                with open(file_path, 'w') as file:
                    json.dump(data, file, indent=4)
            elif self.save_format == 'pickle':
                with open(file_path, 'wb') as file:
                    pickle.dump(data, file)
            else:
                with open(file_path, 'w') as file:
                    file.write(str(data))  # Fall back to a text-based save
            return True  # Indicate successful save
        except Exception as e:
            print(f"Error saving dataset: {str(e)}")
            return False  # Indicate failed save

    def _determine_depth_range(self, num_qubits):
        """
        Determine the depth range based on the number of qubits. For small qubit counts, 
        a default range is applied. For larger qubit counts, the max depth grows exponentially.

        Parameters:
        num_qubits (int): The number of qubits in the circuit.

        Returns:
        tuple: A tuple (min_depth, max_depth) representing the depth range.
        """
        # Define threshold for switching from default range to exponential scaling
        qubit_threshold = 10
        default_min_depth = 4
        default_max_factor = 6

        if num_qubits <= qubit_threshold:
            # Apply a fixed default depth range for smaller qubit numbers
            min_depth = default_min_depth
            max_depth = int(num_qubits * default_max_factor) # Linear scaling
        else:
            # Use an exponential function for larger qubit numbers
            min_depth = default_min_depth  # Small fixed minimum depth
            max_depth = int(2 ** num_qubits / num_qubits) # Exponential scaling

        return min_depth, max_depth
    

    def load_dataset(self, file_path):
        self.save_path = file_path
        self._load_existing_dataset()

        print(f"Loaded {len(self.dataset)} circuits from {file_path}.")
        self.display_current_progress()

    def display_current_progress(self):
        """Display the current progress of circuit generation for each qubit count."""
        print("Current progress:")
        for num_qubits, count in self.generation_progress.items():
            print(f"{count} circuits generated for {num_qubits} qubits.")


    def visualize_circuit(self, index=None, graph_visual_mode='circuit-like'):
        """
        Visualize a specific circuit or its corresponding graph from the dataset.

        Parameters:
        index (int): Index of the circuit in the dataset to visualize (default is 0, the first one).
        graph_visual_mode (str): Mode for visualizing the graph. 
            - 'circuit-like' (default): Visualizes the graph in a way that resembles the quantum circuit structure.
            - 'graph': Visualizes the graph without the circuit-like layout.
            - None: Visualizes the quantum circuit itself (Qiskit's circuit diagram).
        
        If the index is out of range, an error message will be displayed.
        """
        if not index:
            # random index if not provided
            index = random.randint(0, len(self.dataset) - 1)
        if index >= len(self.dataset):
            print(f"Index {index} is out of range. There are {len(self.dataset)} circuits in the dataset.")
            return
        
        if graph_visual_mode is None:
            self.dataset[index].circuit.draw(output='mpl')

        elif graph_visual_mode == 'circuit-like':
            self.dataset[index].draw_circuit_and_graph(circuit_like_graph=True)

        elif graph_visual_mode == 'graph':
            self.dataset[index].draw_circuit_and_graph(circuit_like_graph=False)

    def visualize_dataset(self, num_samples=12, graph_visual_mode='circuit-like'):
        """
        Visualize a sample of circuits and their corresponding graphs from the dataset.

        Parameters:
        num_samples (int): Number of circuits to visualize. Defaults to 12.
        graph_visual_mode (str): Mode for visualizing the graph. 
            - 'circuit-like' (default): Visualizes the graph to resemble the quantum circuit structure.
            - 'graph': Visualizes the graph layout without the circuit-like appearance.
            - None: Visualizes the quantum circuit itself (Qiskit's circuit diagram).
        """
        if not self.dataset:
            print("No circuits have been generated yet. Please generate circuits first.")
            return
        
        # Adjust number of samples if more requested than available
        num_samples = min(num_samples, len(self.dataset))
        visual_mode = {'circuit-like': True, 'graph': False}

        if graph_visual_mode:
            samples_per_row = 1
            columns = 2 * samples_per_row  # Each sample gets 2 subplots (circuit and graph)
        else:
            samples_per_row = 2
            columns = samples_per_row  # Only circuits are visualized

        rows = ceil(num_samples / samples_per_row)
        add_height = 5
        fig, axs = plt.subplots(rows, columns, figsize=(15, num_samples*4 + add_height*(1/(rows)**2)))

        if rows == 1:
            axs = np.expand_dims(axs, axis=0)
        
        for i in range(num_samples):
            row = i // samples_per_row
            col = i % samples_per_row
            
            if graph_visual_mode:
                col_start = 2 * col  # Circuit on one, graph on the next
                self.dataset[i].draw_circuit_and_graph(
                    circuit_like_graph=visual_mode[graph_visual_mode], 
                    axes=axs[row, col_start:col_start + 2],
                    titles=False
                )
            else:
                # Circuit-only visualization
                self.dataset[i].quantum_circuit.draw(output='mpl', ax=axs[row, col])


        plt.tight_layout()
        plt.axis('off')
        plt.show()
