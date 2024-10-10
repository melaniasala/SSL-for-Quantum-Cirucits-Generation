from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import (IGate, U1Gate, U2Gate, U3Gate, XGate,
                                                   YGate, ZGate, HGate, SGate, SdgGate, TGate,
                                                   TdgGate, RXGate, RYGate, RZGate, CXGate,
                                                   CYGate, CZGate, CHGate, CRZGate, CU1Gate,
                                                   CU3Gate, SwapGate, RZZGate,
                                                   CCXGate, CSwapGate)
from qiskit.exceptions import QiskitError
import numpy as np
import random
from math import ceil
import matplotlib.pyplot as plt

class RandomCircuitGenerator:
    def __init__(self, gate_pool=None, max_depth=None, max_gates=None):
        """
        Initialize the RandomCircuitGenerator with parameters.
        
        Parameters:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Default depth of the circuit.
        gate_pool (list): List of gates to be used in the random circuit.
        max_depth (int): Maximum depth of the circuit.
        connected (bool): Whether all qubits have to be connected.
        """
        self.gate_pool = gate_pool if gate_pool is not None else [HGate, XGate, ZGate, CXGate, TGate]
        self.max_depth = max_depth
        self.max_n_gates = max_gates

        self.circuits = []

    def set_gate_pool(self, gate_pool):
        """
        Set the pool of gates to be used in the random circuit.
        
        Parameters:
        gate_pool (list): List of gates to be used in the random circuit.
        """
        self.gate_pool = gate_pool

    def generate_circuit(self, num_qubits=None, depth=None):
        """
        Generate a single random quantum circuit using the specified gate pool.
        
        Parameters:
        num_qubits (int): Number of qubits in the circuit. Uses the instance's num_qubits if not provided.
        depth (int): Depth of the circuit. Uses the instance's depth if not provided.
        
        Returns:
        QuantumCircuit: A randomly generated quantum circuit.
        """
        if num_qubits is None:
            num_qubits = random.randint(3, 20)
        if depth is None:
            depth = self.max_depth

        if self.max_depth and depth > self.max_depth:
            depth = self.max_depth

        # max_operands=2 ensures that only 1 or 2 qubit gates are used
        circuit = random_circuit(num_qubits, depth, max_operands=2, op_list=self.gate_pool)

        return circuit
    


    def generate_circuits(self, n_samples, qubit_range=(2, None), depth_range=(2, None), max_gates=None):
        """
        Generate a specified quantity of random quantum circuits.
        
        Parameters:
        n_samples (int): Number of circuits to generate.
        qubit_range (tuple): Range of qubits (min, max). If max is None, defaults to instance's num_qubits.
        depth_range (tuple): Range of depth (min, max). If max is None, defaults to instance's max_depth.
        """
        if isinstance(qubit_range, int):
            min_qubits = max_qubits = qubit_range
        elif qubit_range is None:
            min_qubits = max_qubits = self.num_qubits
        else:
            min_qubits, max_qubits = qubit_range
            if max_qubits is None:
                max_qubits = self.num_qubits

        min_depth, max_depth = depth_range
        if max_depth is None:
            max_depth = self.max_depth

        if max_gates is None:
            max_gates = self.max_gates

        for _ in range(n_samples):
            num_qubits = random.randint(min_qubits, max_qubits)
            depth = random.randint(min_depth, max_depth)
            circuit = self.generate_circuit(num_qubits, depth, max_gates)
            self.circuits.append(circuit)



    def draw_circuit(self, index=0):
        """
        Draw a specific generated random quantum circuit.
        
        Parameters:
        index (int): Index of the circuit in the list to draw.
        """
        if self.circuits and index < len(self.circuits):
            self.circuits[index].draw('mpl')
        else:
            print("No circuit has been generated yet or index out of range. Please generate a circuit first.")


    def draw_circuits(self):
        """
        Draw all generated random quantum circuits.
        """
        if self.circuits:
            n_samples = len(self.circuits)
            samples_per_row = 5
            n_rows = ceil(n_samples / samples_per_row)
            add_height = 2
            fig, axs = plt.subplots(n_rows, samples_per_row, figsize=(15, n_samples + add_height*(1/(n_rows)**2)))

        if n_rows == 1:
            axs = np.expand_dims(axs, axis=0)

        # dataset is a list of tuples (qc, qcg)
        for i, qc in enumerate(self.circuits):
            row = i // samples_per_row
            col = i % samples_per_row

            qc.draw('mpl', ax=axs[row, col])

            plt.tight_layout()
            plt.axis('off')
            plt.show()
        else:
            print("No circuits have been generated yet. Please generate circuits first.")




def random_circuit(num_qubits, depth, max_gates=None, max_operands=2, seed=None, op_list=None):  
    """Generate random circuit of arbitrary size and form, with a maximum number of gates.

    Modified from qiskit.circuit.random.random_circuit to allow for custom gate pools and
    to stop gate additions once the depth or the maximum number of gates is exhausted.

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
        max_gates (int): maximum number of gates allowed in the circuit (optional).
                         If None, there is no limit on the number of gates.
        max_operands (int): maximum operands of each gate (between 1 and 3)
        seed (int): sets random seed (optional)
        op_list (list): list of operations to choose from (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    
    if max_operands < 1 or max_operands > 3:
        raise QiskitError("max_operands must be between 1 and 3")
    
    if num_qubits < 2:
        raise QiskitError("Number of qubits must be at least 2")

    one_q_ops = [IGate, U1Gate, U2Gate, U3Gate, XGate, YGate, ZGate,
                 HGate, SGate, SdgGate, TGate, TdgGate, RXGate, RYGate, RZGate]
    two_q_ops = [CXGate, CYGate, CZGate, CHGate, CRZGate,
                 CU1Gate, CU3Gate, SwapGate, RZZGate]
    three_q_ops = [CCXGate, CSwapGate]

    one_param = [U1Gate, RXGate, RYGate, RZGate, RZZGate, CU1Gate, CRZGate]
    two_param = [U2Gate]
    three_param = [U3Gate, CU3Gate]

    if op_list is not None:
        one_q_ops = [op for op in one_q_ops if op in op_list]
        two_q_ops = [op for op in two_q_ops if op in op_list]
        three_q_ops = [op for op in three_q_ops if op in op_list]

        one_param = [op for op in one_param if op in op_list]
        two_param = [op for op in two_param if op in op_list]
        three_param = [op for op in three_param if op in op_list]

    qr = QuantumRegister(num_qubits, 'q')
    qc = QuantumCircuit(num_qubits)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    # track qubit connectivity
    connected_qubits = set()
    prob_unconnected = 1.0
    gate_count = 0

    print(f"Generating random circuit with {num_qubits} qubits and depth {depth}...\n")

    # apply arbitrary random operations at every depth
    for d in range(depth):
        if max_gates is not None and gate_count >= max_gates:
            print(f"Maximum number of gates ({max_gates}) reached, stopping early.")
            break
        # choose how many qubits and which ones will be used in this layer
        if rng.random() < prob_unconnected:
            print("Forcing 2 or more qubit gates to connect qubits.")
            n_active_qubits = rng.integers(2, num_qubits + 1)
        else:
            n_active_qubits = rng.integers(1, num_qubits + 1)

        active_qubits = set()

        # apply random operations until all qubits have been used
        while n_active_qubits - len(active_qubits) > 0:
            # Increase probability for two-qubit gates as depth increases
            prob_unconnected = 0 if len(connected_qubits) == num_qubits else 1 - 0.1 * len(connected_qubits) / num_qubits
            two_q_gate_prob = (d + 1) / depth * prob_unconnected

            # choose number of operands for the operation
            max_possible_operands = min(n_active_qubits - len(active_qubits), max_operands)

            if rng.random() < two_q_gate_prob and max_possible_operands > 1:
                num_operands = 2
            else:
                num_operands = rng.choice(range(1, max_possible_operands + 1))

            non_active_qubits = set(range(num_qubits)) - active_qubits

            # bias selection towards unconnected qubits
            if num_operands == 2:
                if (non_active_qubits - connected_qubits) and rng.random() < prob_unconnected:
                    if (connected_qubits - active_qubits):
                        operand1 = rng.choice(list(non_active_qubits - connected_qubits)) # unconnected qubit
                        operand2 = rng.choice(list(connected_qubits - active_qubits)) # connected qubit
                    else:
                        operand1, operand2 = rng.choice(list(non_active_qubits - connected_qubits), 2, replace=False) # two unconnected qubits
                        
                    connected_qubits.update([operand1])
                    operands = [operand1, operand2]

            # else choose random operands among qubits not active in this layer
                else:
                    operands = rng.choice(list(non_active_qubits), num_operands, replace=False)
                    connected_qubits.update(operands)

            else:
                operands = rng.choice(list(non_active_qubits), num_operands, replace=False)
                

            active_qubits.update(operands) # update active qubits

            # choose random operation
            if num_operands == 1 and one_q_ops:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2 and two_q_ops:
                operation = rng.choice(two_q_ops)
            elif num_operands == 3 and three_q_ops:
                operation = rng.choice(three_q_ops)

            # choose random angles for the operation (if parametric gate)
            if operation in one_param:
                num_angles = 1
            elif operation in two_param:
                num_angles = 2
            elif operation in three_param:
                num_angles = 3
            else:
                num_angles = 0
            angles = [rng.uniform(0, 2 * np.pi) for _ in range(num_angles)]

            # apply operation to the chosen operands
            register_operands = [qr[i] for i in operands]
            op = operation(*angles)
            qc.append(op, register_operands)

            gate_count += 1

    return qc
