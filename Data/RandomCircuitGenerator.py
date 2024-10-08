from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import (IGate, U1Gate, U2Gate, U3Gate, XGate,
                                                   YGate, ZGate, HGate, SGate, SdgGate, TGate,
                                                   TdgGate, RXGate, RYGate, RZGate, CXGate,
                                                   CYGate, CZGate, CHGate, CRZGate, CU1Gate,
                                                   CU3Gate, SwapGate, RZZGate,
                                                   CCXGate, CSwapGate)
from qiskit.circuit import Reset
from qiskit.exceptions import QiskitError
import numpy as np
import random

class RandomCircuitGenerator:
    def __init__(self, num_qubits, depth, gate_pool=None, max_depth=None):
        """
        Initialize the RandomCircuitGenerator with parameters.
        
        Parameters:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Default depth of the circuit.
        gate_pool (list): List of gates to be used in the random circuit.
        max_depth (int): Maximum depth of the circuit.
        connected (bool): Whether all qubits have to be connected.
        """
        self.num_qubits = num_qubits
        self.depth = depth
        self.gate_pool = gate_pool if gate_pool is not None else [HGate, XGate, ZGate, CXGate, TGate]
        self.max_depth = max_depth
        self.circuits = []

    def set_gate_pool(self, gate_pool):
        """
        Set the pool of gates to be used in the random circuit.
        
        Parameters:
        gate_pool (list): List of gates to be used in the random circuit.
        """
        self.gate_pool = gate_pool

    # def generate_circuit(self, num_qubits=None, depth=None):
    #     """
    #     Generate a single random quantum circuit using the specified gate pool.
        
    #     Parameters:
    #     num_qubits (int): Number of qubits in the circuit. Uses the instance's num_qubits if not provided.
    #     depth (int): Depth of the circuit. Uses the instance's depth if not provided.
        
    #     Returns:
    #     QuantumCircuit: A randomly generated quantum circuit.
    #     """
    #     if num_qubits is None:
    #         num_qubits = self.num_qubits
    #     if depth is None:
    #         depth = self.depth

    #     if self.max_depth and depth > self.max_depth:
    #         depth = self.max_depth

    #     # max_operands=2 ensures that only 1 or 2 qubit gates are used
    #     circuit = random_circuit(num_qubits, depth, max_operands=2, op_list=self.gate_pool)

    #     return circuit

    # def generate_circuits(self, n_samples, min_qubits=2, max_qubits=None, min_depth=2, max_depth=None):
    #     """
    #     Generate a specified quantity of random quantum circuits.
        
    #     Parameters:
    #     n_samples (int): Number of circuits to generate.
    #     min_qubits (int): Minimum number of qubits in the circuits.
    #     max_qubits (int): Maximum number of qubits in the circuits. Defaults to instance's num_qubits.
    #     min_depth (int): Minimum depth of the circuits.
    #     max_depth (int): Maximum depth of the circuits. Defaults to instance's max_depth.
    #     """
    #     if max_qubits is None:
    #         max_qubits = self.num_qubits
    #     if max_depth is None:
    #         max_depth = self.max_depth

    #     for _ in range(n_samples):
    #         num_qubits = random.randint(min_qubits, max_qubits)
    #         depth = random.randint(min_depth, max_depth)
    #         circuit = self.generate_circuit(num_qubits, depth)
    #         self.circuits.append(circuit)

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



def random_circuit(num_qubits, depth, max_operands=2, seed=None, op_list=None):  
    """Generate random circuit of arbitrary size and form.

    Modified from qiskit.circuit.random.random_circuit to allow for custom gate pools.
    This function will generate a random circuit by randomly selecting gates
    from the set provided in `op_list`. If `op_list` is not provided, the   
    function will use a default set of gates.

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length)
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

    print(f"Generating random circuit with {num_qubits} qubits and depth {depth}...\n")

    # apply arbitrary random operations at every depth
    for d in range(depth):
        print("\n", "-" * 50)
        print(f"DEPTH {d}:")
        # choose how many qubits and which ones will be used in this layer
        if rng.random() < prob_unconnected:
            remaining_qubits = list(rng.choice(num_qubits, rng.integers(2, num_qubits + 1), replace=False))
        else:
            remaining_qubits = list(rng.choice(num_qubits, rng.integers(1, num_qubits + 1), replace=False))

        # apply random operations until all qubits have been used
        while remaining_qubits:
            # Increase probability for two-qubit gates as depth increases
            prob_unconnected = 0 if len(connected_qubits) == num_qubits else 1 - 0.1 * len(connected_qubits) / num_qubits
            two_q_gate_prob = (d + 1) / depth * prob_unconnected

            print("Probability of two-qubit gate:", two_q_gate_prob)
            print("Probability of unconnected qubit:", prob_unconnected)

            print(f"Remaining qubits: {remaining_qubits}")

            # choose number of operands for the operation
            max_possible_operands = min(len(remaining_qubits), max_operands)

            if rng.random() < two_q_gate_prob and max_possible_operands > 1:
                num_operands = 2
            else:
                num_operands = rng.choice(range(1, max_possible_operands + 1))

            print(f"Choosing {num_operands} operands...")

            # bias selection towards unconnected qubits
            if num_operands == 2 and connected_qubits and rng.random() < prob_unconnected:
                    print("Bias towards unconnected qubits...")
                    # Select an unconnected qubit
                    unconnected_qubit = rng.choice(list(set(range(num_qubits)) - connected_qubits))
                    operands = [unconnected_qubit, rng.choice(list(connected_qubits))]

            # else choose random operands
            else:
                print("Choosing random operands...")
                rng.shuffle(remaining_qubits)
                operands = remaining_qubits[:num_operands]

            print(f"Operands: {operands}")

            # remove operands from the list of remaining qubits
            remaining_qubits = [q for q in remaining_qubits if q not in operands]

            # choose random operation
            if num_operands == 1 and one_q_ops:
                operation = rng.choice(one_q_ops)
            elif num_operands == 2 and two_q_ops:
                operation = rng.choice(two_q_ops)
                connected_qubits.update(operands) # update connected qubits
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
            print(f"Applying {op} to {register_operands}\n")
            qc.append(op, register_operands)

    return qc
