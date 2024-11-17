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

class RandomCircuitGenerator:
    def __init__(self, gate_pool=None, max_depth=None, qubits_range=(3,20)):
        """
        Initialize the RandomCircuitGenerator with parameters.
        
        Parameters:
        num_qubits (int): Number of qubits in the circuit.
        depth (int): Default depth of the circuit.
        gate_pool (list): List of gates to be used in the random circuit.
        max_depth (int): Maximum depth of the circuit.
        connected (bool): Whether all qubits have to be connected.
        """
        self.gate_pool = gate_pool or [HGate, XGate, ZGate, CXGate, TGate]
        self.max_depth = max_depth
        self.qubits_range = qubits_range

    def generate_circuit(self, num_qubits=None, depth=None, max_gates=None, seed=None, return_num_gates=False, verbose=False):
        """
        Generate a single random quantum circuit using the specified gate pool.
        
        Parameters:
        num_qubits (int): Number of qubits in the circuit. Uses the instance's num_qubits if not provided.
        depth (int): Depth of the circuit. Uses the instance's depth if not provided.
        
        Returns:
        QuantumCircuit: A randomly generated quantum circuit.
        """
        # set random seed if provided
        if seed:
            random.seed(seed)

        num_qubits = num_qubits or random.randint(*self.qubits_range)

        if depth is None:
            depth = self.max_depth
        if self.max_depth and depth > self.max_depth:
            depth = self.max_depth

        # max_operands=2 ensures that only 1 or 2 qubit gates are used
        if return_num_gates:
            return random_circuit(num_qubits, depth, max_gates=max_gates, max_operands=2, op_list=self.gate_pool, return_num_gates=return_num_gates)
        return random_circuit(num_qubits, depth, max_gates=max_gates, max_operands=2, op_list=self.gate_pool)
    


    def generate_circuits(self, n_samples, qubit_range=None, depth_range=None, max_gates=None, seed=None):
        """
        Generate a specified quantity of random quantum circuits.
        
        Parameters:
        n_samples (int): Number of circuits to generate.
        qubit_range (tuple): Range of qubits (min, max). If max is None, defaults to instance's num_qubits.
        depth_range (tuple): Range of depth (min, max). If max is None, defaults to instance's max_depth.
        """
        # set random seed if provided
        if seed:
            random.seed(seed)

        if isinstance(qubit_range, int):
            min_qubits = max_qubits = qubit_range
        elif qubit_range is None:
            min_qubits, max_qubits = self.qubits_range
        else:
            min_qubits, max_qubits = qubit_range

        if isinstance(depth_range, int):
            min_depth = max_depth = depth_range
        elif depth_range is None:
            min_depth = 2
            max_depth = self.max_depth
        else:
            min_depth, max_depth = depth_range

        print(f"Generating {n_samples} random circuits with qubits in range {min_qubits}-{max_qubits} and depth in range {min_depth}-{max_depth}...\n")

        circuits = []
        for _ in range(n_samples):
            num_qubits = random.randint(min_qubits, max_qubits)
            depth = random.randint(min_depth, max_depth)
            circuit = self.generate_circuit(num_qubits, depth, max_gates=max_gates)
            circuits.append(circuit)
        
        return circuits




def random_circuit(num_qubits, depth, max_gates=None, max_operands=2, seed=None, op_list=None, return_num_gates=False, verbose=False):
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
    if seed is not None:
        np.random.seed(seed)
    
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

    if verbose:
        print(f"Generating random circuit with {num_qubits} qubits and depth {depth}...")

    # apply arbitrary random operations at every depth
    for d in range(depth):
        # print(f"Depth {d + 1}/{depth}:")
        if max_gates is not None and gate_count >= max_gates:
            if verbose:
                print(f"Maximum number of gates ({max_gates}) reached, stopping early.")
            break
        # choose how many qubits and which ones will be used in this layer
        if rng.random() < prob_unconnected:
            min_q = max(2, (num_qubits - len(connected_qubits)) - (depth - d))
            # print(f"Forcing at least {min_q} active qubits.")
            n_active_qubits = rng.integers(min_q, num_qubits + 1)
        else:
            n_active_qubits = rng.integers(1, num_qubits + 1)

        active_qubits = set()

        # apply random operations until all qubits have been used
        while n_active_qubits - len(active_qubits) > 0:
            # Increase probability for two-qubit gates as depth increases
            prob_unconnected = 0 if len(connected_qubits) == num_qubits else 1 - 0.2 * len(connected_qubits) / num_qubits
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

    if return_num_gates:
        return qc, gate_count
    return qc
