import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector, Barrier
import matplotlib.pyplot as plt

class QUBO:
    """
    Class of QUBO problems
    To generate an instance, put as an input a square symmetric matrix.
    """

    def __init__(self, matrix=None):
        if matrix is None:
            matrix = np.zeros((2,2))
        self.Q = matrix
        self.n = len(matrix)

    def get_linear_terms(self):
        """
        Returns the linear terms of the QAOA Hamiltonian.
        :return: array containing the diagonal elements of Q
        """
        return np.diag(self.Q)

    def get_quadratic_terms(self):
        """
        Returns the quadratic terms of the QAOA Hamiltonian.
        :return: matrix containing off-diagonal elements of Q
        """
        return self.Q - np.diag(self.get_linear_terms())

    def to_ising(self):
        """
        Transform the QUBO problem into an Ising problem.
        QUBO formulation (binary variables, x = 0, 1) 
        -> Q(x) = x^T Q x
        Ising formulation (spin variables, s = +1/2, -1/2) 
        -> H(s) = s^T J s + h^T s + offset
        with s = 2x - 1
        :return: Ising object
        """
        linear_terms = self.get_linear_terms()
        quadratic_terms = self.get_quadratic_terms()

        ones = np.ones(self.n)

        # Calculate coupling (J)
        # J_ij = 1/4 * Q_ij
        coupling = 1/4 * quadratic_terms 

        # Calculate bias (h)
        # h_i = 1/4 * sum_j (Q_ij + Q_ji) + 1/2 * Q_ii
        bias = (1/4 * (np.reshape(ones, (1, self.n)) @ (quadratic_terms + quadratic_terms.T) + 2 * linear_terms))[0]

        # Calculate offset
        # offset = 1/4 * sum_ij (Q_ij) + 1/2 * sum_i (Q_ii)
        offset = 1/4 * ones @ quadratic_terms @ ones + 1/2 * linear_terms @ ones

        return Ising(coupling, bias, offset)


class Ising:

    def __init__(self, coupling, bias, offset):

        self.coupling = coupling
        self.bias = bias
        self.offset = offset


def qaoa_composer(my_qubo, n_layers=1, put_barriers=False, draw_circuit=False):
    assert isinstance(my_qubo, QUBO)

    # Transform qubo into ising
    my_ising = my_qubo.to_ising()

    coupling = my_ising.coupling
    bias = my_ising.bias

    non_zero_elements_coupling = np.count_nonzero(coupling)
    non_zero_elements_bias = np.count_nonzero(bias)

    n_gamma = n_layers * (non_zero_elements_coupling // 2 + non_zero_elements_bias)
    n_beta = n_layers * my_qubo.n

    gamma = ParameterVector('gamma', n_gamma)
    beta = ParameterVector('beta', n_beta)

    # gamma parameter counter
    gamma_counter = 0

    # Create Quantum Register and circuit
    circuit = QuantumCircuit(my_qubo.n)

    for i in range(my_qubo.n):
        circuit.h(i)

    # Generation of layers
    for layer in range(n_layers):
        # -- COST UNITARY --
        # Linear Terms
        for i in range(0, my_qubo.n):
            if my_ising.bias[i] != 0:
                circuit.rz(my_ising.bias[i] * gamma[gamma_counter], i)
                gamma_counter += 1

        # Quadratic Terms
        for i in range(my_qubo.n):
            for j in range(i, my_qubo.n):
                if i != j and my_ising.coupling[i, j] != 0:
                    circuit.cx(i, j)
                    circuit.rz(2 * my_ising.coupling[i, j] * gamma[gamma_counter], j)
                    circuit.cx(i, j)

                    gamma_counter += 1

        if put_barriers:
            circuit.barrier()

        # -- MIXER UNITARY --
        for i in range(my_qubo.n):
            circuit.rx(2 * beta[layer * my_qubo.n + i], i)

        if put_barriers:
            circuit.barrier()

    circuit.draw(output='mpl')
    if draw_circuit:
        plt.show()
    return circuit, gamma, beta


def calculate_energy(counts, Q):
    """
    Calculate the energy of a particular bitstring
    :param counts: dictionary of counts
    :param Q: QUBO matrix
    :return: energy of the bitstring
    """
    energy = 0
    for bitstring, count in counts.items():
        x = np.array([int(bit) for bit in bitstring])
        energy += count * x.T @ Q @ x
    return energy / sum(counts.values())


# Cost Function to optimize parameters beta and gamma
# params contains the initialization of the beta and gamma paramters
def cost_function(params, circuit, gamma, beta, backend, shots=1024):
    """
    Compute the cost function of the QAOA algorithm, given the parameters beta and gamma.
    :param params: initial values for both gamma and beta parameters 
    :param circuit: parameterized quantum circuit
    :param gamma:  parameter placeholders for gamma parameters
    :param beta: parameter placeholders for beta parameters
    :param backend: quantum backend where the circuit will be executed
    :param shots: number of runs of the circuit 
    :return: computed energy (cost) based on the measurements
    """

    gamma_values = params[:len(gamma)]
    beta_values = params[len(gamma):]

    # Map gamma and beta parameters to their values
    param_dict = {}
    for gamma_param, gamma_value in zip(gamma, gamma_values):
        param_dict[gamma_param] = gamma_value
    for beta_param, beta_value in zip(beta, beta_values):
        param_dict[beta_param] = beta_value

    bound_circuit = circuit.bind_parameters(param_dict)

    # Add measurement operations to the circuit
    bound_circuit.measure_all()

    # Execute the circuit on the backend
    job = backend.run(bound_circuit, shots=shots)
    result = job.result()

    if not result.get_counts():
        raise Exception("No counts for experiment.")

    counts = result.get_counts()
    energy = calculate_energy(counts, Q)

    return energy


def get_optimal_solution(optimal_param, circuit, gamma, beta, backend, shots=4096):
    gamma_values = optimal_param[:len(gamma)]
    beta_values = optimal_param[len(gamma):]

    # Map gamma and beta optimal parameters to their values
    param_dict = {}
    for gamma_param, gamma_value in zip(gamma, gamma_values):
        param_dict[gamma_param] = gamma_value
    for beta_param, beta_value in zip(beta, beta_values):
        param_dict[beta_param] = beta_value

    bound_circuit = circuit.bind_parameters(param_dict)

    # Add measurement operations to the circuit
    bound_circuit.measure_all()

    # Execute the circuit on the backend
    job = backend.run(bound_circuit, shots=shots)
    result = job.result()

    if not result.get_counts():
        raise Exception("No counts for experiment.")

    counts = result.get_counts()
    
    # Initialize minimum energy and optimal solution
    min_energy = 0
    optimal_solution = 0

    # Find the optimal solution with minimum energy
    for bitstring, count in counts.items():
        x = np.array([int(bit) for bit in bitstring])
        energy = x.T @ Q @ x
        if min_energy >=  energy:
            min_energy = energy
            optimal_solution = x
    return min_energy, optimal_solution


if __name__ == '__main__':
    Q = np.array([[-2, 1, 1, 0],[1,-2,1,0],[1,1,-3,1],[0,0,1,-1]])
    my_qubo = QUBO(Q)

    n_layers = 1
    circuit, gamma, beta = qaoa_composer(my_qubo, n_layers)


