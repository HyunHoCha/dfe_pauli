import math
import time
import random
import numpy as np
from qiskit.quantum_info import random_statevector, random_density_matrix
import pennylane as qml
from pennylane import numpy as np_penny
import helpers
from helpers import *

# python3 utils.py


class Char_Func_phi:  # just for correctness check; not used in experiments
    def __init__(self, n, phi):
        self.n = n
        self.rtd = np.sqrt(2 ** n)
        self.phi_dm = phi @ phi.T.conj()
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.pauli_list = [self.I, self.Z, self.X, self.Y]
    
    def char_func_phi(self, j, k):
        Pauli_global = np.array([[1]])
        for i in range(self.n):
            # jk: 00 - I, 01 - Z, 10 - X, 11 - Y
            Pauli_local = self.pauli_list[2 * j[i] + k[i]]
            Pauli_global = np.kron(Pauli_global, Pauli_local)
        return np.trace(self.phi_dm @ Pauli_global).real / self.rtd


class Char_Func_GHZ:
    def __init__(self, n):
        ghz_state = GHZ_state(n)
        self.n = n
        self.rtd = np.sqrt(2 ** n)
        self.ghz_state_dm = ghz_state @ ghz_state.T.conj()
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.pauli_list = [self.I, self.Z, self.X, self.Y]
    
    def char_func_ghz(self, j, k):
        Pauli_global = np.array([[1]])
        for i in range(self.n):
            # jk: 00 - I, 01 - Z, 10 - X, 11 - Y
            Pauli_local = self.pauli_list[2 * j[i] + k[i]]
            Pauli_global = np.kron(Pauli_global, Pauli_local)
        return np.trace(self.ghz_state_dm @ Pauli_global).real / self.rtd


class Char_Func_W:
    def __init__(self, n):
        w_state = W_state(n)
        self.n = n
        self.rtd = np.sqrt(2 ** n)
        self.w_state_dm = w_state @ w_state.T.conj()
        self.I = np.array([[1, 0], [0, 1]])
        self.X = np.array([[0, 1], [1, 0]])
        self.Y = np.array([[0, -1j], [1j, 0]])
        self.Z = np.array([[1, 0], [0, -1]])
        self.pauli_list = [self.I, self.Z, self.X, self.Y]
    
    def char_func_w(self, j, k):
        Pauli_global = np.array([[1]])
        for i in range(self.n):
            # jk: 00 - I, 01 - Z, 10 - X, 11 - Y
            Pauli_local = self.pauli_list[2 * j[i] + k[i]]
            Pauli_global = np.kron(Pauli_global, Pauli_local)
        return np.trace(self.w_state_dm @ Pauli_global).real / self.rtd


class Flammia_Sampler_jk_GHZ:
    # GHZ state
    # All (I/Z), even number of Z
    # All (X/Y), even number of Y
    # j for X
    # k for Z
    # In terms of (j, k), all (00/01), even weight k
    # In terms of (j, k), all (10/11), even weight k
    # (Sample step 1) j is all-0 or all-1
    # (Sample step 2) Even weight k

    def __init__(self, n):
        self.n = n
        self.zeros = [0 for _ in range(n)]
        self.ones = [1 for _ in range(n)]

    def sample(self, num_samples):
        jk_list = []
        for _ in range(num_samples):
            if random.random() < 1 / 2:
                j_bitstring = self.zeros
            else:
                j_bitstring = self.ones
            
            k_bitstring = [random.randint(0, 1) for _ in range(self.n)]
            hamming_weight = sum(k_bitstring)
            if hamming_weight % 2 == 1:
                k_bitstring[0] ^= 1

            jk_list.append((j_bitstring, k_bitstring))
        return jk_list


class Flammia_Sampler_jk:
    # W state

    def __init__(self, n):
        self.n = n
        d = 2 ** n
        self.branch_dist = np.array([1 / n, 1 - 1 / n], dtype=float)
        self.w_dist = np.array([math.comb(n, w) * (n - 2 * w) ** 2 / (n * d) for w in range(n + 1)], dtype=float)
        self.w_dist /= np.sum(self.w_dist)

    def sample(self, num_samples):
        jk_list = []
        branch_outcome = np.random.multinomial(num_samples, self.branch_dist)
        first_branch_w_outcome = np.random.multinomial(branch_outcome[0], self.w_dist)
        zero_string = [0] * self.n
        for w in range(self.n + 1):
            w_bitstring = [1] * w + [0] * (self.n - w)
            for _ in range(first_branch_w_outcome[w]):
                random.shuffle(w_bitstring)
                jk_list.append((zero_string, w_bitstring[:]))
        two_string = [1] * 2 + [0] * (self.n - 2)
        for _ in range(branch_outcome[1]):
            random.shuffle(two_string)
            j_bitstring = two_string[:]
            k_bitstring = random.choices([0, 1], k=self.n)
            j_support = [idx for idx, bit in enumerate(j_bitstring) if bit == 1]
            k_bitstring[j_support[0]] = k_bitstring[j_support[1]]
            jk_list.append((j_bitstring, k_bitstring))
        return jk_list


def Cha_DFE_GHZ(n, sigma, num_samples):
    dev = qml.device('default.mixed', wires=n, shots=1)

    @qml.qnode(dev)
    def measure_computational():
        qml.QubitDensityMatrix(sigma, wires=range(n))
        return qml.sample()

    @qml.qnode(dev)
    def measure_observable(y_bitstring):
        qml.QubitDensityMatrix(sigma, wires=range(n))
        obs = []
        for i in range(n):
            if y_bitstring[i] == 0:
                obs.append(qml.PauliX(i))
            else:
                obs.append(qml.PauliY(i))
        return [qml.sample(o) for o in obs]

    diag_sum = 0
    off_diag_sum = 0
    for _ in range(num_samples):
        if random.random() < 1 / 3:
            computational_sample_sum = round(sum(measure_computational()))
            if computational_sample_sum == 0 or computational_sample_sum == n:
                diag_sum += 3 / 4
            else:
                diag_sum -= 3 / 4
        else:
            y_bitstring = [random.randint(0, 1) for _ in range(n)]
            hamming_weight = sum(y_bitstring)
            if hamming_weight % 2 == 1:
                y_bitstring[0] ^= 1
            b = measure_observable(y_bitstring)
            b = [round((1 - elem) / 2) for elem in b]
            exponent = round(sum(y_bitstring) / 2) + sum(b)
            off_diag_sum += 3 * (-1) ** exponent / 4
    
    diag_est = diag_sum / num_samples + 1 / 4
    off_diag_est = off_diag_sum / num_samples
    return diag_est + off_diag_est


def Cha_DFE_W(n, sigma, num_samples):
    dev = qml.device('default.mixed', wires=n, shots=1)

    sample_unit = (n ** 2 - n + 1) / (2 * n)

    n_indxs = range(n)

    meas_list = ['X', 'Y']  # for the off-diagonal part

    @qml.qnode(dev)
    def measure_computational():
        qml.QubitDensityMatrix(sigma, wires=range(n))
        return qml.sample()

    @qml.qnode(dev)
    def measure_two(i, j, meas_type):
        qml.QubitDensityMatrix(sigma, wires=range(n))
        if meas_type == 'X':
            return [qml.sample(qml.PauliX(wires=i)), qml.sample(qml.PauliX(wires=j))] + [qml.sample(qml.PauliZ(wires=k)) for k in range(n) if k not in {i, j}]
        if meas_type == 'Y':
            return [qml.sample(qml.PauliY(wires=i)), qml.sample(qml.PauliY(wires=j))] + [qml.sample(qml.PauliZ(wires=k)) for k in range(n) if k not in {i, j}]

    branch_th = 1 / (n ** 2 - n + 1)
    diag_sum = 0
    off_diag_sum = 0
    for _ in range(num_samples):
        if random.random() < branch_th:
            if round(sum(measure_computational())) == 1:
                diag_sum += sample_unit
            else:
                diag_sum -= sample_unit
        else:
            i, j = random.sample(n_indxs, 2)
            # T = {I, H, H S^\dagger} >> {Z, X, Y}
            b = (np.array(measure_two(i, j, random.choice(meas_list))) + 1) / 2
            if round(sum(b[2:])) == n - 2:  # |0> measured at other positions
                if b[0] == b[1]:
                    off_diag_sum += sample_unit
                else:
                    off_diag_sum -= sample_unit
    diag_est = diag_sum / num_samples + 1 / (2 * n)
    off_diag_est = off_diag_sum / num_samples
    return diag_est + off_diag_est
