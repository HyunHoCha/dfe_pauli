import math
import time
import random
import numpy as np
from qiskit.quantum_info import random_statevector, random_density_matrix
import pennylane as qml
from pennylane import numpy as np_penny
import helpers
from helpers import *


def random_pure_state(d):
    return random_statevector(d).data.reshape(d, 1)


def random_dm(d):
    return random_density_matrix(d).data


def GHZ_state(n):
    ghz_state = np.zeros((2 ** n, 1), dtype=complex)
    ghz_state[0][0] = 1
    ghz_state[-1][0] = 1
    ghz_state /= np.sqrt(2)
    return ghz_state


def W_state(n):
    w_state = np.zeros((2 ** n, 1), dtype=complex)
    for i in range(n):
        w_state[2 ** i] = 1
    w_state /= np.sqrt(n)
    return w_state


def m_i_func(n, char_val, l, epsilon, delta):
    d = 2 ** n
    return math.ceil(2 * math.log(2 / delta) / (d * char_val ** 2 * l * epsilon ** 2))


def ortho_dm(n, rho):
    d = 2 ** n
    sigma = random_dm(d)
    projector = np.eye(d) - rho
    sigma = projector @ sigma @ projector
    sigma /= np.trace(sigma)
    return sigma


def given_fid(n, rho, f):
    ortho = ortho_dm(n, rho)
    return rho * f + ortho * (1 - f)
