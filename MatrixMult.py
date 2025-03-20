import numpy as np
import math
from functools import reduce
import operator

# For matrix exponentials, matrix square roots, etc.
from scipy.linalg import expm, sqrtm

# Still using scipy.optimize on CPU for 'minimize':
from scipy.optimize import minimize

# (Optional) Print large NumPy arrays fully (as with cp.set_printoptions(threshold=cp.inf))
np.set_printoptions(threshold=np.inf)

###########################################################
# Helper: Build wavefunction from binary string
###########################################################
def binary_to_state_vector(bin_str):
    """
    Given a binary string of length n, returns a 1D NumPy array
    of length 2^n with zeros except a 1 at the decimal index of bin_str.
    """
    n = len(bin_str)
    index = int(bin_str, 2)
    vec = np.zeros(2**n, dtype=np.complex128)
    vec[index] = 1.0
    return vec
def superposition_to_state_vector(super_str, phases):
    n = len(super_str)
    vec = np.zeros(2**len(super_str[0]), dtype=np.complex128)
    for str, phase in zip(super_str, phases):
        index = int(str, 2)
        vec[index] = phase
    vec = vec * (1/(n**(1/2)))
    return vec


###########################################################
# Helper: Print basis state if single amplitude is nonzero
###########################################################
def print_basis_state(vec):
    """
    Checks if 'vec' (NumPy array) has exactly one nonzero amplitude,
    and if so prints its index in binary.
    """
    nonzeros = np.flatnonzero(np.abs(vec) > 1e-10)  # purely on CPU
    if len(nonzeros) == 1:
        index = nonzeros[0]
        n = int(math.log2(len(vec)))
        bin_str = format(index, '0{}b'.format(n))
        print(f"This state is |{bin_str}> in the computational basis.")
    else:
        print_superposition_state(vec)
def print_superposition_state(vec):
 
    nonzeros = np.flatnonzero(np.abs(vec) > 1e-10)  # move to CPU
    print("superposition |Psi> = ")
    for i, state in enumerate(nonzeros): 
        n = int(math.log2(len(vec)))
        bin_str = format(state, '0{}b'.format(n))
        print_a_b(bin_str)
        print("|"+bin_str+">")
      

class Circuit:
    _active_circuit = None

    def __init__(self, state_vector):
        Circuit._active_circuit = self
        self.state_vec = state_vector
        self.layers = []

    def addLayer(self, module):
        self.layers.append(module)

    def run(self):
        for layer in self.layers:
            layer.act()
        print_basis_state(self.state_vec)
        return self.state_vec

    def decompose(self):
        for layer in self.layers:
            layer.decompose()

    def apply_two_qubit_gate_inplace(self, gate_4x4, ctrl, target):
        """
        In-place application of a controlled 4x4 gate to qubits (ctrl) and (target).
        gate_4x4 is assumed structured as:
           [ I_2   0 ]
           [  0   U_2 ]

        - self.state_vec: 1D NumPy array of length 2^n (dtype=np.complex128)
        - gate_4x4: 4x4 NumPy array
        - ctrl: control-qubit index
        - target: target-qubit index
        """
        n = int(round(np.log2(self.state_vec.size)))

        q0 = n - 1 - ctrl
        q1 = n - 1 - target

        # Reshape state vector into an n-dimensional tensor with shape [2, 2, ..., 2]
        state_tensor = np.reshape(self.state_vec, [2] * n)
        # print("state_tensor[0], state_tensor[1]: ", state_tensor[0], state_tensor[1])
        # Bring q0 (control) and q1 (target) to the last two positions.
        remaining_axes = [i for i in range(n) if i not in (q0, q1)]
        new_order = remaining_axes + [q0, q1]
        # print("new_order: ", new_order)
        state_tensor = np.transpose(state_tensor, new_order)
        # print(state_tensor)
        # Now the tensor shape is (2^(n-2), 2, 2)
        M = 2 ** (n - 2)
        state_tensor = np.reshape(state_tensor, (M, 2, 2))
        # print(state_tensor)
        # For a controlled gate:
        #   The block for control=0 remains identity,
        #   The block for control=1 is gate_4x4[2:4, 2:4].
        target_gate = gate_4x4[2:4, 2:4]  # 2x2 block for control=1

        # Apply to every row where the control qubit is 1:
        #   shape (M, 2) => row [:, 1, :]
        state_tensor[:, 1, :] = np.dot(state_tensor[:, 1, :], target_gate.T)

        # Invert the reshape and transpose to recover original ordering
        state_tensor = np.reshape(state_tensor, [2] * (n - 2) + [2, 2])

        inv_order = [0] * n
        for i, axis in enumerate(new_order):
            inv_order[axis] = i
        state_tensor = np.transpose(state_tensor, inv_order)

        # Flatten back to 1D state vector
        self.state_vec = np.reshape(state_tensor, (-1,))
        return self.state_vec

    def apply_single_qubit_gate_inplace(self, gate_2x2, qubit):
        """
        Apply a single-qubit gate 'gate_2x2' in-place to qubit 'qubit'.
        self.state_vec is a 1D NumPy array of length 2^n.
        """
        n = int(round(np.log2(self.state_vec.size)))

        state_tensor = np.reshape(self.state_vec, [2] * n)
        # Move the target qubit axis to the last position
        axis = n - 1 - qubit
        axes = list(range(n))
        axes.remove(axis)
        axes.append(axis)
        state_tensor = np.transpose(state_tensor, axes)

        # Merge all other axes => shape (2**(n-1), 2)
        state_tensor = np.reshape(state_tensor, (-1, 2))

        # Apply the gate in batch
        state_tensor = np.dot(state_tensor, gate_2x2.T)

        # Reshape back and invert the transpose
        state_tensor = np.reshape(state_tensor, [2] * n)
        inv_axes = np.argsort(np.array(axes))
        state_tensor = np.transpose(state_tensor, inv_axes.tolist())

        self.state_vec = np.reshape(state_tensor, (-1,))
        return self.state_vec

    ###########################################################
    # Class for circuit tuning and optimization
    ###########################################################
    class Tuner:
        # Functions to build two qubit gates from magnetic fields
        def build_hamiltonian(self, H1, H2, J):
            a = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, -1]], dtype=np.complex128)
            b = np.array([[0, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, -1, 0],
                          [0, 0, 0, 0]], dtype=np.complex128)
            c = np.array([[1, 0, 0, 0],
                          [0, -1, 2, 0],
                          [0, 2, -1, 0],
                          [0, 0, 0, 1]], dtype=np.complex128)

            H_plus = H1 + H2
            H_minus = H1 - H2
            H = (H_plus * a) + (H_minus * b) + (J * c)
            return H

        def cost_function(self, params, U_target):
            H1, H2, J, T = params
            H = self.build_hamiltonian(H1, H2, J)
            U_guess = expm(-1j * H * T)

            diff = U_guess - U_target
            cost_val = np.linalg.norm(diff, ord='fro')
            return float(cost_val)

        def get_parameters(self, U_target):
            print("U_target:", U_target)
            params0 = [1.0, 0.5, 0.2, 1.0]

            def wrap_cost(p):
                return self.cost_function(p, U_target)

            res = minimize(wrap_cost, params0, method='BFGS', options={'disp': True})
            print("Optimal:", res.x)
            print("Cost:", res.fun)
            return res.x

    ###########################################################
    # Helper: Round a complex NumPy matrix
    ###########################################################
    def round_complex_matrix(self, matrix, decimals=5):
        """
        Round a complex matrix (NumPy array) to 'decimals' decimal places
        in both real and imaginary parts.
        Returns a new complex NumPy array (does not modify the input in place).
        """
        real_rounded = np.around(matrix.real, decimals=decimals)
        imag_rounded = np.around(matrix.imag, decimals=decimals)
        return real_rounded + 1j * imag_rounded

    ###########################################################
    # Helper: Build wavefunction from binary string
    ###########################################################
    def binary_to_state_vector(self, bin_str):
        """
        Same as global helper, method version for the class.
        """
        n = len(bin_str)
        index = int(bin_str, 2)
        vec = np.zeros(2**n, dtype=np.complex128)
        vec[index] = 1.0
        return vec

    ###########################################################
    # Helper: Print basis state if single amplitude is nonzero
    ###########################################################
    def print_basis_state(self, vec):
        nonzeros = np.flatnonzero(np.abs(vec) > 1e-15)
        if len(nonzeros) == 1:
            index = nonzeros[0]
            n = int(math.log2(len(vec)))
            bin_str = format(index, '0{}b'.format(n))
            print(f"This state is |{bin_str}> in the computational basis.")
        else:
            print("This is not a pure computational basis state (or multiple nonzero entries).")

    ###########################################################
    # Gate Class and Subclasses
    ###########################################################
    class OneQubitGate:
        def __init__(self, matrix, q):
            self.circuit = Circuit._active_circuit
            self.matrix = np.asarray(matrix, dtype=np.complex128)
            self.q = q

        def act(self):
            self.circuit.apply_single_qubit_gate_inplace(self.matrix, self.q)

        def sqrt(self, matrix):
            # In NumPy, we just do sqrtm directly:
            sqrtU_cpu = sqrtm(matrix)  # from scipy.linalg import sqrtm
            sqrtU = np.asarray(sqrtU_cpu)
            return sqrtU

        def dagger(self, matrix):
            return matrix.conj().T

    class TwoQubitGate:
        def __init__(self, matrix, ctrl, target):
            self.circuit = Circuit._active_circuit
            self.matrix = np.asarray(matrix, dtype=np.complex128)
            self.U = self.matrix[2:, 2:]
            self.dagger = self.matrix.conj().T
            self.ctrl = ctrl
            self.target = target

        def sqrt(self):
            sqrtU_cpu = sqrtm(self.matrix)
            sqrtU = np.asarray(sqrtU_cpu)
            self.matrix = sqrtU
            return self

        def act(self):
            self.circuit.apply_two_qubit_gate_inplace(self.matrix, self.ctrl, self.target)

    class X(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[0, 1], [1, 0]], dtype=np.complex128)
            self.U = mat
            super().__init__(mat, q)

    class SqZ(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
            super().__init__(mat, q)

    class SqX(OneQubitGate):
        def __init__(self, q):
            h = np.array([[1/math.sqrt(2), 1/math.sqrt(2)],
                          [1/math.sqrt(2), -1/math.sqrt(2)]], dtype=np.complex128)
            sqZ = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
            mat = h @ sqZ @ h
            super().__init__(mat, q)

    class Y(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
            super().__init__(mat, q)

    class Z(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[1, 0], [0, -1]], dtype=np.complex128)
            super().__init__(mat, q)

    class H(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[1/math.sqrt(2),  1/math.sqrt(2)],
                            [1/math.sqrt(2), -1/math.sqrt(2)]],
                           dtype=np.complex128)
            super().__init__(mat, q)

    class I(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[1, 0], [0, 1]], dtype=np.complex128)
            super().__init__(mat, q)

    class T(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[1, 0],
                            [0, np.exp(1j*math.pi/4)]], dtype=np.complex128)
            super().__init__(mat, q)

    class CNOT(TwoQubitGate):
        def __init__(self, q0, q1):
            mat = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]], dtype=np.complex128)
            super().__init__(mat, q0, q1)

    class V(OneQubitGate):
        def __init__(self, q):
            mat = np.array([[1,   -1j],
                            [-1j,  1]], dtype=np.complex128)
            mat *= ((1j + 1) / 2)
            super().__init__(mat, q)

    #############################################################
    # controlU
    #############################################################
    class controlU(TwoQubitGate):
        def __init__(self, U, q0, q1):
            U_np = np.asarray(U)
            u00, u01 = U_np[0, 0], U_np[0, 1]
            u10, u11 = U_np[1, 0], U_np[1, 1]
            mat = np.array([
                [1,  0,    0,    0],
                [0,  1,    0,    0],
                [0,  0,  u00,  u01],
                [0,  0,  u10,  u11],
            ], dtype=np.complex128)
            super().__init__(mat, q0, q1)

    #############################################################
    # Example S class
    #############################################################
    class Module:
        def __init__(self):
            self.circuit = Circuit._active_circuit
            self.layers = []

        def makeControlled(self, ctrls, ancillas):
            if len(ancillas) < len(ctrls) - 1:
                raise ValueError("Not enough ancillas to make this a controlled module!")
            print(self)

            for i, layer in enumerate(self.layers):
                if isinstance(layer, self.circuit.OneQubitGate):
                    U = layer.matrix
                    controls = ctrls.copy()
                    self.layers[i] = layer.circuit.GeneralToffoliU(controls, [layer.q], ancillas, U)

                elif isinstance(layer, self.circuit.TwoQubitGate):
                    U = layer.U
                    controls = ctrls.copy()
                    controls.append((layer.ctrl, 1))
                    self.layers[i] = layer.circuit.GeneralToffoliU(controls, [layer.target], ancillas, U)

                elif isinstance(layer, self.circuit.Toffoli):
                    controls = ctrls.copy()
                    controls.extend([(layer.c0, 1), (layer.c1, 1)])
                    self.layers[i] = layer.circuit.GeneralToffoli(controls, [layer.target], ancillas)
                else:
                    layer.makeControlled(ctrls, ancillas)

            return self

        def act(self):
            for layer in self.layers:
                layer.act()
            return self.circuit.state_vec

        def decompose(self):
            for layer in self.layers:
                if isinstance(layer, self.circuit.OneQubitGate):
                    print(layer, layer.q)
                elif isinstance(layer, self.circuit.TwoQubitGate):
                    print(layer, layer.ctrl, layer.target)
                else:
                    layer.decompose()

        def sqrt(self, matrix):
            sqrtU_cpu = sqrtm(matrix)
            sqrtU = np.asarray(sqrtU_cpu)
            return sqrtU

        def dagger(self, matrix):
            return matrix.conj().T

    class S(Module):
        def __init__(self, p, zero, q):
            super().__init__()
            self.p = p
            self.zero = zero
            self.q = q
            self.layers.append(self.circuit.CNOT(self.p, self.zero))
            self.layers.append(self.circuit.CNOT(self.zero, self.p))
            self.layers.append(self.circuit.CNOT(self.q, self.p))

    #############################################################
    # Example Toffoli
    #############################################################
    class Toffoli(Module):
        def __init__(self, c0, c1, target):
            super().__init__()
            self.c0 = c0
            self.c1 = c1
            self.target = target

            sqx_matrix = self.circuit.SqX(self.target).matrix
            self.layers.append(self.circuit.controlU(sqx_matrix, self.c0, self.target))
            self.layers.append(self.circuit.CNOT(self.c0, self.c1))
            self.layers.append(self.circuit.controlU(self.circuit.SqX(self.target).dagger(sqx_matrix),
                                                     self.c1, self.target))
            self.layers.append(self.circuit.CNOT(self.c0, self.c1))
            self.layers.append(self.circuit.controlU(sqx_matrix, self.c1, self.target))

    class ToffoliU(Module):
        def __init__(self, c0, c1, target, U):
            super().__init__()
            self.U = U
            self.c0 = c0
            self.c1 = c1
            self.target = target

            sqrtU = self.circuit.SqX(self.target).sqrt(U)
            dagger_sqrtU = self.circuit.SqX(self.target).dagger(sqrtU)

            self.layers.append(self.circuit.controlU(sqrtU, self.c0, self.target))
            self.layers.append(self.circuit.CNOT(self.c0, self.c1))
            self.layers.append(self.circuit.controlU(dagger_sqrtU, self.c1, self.target))
            self.layers.append(self.circuit.CNOT(self.c0, self.c1))
            self.layers.append(self.circuit.controlU(sqrtU, self.c1, self.target))

    #############################################################
    # nFoldX
    #############################################################
    class nFoldX(Module):
        def __init__(self, controls, target, ancillas):
            super().__init__()
            n = len(controls)
            if len(ancillas) < n - 2:
                raise ValueError("Must have at least n-2 ancillas")

            if n - 2 == 0:
                # standard Toffoli
                self.layers.append(self.circuit.Toffoli(controls[0], controls[1], target))
            elif n - 2 == 1:
                tX = self.circuit.Toffoli(ancillas[0], controls[n - 1], target)
                toff = self.circuit.Toffoli(controls[0], controls[1], ancillas[0])
                self.layers.append(tX)
                self.layers.append(toff)
                self.layers.append(tX)
                self.layers.append(toff)
            else:
                tX = self.circuit.Toffoli(ancillas[0], controls[n - 1], target)
                toffoli_gates = []
                for i in range(n - 3):
                    tgate = self.circuit.Toffoli(ancillas[i + 1], controls[n - 2 - i], ancillas[i])
                    toffoli_gates.append(tgate)
                middle_toffoli = self.circuit.Toffoli(controls[0], controls[1], ancillas[n - 3])

                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))
                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))

    class nFoldU(Module):
        def __init__(self, controls, target, ancillas, U):
            super().__init__()
            self.U = U
            n = len(controls)
            if len(ancillas) < n - 2:
                raise ValueError("Must have at least n-2 ancillas")

            if n - 2 == 0:
                self.layers.append(self.circuit.ToffoliU(controls[0], controls[1], target, U))
            elif n - 2 == 1:
                tX = self.circuit.ToffoliU(ancillas[0], controls[n - 1], target, U)
                toff = self.circuit.ToffoliU(controls[0], controls[1], ancillas[0], U)
                self.layers.append(tX)
                self.layers.append(toff)
                self.layers.append(tX)
                self.layers.append(toff)
            else:
                tX = self.circuit.ToffoliU(ancillas[0], controls[n - 1], target, U)
                toffoli_gates = []
                for i in range(n - 3):
                    tgate = self.circuit.ToffoliU(ancillas[i + 1], controls[n - 2 - i], ancillas[i], U)
                    toffoli_gates.append(tgate)
                middle_toffoli = self.circuit.ToffoliU(controls[0], controls[1], ancillas[n - 3], U)

                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))
                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))

    #############################################################
    # GeneralToffoli
    #############################################################
    class GeneralToffoli(Module):
        def __init__(self, controls, targets, ancillas):
            super().__init__()
            self.controls = controls
            self.targets = targets
            self.ancillas = ancillas

            ctrls = [control[0] for control in controls]
            # Flip those controls that are 0
            for (control, val) in self.controls:
                if val == 0:
                    self.layers.append(self.circuit.X(control))
            # Apply multi-controlled X for each target
            for target in targets:
                self.layers.append(self.circuit.nFoldX(ctrls, target, ancillas))
            # Flip back
            for (control, val) in self.controls:
                if val == 0:
                    self.layers.append(self.circuit.X(control))

    class GeneralToffoliU(Module):
        def __init__(self, controls, targets, ancillas, U):
            super().__init__()
            self.U = U
            self.controls = controls
            self.targets = targets
            ctrls = [control[0] for control in controls]

            # Flip those controls that are 0
            for (control, val) in self.controls:
                if val == 0:
                    self.layers.append(self.circuit.X(control))
            # Apply multi-controlled U for each target
            for target in targets:
                self.layers.append(self.circuit.nFoldU(ctrls, target, ancillas, U))
            # Flip back
            for (control, val) in self.controls:
                if val == 0:
                    self.layers.append(self.circuit.X(control))

    #############################################################
    # Comparator
    #############################################################
    class Comparator(Module):
        def __init__(self, a, b, comparator_zeros, ancillas, c1, c0):
            super().__init__()
            if len(ancillas) != len(comparator_zeros):
                raise ValueError("ancillas, comparator_zeros mismatch")

            self.n = (len(a) + len(b) + len(comparator_zeros) + 2 + len(ancillas))
            self.a = a
            self.b = b
            self.comparator_zeros = comparator_zeros
            self.ancillas = ancillas
            self.c1 = c1
            self.c0 = c0

            i = 0
            for a_i, b_i in zip(self.a, self.b):
                controls = [(a_i, 1), (b_i, 0)]
                j = 0
                while j < i:
                    controls.append((self.comparator_zeros[j], 0))
                    j += 1
                if i < len(self.comparator_zeros):
                    targets = [self.comparator_zeros[i], self.c1]
                else:
                    targets = [self.c1]
                T1 = self.circuit.GeneralToffoli(controls.copy(), targets.copy(), self.ancillas)
                self.layers.append(T1)
                print("T1", controls, targets, self.ancillas)
                i += 1

                # flip the first two controls
                controls[0] = (a_i, 0)
                controls[1] = (b_i, 1)
                if i < len(self.comparator_zeros):
                    targets = [self.comparator_zeros[i], self.c0]
                else:
                    targets = [self.c0]
                T2 = self.circuit.GeneralToffoli(controls.copy(), targets.copy(), self.ancillas)
                self.layers.append(T2)
                print("T2", controls, targets, self.ancillas)
                i += 1

            for layer in self.layers:
                print(layer.controls, layer.targets)

    class Carry(Module):
        def __init__(self, a, b, c, d):
            super().__init__()
            self.a, self.b, self.c, self.d = a, b, c, d
            self.layers.append(self.circuit.Toffoli(self.b, self.c, self.d))
            self.layers.append(self.circuit.CNOT(self.b, self.c))
            self.layers.append(self.circuit.Toffoli(self.a, self.c, self.d))

    class CarryBack(Module):
        def __init__(self, a, b, c, d):
            super().__init__()
            self.a, self.b, self.c, self.d = a, b, c, d
            self.layers.append(self.circuit.Toffoli(self.a, self.c, self.d))
            self.layers.append(self.circuit.CNOT(self.b, self.c))
            self.layers.append(self.circuit.Toffoli(self.b, self.c, self.d))

    class SUM(Module):
        def __init__(self, a, b, s):
            super().__init__()
            self.a, self.b, self.s = a, b, s
            self.layers.append(self.circuit.CNOT(self.b, self.s))
            self.layers.append(self.circuit.CNOT(self.a, self.s))

    class Adder(Module):
        def __init__(self, a, b, c):
            super().__init__()
            assert len(b) == len(a) + 1 and len(a) == len(c)
            self.a, self.b, self.c = a, b, c

            # Ripple down with Carry gates
            for i in range(len(self.a)):
                if i == len(self.a) - 1:
                    carryU = self.circuit.Carry(self.c[i], self.a[i], self.b[i], self.b[i + 1])
                    self.layers.append(carryU)
                else:
                    carryU = self.circuit.Carry(self.c[i], self.a[i], self.b[i], self.c[i + 1])
                    self.layers.append(carryU)

            # Ripple up with CarryBack and SUM gates
            for i in reversed(range(len(self.a))):
                if i == len(self.a) - 1:
                    cnotU = self.circuit.CNOT(self.a[i], self.b[i])
                    self.layers.append(cnotU)
                    sumU = self.circuit.SUM(self.c[i], self.a[i], self.b[i])
                    self.layers.append(sumU)
                else:
                    cbackU = self.circuit.CarryBack(self.c[i], self.a[i], self.b[i], self.c[i + 1])
                    self.layers.append(cbackU)
                    sumU = self.circuit.SUM(self.c[i], self.a[i], self.b[i])
                    self.layers.append(sumU)

    class AS(Module):
        """
        The 'AS' module with no explicit control:
          - Adder(...)
          - CNOT(...)
          - Then S(...) for each triple
        """
        def __init__(self, a, b, c):
            super().__init__()
            self.a = a
            self.b = b
            self.c = c

            self.layers.append(self.circuit.Adder(self.a[:-1], self.b, self.c))
            self.layers.append(self.circuit.CNOT(self.b[-1], self.a[-1]))

            for a_i, b_i, zero_i in zip(self.a[:-1], self.b[:-1], self.c):
                self.layers.append(self.circuit.S(a_i, zero_i, b_i))

    class ASS:
        """
        A bigger module that calls Adder(...), Comparator(...), etc.
        Shown here only as a placeholder for the logic you want.
        """
        def __init__(self, dim, state_vector, x, y, a_x, b_y, c, increment, zeros, c1, c0):
            self.dim = dim
            self.x = x
            self.y = y
            self.a_x = a_x
            self.b_y = b_y
            self.inc = increment
            assert len(increment) == len(a_x) - 1
            self.c = c
            self.zeros = zeros
            self.c1 = c1
            self.c0 = c0
            self.state_vector = state_vector
            self.U = self.build()

        def build(self):
            layers = []
            for i in range(len(self.inc)):
                # Example usage (incomplete):
                AM_MODULE = Adder(self.inc[i], self.y, self.state_vector).U  # or the Module approach
                layers.append(AM_MODULE)
                # ... likewise for Comparator, etc.
                pass

            if layers:
                return reduce(operator.matmul, layers)
            else:
                return np.eye(len(self.state_vector), dtype=np.complex128)


# -------------------- EXAMPLE USAGE -------------------- #
def basis_state(vec):
    """
    Checks if 'vec' (CuPy array) has exactly one nonzero amplitude, 
    and if so prints its index in binary.
    """
    # find nonzeros on GPU
    nonzeros = np.flatnonzero(np.abs(vec) > 1e-10)  # move to CPU
    if len(nonzeros) == 1:
        index = nonzeros[0]
        n = int(math.log2(len(vec)))
        bin_str = format(index, '0{}b'.format(n))
        return bin_str

def print_a_b(bin):
    binary = list(reversed(bin))
    # print(binary)
    print("a: ", list(reversed(binary[4:7])))
    print("b: ", list(reversed(binary[0:3])))
    print("x: ", binary[7])
    print("y: ", binary[3])

#"00----x a_0 a_1 a_2 y b_0 b_1 b_2"
    
#AS test
# binary_str = "0000000000110011"
# state_vec = binary_to_state_vector(binary_str)

# circ = Circuit(state_vec)
# circ.addLayer(circ.AS([0,1,2],[4,5,6],[8, 9]))
# # .makeControlled([(3,0), (7,0)], [10,11])
# state_vec = circ.run()

# binary = basis_state(state_vec)
# print_a_b(binary)

#Compare x==y test:
# binary_str = "0000000000110011"
# state_vec = binary_to_state_vector(binary_str)

# circ = Circuit(state_vec)
# circ.addLayer(circ.Comparator([3],[7],[10,11],[12,13],14,15))
# state_vec = circ.run()

# binary = basis_state(state_vec)
# print_a_b(binary)


#AS only if x==y 
# binary_str = "0000000000110011"
# state_vec = binary_to_state_vector(binary_str)

# circ = Circuit(state_vec)
# circ.addLayer(circ.Comparator([3],[7],[10,11],[12,13],14,15))
# circ.addLayer(circ.AS([0,1,2],[4,5,6],[8, 9]).makeControlled([(15,0),(14,0)], [12,13]))

# state_vec = circ.run()

# binary = basis_state(state_vec)
# print_a_b(binary)



superposition = ["00100010", "00101011", "10110010", "10111011"]
for i in range(0, len(superposition)): 
    superposition[i] = "00000000" + superposition[i]
    

state_vector = superposition_to_state_vector(superposition, [1, 1, 1, 1])
circuit = Circuit(state_vector)
circuit.addLayer(circuit.X(3))
circuit.addLayer(circuit.Comparator([3],[7],[10,11],[12,13],14,15))
circuit.addLayer(circuit.AS([0,1,2],[4,5,6],[8, 9]).makeControlled([(15,0),(14,0)], [12,13]))

circuit.run()

# bin = "10"
# state_vec = binary_to_state_vector(bin)
# circ = Circuit(state_vec)
# circ.addLayer(circ.CNOT(1,0))
# circ.run()

