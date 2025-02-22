import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
import sys
from functools import reduce
import operator



import math

class Gate:
    def __init__(self, matrix, size):
        self.size = size
        self.matrix = matrix
    
    def op(self):
        return self.matrix
    def dagger(self):
        """
        Returns the conjugate transpose (Hermitian adjoint) of this gate's matrix.
        """
        return self.matrix.conj().T
    


    def embed_single_qubit_gate(self, n, qubit):
        """
        Build a (2^n x 2^n) matrix that applies 'gate_2x2' 
        to the specified qubit 'qubit' (0-based), 
        and acts like the 2x2 identity on the other qubits.

        Args:
        gate_2x2 (np.ndarray): shape=(2,2) single-qubit gate
        n (int): total number of qubits (so the state space is dim=2^n)
        qubit (int): which qubit index to act on (0-based).
                    e.g. 0 might be the "left-most" or "right-most" 
                    depending on convention, but be consistent!

        Returns:
        bigU (np.ndarray): shape=(2^n, 2^n) operator
        """
        # 2x2 identity
        I_2 = np.eye(2, dtype=complex)

        # We'll build the operator by repeatedly taking Kronecker products:
        #   out = 1 (scalar)  # or None, and set later
        #   for q in range(n):
        #       if q == qubit: out = np.kron(out, gate_2x2)
        #       else:          out = np.kron(out, I_2)
        #
        # However, the order of qubits (q=0 => left or right) depends on your convention.
        # We'll assume qubit=0 => the *first* position in the product (left-most).
        # If your code interprets qubit=0 as the least significant bit, 
        # you might want to iterate in reverse.

        out = 1
        for q in reversed(range(n)):
            if q == qubit:
                out = np.kron(out, self.matrix)
            else:
                out = np.kron(out, I_2)

        return out
    
    def gen_flip_zero_controls(self, controls, n):
        """
        Given a list of (control_qubit, cval) and total qubit count n,
        build a 2^n x 2^n operator 'bigU' that flips (applies X) to all qubits 
        which have cval=0, ignoring cval=1.

        Args:
        controls: list of (control_qubit, cval) e.g. [(2,0), (5,1), (7,0)]
        n: total number of qubits
        x_gate: your single-qubit X gate class (must have .op() returning 2x2 array)
        
        Returns:
        bigU: a (2^n x 2^n) NumPy array representing the product of all those X embeddings.
                (If no qubit has cval=0, it's just the identity.)
        """
        dim = 2**n
        bigU = np.eye(dim, dtype=complex)
        for (cqubit, val) in controls:
            if val == 0:
                # We want to flip this qubit => embed X gate
                Ux = self.embed_single_qubit_gate(n, cqubit)
                # Multiply it in. If you want left->right order, do bigU = Ux @ bigU
                # or for right->left do bigU = bigU @ Ux.  We'll do left->right:
                bigU = Ux @ bigU
        return bigU




  




class X(Gate):
    def __init__(self):
        matrix = np.array([
            [0, 1],
            [1, 0]
        ], dtype=complex)
        super().__init__(matrix, 2)
class SqZ(Gate):
    def __init__(self):
        matrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=complex)
        super().__init__(matrix, 2)
class SqX(Gate):
    def __init__(self):
        h = H().matrix
        sqZ = SqZ().matrix
        matrix = h @ sqZ @ h
        super().__init__(matrix, 2)



class Y(Gate):
    def __init__(self):
        matrix = np.array([
            [0,  -1j],
            [1j,  0]
        ], dtype=complex)
        super().__init__(matrix, 2)


class Z(Gate):
    def __init__(self):
        matrix = np.array([
            [1,  0],
            [0, -1]
        ], dtype=complex)
        super().__init__(matrix, 2)


class H(Gate):
    def __init__(self):
        # Standard Hadamard is [[1/√2, 1/√2],[1/√2, -1/√2]]
        # If you want your custom version, feel free to adjust.
        matrix = np.array([
            [1/math.sqrt(2),  1/math.sqrt(2)],
            [1/math.sqrt(2), -1/math.sqrt(2)]
        ], dtype=complex)
        super().__init__(matrix, 2)


class I(Gate):
    def __init__(self):
        matrix = np.array([
            [1, 0],
            [0, 1]
        ], dtype=complex)
        super().__init__(matrix, 2)

class T(Gate):
    def __init__(self):
        # diag(1, e^{ i π/4 })
        matrix = np.array([
            [1, 0],
            [0, np.exp(1j*math.pi/4)]
        ], dtype=complex)
        super().__init__(matrix, 2)

class T_DG(Gate):
    def __init__(self):
        # diag(1, e^{ i π/4 })
        matrix = np.array([
            [1, 0],
            [0, np.exp(-1j*math.pi/4)]
        ], dtype=complex)
        super().__init__(matrix, 2)
# class S(Gate):
#     def __init__(self):
#         matrix = np.array([
#             [1, 0],
#             [0, 1j]
#         ], dtype=complex)
#         super().__init__(matrix, 2)

class CNOT(Gate):
    def __init__(self):
        # CNOT is a 4x4 matrix, so size=4
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        super().__init__(matrix, 4)
class V(Gate):
    def __init__(self):
        matrix = np.array([
            [1,  -1j],
            [-1j, 1]
        ], dtype=complex)
        matrix *= ((1j + 1)/2)
        super().__init__(matrix, 2)

class V_plus(Gate):
    def __init__(self):
        matrix = V().op().conj().T
        super().__init__(matrix, 2)


class controlV(Gate):
    def __init__(self):
     
        v00, v01 = V().op()[0, 0], V().op()[0, 1]
        v10, v11 = V().op()[1, 0], V().op()[1, 1]
        
        # Build the 4x4 matrix in the standard basis
        CV = np.array([
            [1,    0,    0,    0   ],
            [0,    1,    0,    0   ],
            [0,    0,    v00,  v01 ],
            [0,    0,    v10,  v11 ]
        ], dtype=complex)
        super().__init__(CV, 4)

class controlV_plus(Gate):
    def __init__(self):
     
        v00, v01 = V_plus().op()[0, 0], V_plus().op()[0, 1]
        v10, v11 = V_plus().op()[1, 0], V_plus().op()[1, 1]
        
        # Build the 4x4 matrix in the standard basis
        CV_plus = np.array([
            [1,    0,    0,    0   ],
            [0,    1,    0,    0   ],
            [0,    0,    v00,  v01 ],
            [0,    0,    v10,  v11 ]
        ], dtype=complex)
        super().__init__(CV_plus, 4)
class Operator():
    def __init__(self, state_vector, qubits, gate):
        self.state_vector = state_vector
        self.gate = gate
        self.qubits = qubits
        self.U = self.embed_two_qubit_gate()


    def get_bit(self, integer, bit_pos):
        """
        Returns the bit (0 or 1) of 'integer' at position 'bit_pos'
        (where bit_pos=0 is the least-significant bit).
        """
        return (integer >> bit_pos) & 1

    def set_bit(self, integer, bit_pos, new_val):
        """
        Returns a new integer, identical to 'integer' except the bit
        at 'bit_pos' is replaced by 'new_val' (0 or 1).
        """
        if new_val not in (0,1):
            raise ValueError("new_val must be 0 or 1")
        mask = 1 << bit_pos
        # Clear the bit
        integer_cleared = integer & ~mask
        # Set it to new_val
        return integer_cleared | (new_val << bit_pos)

    
    def embed_two_qubit_gate(self):
        q0, q1 = self.qubits
        dim = len(self.state_vector)
        # print("dim: ", dim)
        # print("q0: ", q0)
        # print("q1: ", q1)
        if q0 >= np.log2(dim) or q1 >= np.log2(dim):
            raise IndexError("qubit position(s) out of range!")
        big_U = np.zeros((dim, dim), dtype=complex)

        for col_state in range(dim):
            bit0_in = self.get_bit(col_state, q0)
            bit1_in = self.get_bit(col_state, q1)

            # SWAP the order so qubit0 is MSB:
            twoqubit_col = (bit0_in << 1) | bit1_in

            for twoqubit_row in range(4):
                amp = self.gate[twoqubit_row, twoqubit_col]
                if abs(amp) > 1e-15:
                    # decode row bits
                    out_bit0 = (twoqubit_row >> 1) & 1
                    out_bit1 = (twoqubit_row >> 0) & 1

                    # set them back
                    row_state = col_state
                    row_state = self.set_bit(row_state, q0, out_bit0)
                    row_state = self.set_bit(row_state, q1, out_bit1)

                    big_U[row_state, col_state] = amp

        return big_U
    
    


    def act(self):
        U = self.embed_two_qubit_gate()
        return self.state_vector @ U



    def build_hamiltonian(self, H1, H2, J):
    # Put your actual matrices here. 
        a = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        
        b = np.array([
            [0, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 0]
        ], dtype=complex)
        
        c = np.array([
            [1,  0,  0,  0],
            [0, -1,  2,  0],
            [0,  2, -1,  0],
            [0,  0,  0,  1]
        ], dtype=complex)

        H_plus  = H1 + H2
        H_minus = H1 - H2
        # Combine them
        H = (H_plus * a) + (H_minus * b) + (J * c)
        return H

    def cost_function(self, params, U_target):
        # params might be [H1, H2, J, T]
        H1, H2, J, T = params
        H = self.build_hamiltonian(H1, H2, J)
        U_guess = expm(-1j * H * T)
        diff = U_guess - U_target
        # Norm (Frobenius for example)
        cost_val = np.linalg.norm(diff, 'fro')
        return cost_val

# Provide some target 4x4 matrix (unitary) to replicate

    def get_parameters(self, U_target): 
        # For instance, or a CNOT or anything
        print("U_target: ", U_target)

        # Initial guess
        params0 = [1.0, 0.5, 0.2, 1.0]

        res = minimize(self.cost_function, params0, args=(U_target,),
                    method='BFGS', options={'disp':True})

        print("Optimal parameters found:", res.x)
        print("Cost function at optimum:", res.fun)
        return res.x


class controlU(Gate):
    def __init__(self, U):
        self.U = U
        u00, u01 = U[0, 0], U[0, 1]
        u10, u11 = U[1, 0], U[1, 1]
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, u00, u01],
            [0, 0, u10, u11]
        ], dtype=complex)
        super().__init__(matrix, 4)


    def act(self):
        return self.state_vector @ self.U
    
class S(): 
    def __init__(self, qubits, state_vector):
        self.a, self.zero, self.b = qubits
        self.state_vector = state_vector
        self.U = self.buildS()
    def buildS(self):
       
        C1 = Operator(self.state_vector, [self.a, self.zero], CNOT().matrix).U
        C2 = Operator(self.state_vector, [self.zero, self.a], CNOT().matrix).U
        C3 = Operator(self.state_vector, [self.b, self.a], CNOT().matrix).U

        big_U = C3 @ C2 @ C1
        return big_U
class Toffoli():
    def __init__(self, controls, target, state_vector):
        assert len(controls) == 2, "Toffoli must have 2 controls!"
        cU_1_0 = Operator(state_vector, [controls[1],target], controlU(SqX().matrix).matrix).U
        C_2_1 = Operator(state_vector, controls, CNOT().matrix).U
        cU_1_0_dag= Operator(state_vector, [controls[1],target], controlU(SqX().matrix).dagger()).U
        cU_2_0 = Operator(state_vector, [controls[0],target], controlU(SqX().matrix).matrix).U
        self.U = cU_1_0 @ C_2_1 @ cU_1_0_dag @ C_2_1 @ cU_2_0
        self.state_vector = state_vector

    def act(self):
        return self.U @ state_vector

class ToffoliZ():
    def __init__(self, controls, target, state_vector):
        assert len(controls) == 2, "ToffoliZ must have 2 controls!"
        cU_1_0 = Operator(state_vector, [controls[1],target], controlU(SqZ().matrix).matrix).U
        C_2_1 = Operator(state_vector, controls, CNOT().matrix).U
        cU_1_0_dag= Operator(state_vector, [controls[1],target], controlU(SqZ().matrix).dagger()).U
        cU_2_0 = Operator(state_vector, [controls[0],target], controlU(SqZ().matrix).matrix).U
        self.U = cU_1_0 @ C_2_1 @ cU_1_0_dag @ C_2_1 @ cU_2_0
        self.state_vector = state_vector

    def act(self):
        return self.U @ state_vector

class nFoldX():
    """
    A general multi-controlled X gate on 'targets', 
    conditioned on a set of 'controls' each of which 
    can be 0-controlled or 1-controlled.

    E.g., controls = [(2,1), (0,0)] means qubit#2 must be '1' 
    and qubit#0 must be '0' for the flip to occur.
    targets = [1, 3] means we flip qubit#1 and qubit#3.

    We'll build a 2^n x 2^n matrix big_U, 
    then in 'act()' we multiply big_U @ state_vector.
    """
    def __init__(self, n, controls, target, ancillas, state_vector):
        assert len(ancillas) >= n - 2, "must have n-2 ancillas"
        # print_basis_state(state_vector)
        # print("ancillas: ", len(ancillas))
        # print("n-2: ", n-2)
        self.U = np.eye(len(state_vector), dtype=complex)
        if n-2 == 0: 
            self.U = Toffoli(controls, target, state_vector).U
        elif n-2 ==1: 
            tX = Toffoli([ancillas[0], controls[n-1]], target, state_vector).U
            toff = Toffoli([controls[0], controls[1]], ancillas[0], state_vector).U
            self.U = tX @ toff @ tX @ toff
     
        else: 
            tX = Toffoli([ancillas[0], controls[n-1]], target, state_vector).U
            toffoli_gates = [0 for _ in range(n-3)] 
            for i in range(0, n-3):
                toffoli_gates[i] = Toffoli([ancillas[i+1], controls[n-2-i]], ancillas[i], state_vector).U
            middle_toffoli =  Toffoli([controls[0], controls[1]], ancillas[n-3], state_vector).U

    
            product_L2R = reduce(operator.matmul, toffoli_gates)
            product_R2L = reduce(operator.matmul, reversed(toffoli_gates))
        
            # H_ = H().embed_single_qubit_gate((2 * n) - 1, target)
            self.U = tX @ product_L2R @ middle_toffoli @ product_R2L @ tX @ product_L2R @ middle_toffoli @ product_R2L 
            # print(self.U)

class GeneralToffoli():
    def __init__(self, dim, controls, targets, ancillas, state_vector):
        Xs = X().gen_flip_zero_controls(controls, dim)
        nFoldXs = [0 for _ in range(len(targets))]
        ctrls  = [control[0] for control in controls]

        for i in range(len(targets)):
            nFoldXs[i] = nFoldX(len(controls), ctrls, targets[i], ancillas, state_vector).U

        self.U = Xs @ reduce(operator.matmul, nFoldXs) @ Xs
        

        


class Comparator(): 
    def __init__(self, a, b, comparator_zeros, ancillas, c1, c0, state_vector):
        assert len(ancillas) == (len(comparator_zeros))
        self.n = (len(a) + len(b) + len(comparator_zeros) + 2 + len(ancillas))
        self.a = a
        self.b = b
        self.comparator_zeros = comparator_zeros
        self.ancillas = ancillas
        self.c1 = c1
        self.c0 = c0
        self.state_vector = state_vector
        self.U = self.buildComparator()
    def buildComparator(self):

        Us = []
        i = 0
        for a_i,b_i in zip(self.a,self.b):
            controls = [(a_i, 1), (b_i, 0)]
            j = 0
            while j < i: 
                controls.append((self.comparator_zeros[j], 0))
                j+=1
            targets = []
            if i < len(self.comparator_zeros):
                targets = [self.comparator_zeros[i], self.c1]
            else: 
                targets = [self.c1]
            T1 = GeneralToffoli(self.n, controls, targets, self.ancillas, self.state_vector)
            Us.append(round_complex_matrix(T1.U))
            print("T1", controls, targets, self.ancillas)
            i+=1
            controls[0] = (a_i, 0)
            controls[1] = (b_i, 1)
            if i < len(self.comparator_zeros):
                targets = [self.comparator_zeros[i], self.c0]
            else: 
                targets = [self.c0]
            T2 = GeneralToffoli(self.n, controls, targets, self.ancillas, self.state_vector)
            Us.append(round_complex_matrix(T2.U))
            print("T2", controls, targets, self.ancillas)
       
        
            i+=1

        big_U = reduce(operator.matmul, Us)
     
        return big_U
        
    def act(self):
        """Apply the general toffoli operator to the stored state_vector."""
        return self.state_vector @ self.U

class Carry():
    def __init__(self, qubits, state_vector):
       self.a, self.b, self.c, self.d = qubits
       print(qubits)
       self.state_vector = state_vector
       self.U = self.buildCarry()
    def buildCarry(self):
        T1 = Toffoli([self.b, self.c], self.d, self.state_vector).U
        C2 = Operator(self.state_vector, [self.b, self.c], CNOT().matrix).U
        T2 = Toffoli([self.a, self.c], self.d, self.state_vector).U
        U = T2 @ C2 @ T1
        return U

class CarryBack():
    def __init__(self, qubits, state_vector):
       self.a, self.b, self.c, self.d = qubits
       print(qubits)
       self.state_vector = state_vector
       self.U = self.buildCarry()
    def buildCarry(self):
        # T1 = Toffoli([self.d, self.b], self.a, self.state_vector).U
        # C2 = Operator(self.state_vector, [self.c, self.b], CNOT().matrix).U
        # T2 = Toffoli([self.c, self.b], self.a, self.state_vector).U

        T1 = Toffoli([self.b, self.c], self.d, self.state_vector).U
        C2 = Operator(self.state_vector, [self.b, self.c], CNOT().matrix).U
        T2 = Toffoli([self.a, self.c], self.d, self.state_vector).U
        U = T1 @ C2 @ T2
        return U

    def act(self):
        return self.U @ self.state_vector

    
class SUM():
    def __init__(self, qubits, state_vector):
       self.a, self.b, self.sum = qubits
       self.state_vector = state_vector
       self.U = self.buildSUM()
    def buildSUM(self):
        C1 = Operator(self.state_vector, [self.b, self.sum], CNOT().matrix).U

        C2 = Operator(self.state_vector, [self.a, self.sum], CNOT().matrix).U
        
        U = C2 @ C1 
        return U

    
class Adder():
    def __init__(self, a, b, c, state_vector):
       assert len(b) == len(a)+1 and len(a) == len(c)
       self.a, self.b, self.c = a, b, c
       self.state_vector = state_vector
       self.U = self.buildAdder()

    def buildAdder(self):
        # Ripple down
        # ripple = []
        # vec = self.state_vector 
        print_basis_state(vec)
        for i in range(len(self.a)):
            if i == len(self.a)-1:
                ripple.append(Carry([self.c[i], self.a[i], self.b[i], self.b[i+1]], self.state_vector).U)

            else: 
                ripple.append(Carry([self.c[i], self.a[i], self.b[i], self.c[i+1]], self.state_vector).U)
            # vec = ripple[-1] @ vec
            # print_basis_state(vec)
           

        # Ripple up
        for i in reversed(range(len(self.a))):
            if i == len(self.a)-1:
                ripple.append(Operator(self.state_vector, [self.a[i], self.b[i]], CNOT().matrix).U)
                ripple.append(SUM([self.c[i], self.a[i], self.b[i]], self.state_vector).U)
            else: 
                ripple.append(CarryBack([self.c[i], self.a[i], self.b[i], self.c[i+1]], self.state_vector).U)
                ripple.append(SUM([self.c[i], self.a[i], self.b[i]], self.state_vector).U)
            # vec = ripple[-1] @ vec
            # print_basis_state(vec)
        big_U = reduce(operator.matmul, reversed(ripple))
        return big_U

    def act(self):
        return self.U @ self.state_vector 
class AS():
    def __init__(self, a, b, c, state_vector):

        self.a, self.b, self.c = a, b, c
        self.state_vector = state_vector
        self.U = self.buildAS()
    
    def buildAS(self):
        big_U = np.eye(len(self.state_vector))
        for a, b, zero in zip(self.a[:-1], self.b[:-1], self.c):
            big_U = big_U @ S([a,zero,b], self.state_vector).U

        big_U = big_U @ Operator(self.state_vector, [self.b[-1], self.a[-1]], CNOT().matrix).U
        big_U = big_U @ Adder(self.a[:-1], self.b, self.c, self.state_vector).U
        return big_U
    
class ASS():
    def __init__(self, dim, state_vector, x, y, a_x, b_y, c, increment, zeros, c1, c0):
        self.dim = dim
        self.x = x
        self.y = y
        self.a_x = a_x
        self.b_y = b_y
        self.inc = increment
        assert len(increment) == len(a_x)-1
        self.c = c
        self.zeros = zeros
        self.c1 = c1
        self.c0 = c0
        self.state_vector = state_vector
        self.U = self.build()

    def build(self):
        layers = []
        for i in range(len(self.inc)):
            AM_MODULE = Adder(self.inc[i], self.y, self.state_vector).U
            CMP_MODULE = Comparator(self.x, self.y, self.zeros, self.c1, self.c0, self.state_vector).U
            AS_MODULE = AS_00(self.dim, [self.c1, self.c0], self.a_x, self.b_y, self.c, self.inc, self.state_vector).U

class AS_00:
    """
    A subcircuit identical to 'AS', but now controlled by two qubits
    that must both be 0. If either is 1, the subcircuit is effectively 
    an identity on (a,b,c).
    """
    def __init__(self, dim, ctrl_zeros, a, b, c, ancillas, state_vector):

        self.a, self.b, self.c = a, b, c
        self.dim = dim
        self.ancillas = ancillas
        self.ctrl_zeros = ctrl_zeros
        self.state_vector = state_vector
        self.U = self.buildAS()
    
    def buildAS(self):
        big_U = np.eye(len(self.state_vector))
        for a, b, zero in zip(self.a[:-1], self.b[:-1], self.c):
            print("+")
            big_U = big_U @ S_00(self.dim, self.ctrl_zeros, [a,zero,b], self.ancillas, self.state_vector).U
        print("+")
        big_U = big_U @ GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0), (self.ctrl_zeros[1], 0), (self.b[-1], 1)], [self.a[-1]], self.ancillas, self.state_vector).U
        print("last")
        big_U = big_U @ Adder_00(self.dim, self.a[:-1], self.b, self.c, self.ancillas, self.state_vector).U
        return big_U


class Adder_00: 
    def __init__(self, dim, ctrl_zeros, a, b, c, ancillas, state_vector):
       assert len(b) == len(a)+1 and len(a) == len(c)
       self.dim = dim
       self.ancillas = ancillas
       self.ctrl_zeros = ctrl_zeros
       self.a, self.b, self.c = a, b, c
       self.state_vector = state_vector
       self.U = self.buildAdder()

    def buildAdder(self):
        # Ripple down
        ripple = []
        # vec = self.state_vector 
        # print_basis_state(vec)
        for i in range(len(self.a)):
            if i == len(self.a)-1:
                ripple.append(Carry_00(self.dim, self.ctrl_zeros, [self.c[i], self.a[i], self.b[i], self.b[i+1]], self.ancillas, self.state_vector).U)

            else: 
                ripple.append(Carry_00(self.dim, self.ctrl_zeros, [self.c[i], self.a[i], self.b[i], self.c[i+1]], self.ancillas,  self.state_vector).U)
            # vec = ripple[-1] @ vec
            # print_basis_state(vec)
           

        # Ripple up
        for i in reversed(range(len(self.a))):
            if i == len(self.a)-1:
                ripple.append(GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0), (self.ctrl_zeros[1], 0), (self.a[i], 1)], [self.b[i]], self.ancillas, self.state_vector).U)

                ripple.append(SUM_00(self.dim, self.ctrl_zeros, [self.c[i], self.a[i], self.b[i]], self.ancillas, self.state_vector).U)
            else: 
                ripple.append(CarryBack_00(self.dim, self.ctrl_zeros, [self.c[i], self.a[i], self.b[i], self.c[i+1]], self.ancillas, self.state_vector).U)
                ripple.append(SUM_00(self.dim, self.ctrl_zeros, [self.c[i], self.a[i], self.b[i]], self.ancillas, self.state_vector).U)
            # vec = ripple[-1] @ vec
            # print_basis_state(vec)
        big_U = reduce(operator.matmul, reversed(ripple))
        return big_U

class Carry_00():
    def __init__(self, dim, ctrl_zeros, qubits, ancillas, state_vector):
       self.a, self.b, self.c, self.d = qubits
       self.dim = dim
       self.ancillas = ancillas
       self.ctrl_zeros = ctrl_zeros
       print(qubits)
       self.state_vector = state_vector
       self.U = self.buildCarry()
    def buildCarry(self):
        T1 = GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0),(self.ctrl_zeros[1], 0),(self.b, 1), (self.c, 1)], [self.d], self.ancillas, self.state_vector).U
        C2 = GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0),(self.ctrl_zeros[1], 0),(self.b, 1)], [self.c], self.ancillas, self.state_vector).U
        T2 = GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0),(self.ctrl_zeros[1], 0), (self.a, 1), (self.c, 1)], [self.d], self.ancillas, self.state_vector).U
        U = T2 @ C2 @ T1
        return U
class CarryBack_00():
    def __init__(self, ctrl_zeros, qubits, ancillas, state_vector):
       self.a, self.b, self.c, self.d = qubits
       self.ancillas = ancillas
       self.ctrl_zeros = ctrl_zeros
       print(qubits)
       self.state_vector = state_vector
       self.U = self.buildCarry()
    def buildCarry(self):
        T1 = GeneralToffoli([(self.ctrl_zeros[0], 0),(self.ctrl_zeros[1], 0),(self.b, 1), (self.c, 1)], [self.d], self.ancillas, self.state_vector).U
        C2 = GeneralToffoli([(self.ctrl_zeros[0], 0),(self.ctrl_zeros[1], 0),(self.b, 1)], [self.c], self.ancillas, self.state_vector).U
        T2 = GeneralToffoli([(self.ctrl_zeros[0], 0),(self.ctrl_zeros[1], 0), (self.a, 1), (self.c, 1)], [self.d], self.ancillas, self.state_vector).U
        U = T1 @ C2 @ T2
        return U
class SUM_00():
    def __init__(self, dim, ctrl_zeros, qubits, ancillas, state_vector):
       self.a, self.b, self.sum = qubits
       self.dim = dim
       self.ancillas = ancillas
       self.ctrl_zeros = ctrl_zeros
       self.state_vector = state_vector
       self.U = self.buildSUM()
    def buildSUM(self):
        C1 = GeneralToffoli(3, [(self.ctrl_zeros[0], 0), (self.ctrl_zeros[1], 0), (self.b, 1)], [self.sum], self.ancillas, self.state_vector).U

        C2 = GeneralToffoli(3, [(self.ctrl_zeros[0], 0), (self.ctrl_zeros[1], 0), (self.a, 1)], [self.sum], self.ancillas, self.state_vector).U
        
        U = C2 @ C1 
        return U
    
class S_00(): 
    def __init__(self, dim, ctrl_zeros, qubits, ancillas, state_vector):
        self.a, self.zero, self.b = qubits
        self.dim = dim
        self.ancillas = ancillas
        self.ctrl_zeros = ctrl_zeros
        self.state_vector = state_vector
        self.U = self.buildS()
    def buildS(self):
       
        C1 = GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0), (self.ctrl_zeros[1], 0), (self.a, 1)], [self.zero], self.ancillas, self.state_vector).U
        C2 = GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0), (self.ctrl_zeros[1], 0), (self.zero, 1)], [self.a], self.ancillas, self.state_vector).U
        C3 = GeneralToffoli(self.dim, [(self.ctrl_zeros[0], 0), (self.ctrl_zeros[1], 0), (self.b, 1)], [self.a], self.ancillas, self.state_vector).U

        big_U = C3 @ C2 @ C1
        return big_U
# Helpful function for converting binary to state vector
def binary_to_state_vector(bin_str):
    """
    Given a binary string of length n, returns a 1D NumPy array 
    of length 2^n with zeros everywhere except a 1.0 at the 
    decimal index of the binary number.
    
    Example:
        '0101' -> index = 5 -> length=16 state vector with
            state[5] = 1.0, all others = 0.0
    """
    # 1) Determine n from the length of the binary string
    n = len(bin_str)
    # 2) Convert binary -> integer index
    index = int(bin_str, 2)
    # 3) Create a zero vector of length 2^n
    vec = np.zeros(2**n, dtype=complex)
    # 4) Set the position 'index' to 1
    vec[index] = 1.0
    return vec
# Helpful printing function
def print_basis_state(vec):
    """
    If 'vec' is a length 2^n array with exactly one nonzero element, 
    print its binary index in n bits.
    Otherwise, indicate it's not a pure computational basis state.
    """
    # 1) Find indices of nonzero elements
    nonzero_indices = np.flatnonzero(vec)
    
    # 2) Check if there's exactly one
    if len(nonzero_indices) == 1:
        index = nonzero_indices[0]
        # 3) Deduce n from the vector length = 2^n
        n = int(np.log2(len(vec)))
        # 4) Build a binary string of length n, e.g. '0101'
        bin_str = format(index, '0{}b'.format(n))
        print(f"This state is |{bin_str}> in the computational basis.")
    else:
        print("This is not a pure computational basis state (or has multiple nonzero entries).")
# def threshold_basis_state(vec, eps=1e-12):
#     """
#     Returns a copy of 'vec' (which can be complex) where each element 
#     whose absolute value < eps is set to 0+0j.
    
#     This helps remove small floating-point noise in complex arrays.
#     """
#     new_vec = vec.copy()
#     small_mask = (np.abs(new_vec) < eps)
#     new_vec[small_mask] = 0+0j
#     return new_vec

def round_complex_matrix(matrix, decimals=5):
    """
    Round a complex matrix to 'decimals' decimal places in both real and imaginary parts.
    Returns a new complex array (does not modify the input in place).
    """
    # Separate real and imaginary parts
    real_rounded = np.round(matrix.real, decimals=decimals)
    imag_rounded = np.round(matrix.imag, decimals=decimals)
    # Combine back into a complex array
    return real_rounded + 1j * imag_rounded

# -----------------------------------------
# EXAMPLE USAGES:

# SS_MODULE
# initial_state = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=complex)
# norm = np.linalg.norm(initial_state)
# if norm != 0:
#     initial_state /= norm

# SS = SS_Module([1,0,2],initial_state)
# state_vector = SS.act()

# GENERALIZED TOFFOLI GATE
# initial_state = binary_to_state_vector("0101000")

# norm = np.linalg.norm(initial_state)
# if norm != 0:
#     initial_state /= norm

# T = GeneralToffoli(7, [(3,1), (4,0), (5,1), (6,0)],[0,1,2], initial_state)
# state_vector = T.act()
# print(state_vector)
# print_basis_state(state_vector)

# CMP
# initial_state = binary_to_state_vector("1010000000")

# norm = np.linalg.norm(initial_state)
# if norm != 0:
#     initial_state /= norm

# CMP = Comparator([9, 8], [7, 6], [5, 4, 3, 2], 1, 0, initial_state)
# state_vector = CMP.act()
# print(state_vector)
# print_basis_state(state_vector)


#Toffoli
# ancillas = [10, 9, 8, 7]
# controls = [6, 5, 4, 3, 2, 1]
# target = 0
# nFold = nFoldX(6, controls, target, ancillas, state_vector)
# final_state = nFold.U @ state_vector
# print_basis_state(final_state)


# Generalized Toffoli
# state_vector = binary_to_state_vector("10100")
# controls = [(3,0), (2, 1), (1,0)]
# targets = [0]
# ancillas = [4]
# generalToffoli = GeneralToffoli(5, controls, targets, ancillas, state_vector)
# print_basis_state(generalToffoli.U @ state_vector)


#Comparator

# state_vector = binary_to_state_vector("0000101100")
# # T1 = GeneralToffoli(10, [(5, 1), (3, 0)], [7,1], [9,8], state_vector).U
# # print_basis_state(T1 @ state_vector)
# a=[5,4]
# b=[3,2]
# comp_zeros = [7,6]
# ancillas = [9,8]

# comparator = Comparator(a, b, comp_zeros, ancillas, 1, 0, state_vector)
# # np.savetxt("comparator(a_2, b_2).csv", round_complex_matrix(comparator.big_U), delimiter=",")
# # comparator =  np.loadtxt("comparator(a_2, b_2).csv", delimiter=",", dtype=complex)

# # print(comparator.big_U @ state_vector)
# print_basis_state(state_vector @ comparator.U)


# Carry
# state_vector = binary_to_state_vector("0110010")
# carry = Carry([6, 5, 4, 3], state_vector).U
# print_basis_state(state_vector @ carry)


# Sum
# state_vector = binary_to_state_vector("101")
# sum = SUM([2, 1, 0], state_vector).U
# print_basis_state(state_vector @ sum)

#Adder 
# state_vector = binary_to_state_vector("0110100")
# c = [6, 3]
# a = [5, 2]
# b = [4, 1, 0]
# adder = Adder(a, b, c, state_vector).U
# carry1 = Carry([6, 5, 4, 3], state_vector).U
# carry2 = Carry([3, 2, 1, 0], state_vector).U
# cnot = Operator(state_vector, [2, 1], CNOT().matrix).U
# sum1 = SUM([3, 2, 1], state_vector).U
# carryback = CarryBack([6, 5, 4, 3], state_vector).U
# sum2 = SUM([6, 5, 4], state_vector).U
# # vec = sum2 @ carryback @ sum1 @ cnot @ carry2 @ carry1 @ state_vector
# vec = adder @ state_vector
# print_basis_state(vec)

# S module
# state_vector = binary_to_state_vector("001")
# s = S([0,1,2], state_vector).U
# c1 = Operator(state_vector, [0,1], CNOT().matrix).U
# c2 = Operator(state_vector, [1,0], CNOT().matrix).U
# c3 = Operator(state_vector, [2,0], CNOT().matrix).U
# u = c3 @ c2 @c1 
# # print(u)
# print_basis_state(s@ state_vector)

# AS module
# state_vector = binary_to_state_vector("01100100")

# c = [7, 4]
# a = [6, 3, 0]
# b = [5, 2, 1]
# adder = Adder(a[:-1], b, c, state_vector).U
# as_module = AS(a, b, c, state_vector).U
# print_basis_state(as_module @ state_vector) 

# AS_00 module
state_vector = binary_to_state_vector("000001100100")
zeros = [9,8]
ancillas = [11, 10]
c = [7, 4]
a = [6, 3, 0]
b = [5, 2, 1]

as_module = AS_00(12, zeros, a, b, c, ancillas, state_vector).U
print_basis_state(as_module @ state_vector) 




