import cupy as cp
import scipy.linalg
import math
from functools import reduce
import operator
import time

#############################################
# Rewritten Code Using CuPy
#############################################

import cupy as cp
import cupyx.scipy.linalg as cpx_linalg  # for expm
import math
from functools import reduce
import operator

# Still using scipy.optimize on CPU for 'minimize':
from scipy.optimize import minimize

cp.set_printoptions(threshold=cp.inf) 

cp.cuda.Device(0).use()


 ###########################################################
    # Helper: Build wavefunction from binary string
    ###########################################################
def binary_to_state_vector(bin_str):
    """
    Given a binary string of length n, returns a 1D CuPy array 
    of length 2^n with zeros except a 1 at the decimal index of bin_str.
    """
    n = len(bin_str)
    index = int(bin_str, 2)
    vec = cp.zeros(2**n, dtype=cp.complex128)
    vec[index] = 1.0
    return vec

###########################################################
# Helper: Print basis state if single amplitude is nonzero
###########################################################
def print_basis_state(vec):
    """
    Checks if 'vec' (CuPy array) has exactly one nonzero amplitude, 
    and if so prints its index in binary.
    """
    # find nonzeros on GPU
    nonzeros = cp.flatnonzero(cp.abs(vec) > 1e-10).get()  # move to CPU
    if len(nonzeros) == 1:
        index = nonzeros[0]
        n = int(math.log2(len(vec)))
        bin_str = format(index, '0{}b'.format(n))
        print(f"This state is |{bin_str}> in the computational basis.")
    else:
        print("This is not a pure computational basis state (or multiple nonzero entries).")
        # print(vec)

class Circuit(): 
    _active_circuit = None
    def __init__(self, state_vector):
        Circuit._active_circuit = self
        self.state_vec = state_vector
        self.layers=[]

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
        In-place application of a controlled 4x4 gate to qubits q0 (control) and q1 (target)
        using vectorized operations. Assumes that gate_4x4 is structured as:
        
            [ I_2    0 ]
            [  0   U_2 ]
        
        where I_2 is the 2x2 identity (applied when control is 0) and U_2 is a 2x2 unitary
        acting on the target qubit when the control is 1.
        
        - self.state_vec: 1D CuPy array of length 2^n (dtype=cp.complex128)
        - gate_4x4: 4x4 CuPy array
        - q0: control qubit index
        - q1: target qubit index
        """
        import cupy as cp
        n = int(cp.log2(self.state_vec.size))
        q0 = n-1-ctrl
        q1 = n-1-target
        
        # Reshape state vector into an n-dimensional tensor with shape [2, 2, ..., 2]
        state_tensor = cp.reshape(self.state_vec, [2] * n)
        
        # Bring q0 (control) and q1 (target) to the last two positions.
        # 'remaining_axes' holds the indices of all other qubits.
        remaining_axes = [i for i in range(n) if i not in (q0, q1)]
        new_order = remaining_axes + [q0, q1]
        state_tensor = cp.transpose(state_tensor, new_order)
        
        # Now the tensor shape is (2^(n-2), 2, 2)
        M = 2 ** (n - 2)
        state_tensor = cp.reshape(state_tensor, (M, 2, 2))
        
        # For a controlled gate:
        # - The block corresponding to control=0 (first row of the 2nd axis) remains unchanged.
        # - For control=1 (second row), apply the target gate U_2.
        target_gate = gate_4x4[2:4, 2:4]  # extract the 2x2 block for control=1
        # Apply the target gate to every row where the control qubit is 1.
        # Each such row (shape: (2,)) is updated in batch:
        state_tensor[:, 1, :] = cp.dot(state_tensor[:, 1, :], target_gate.T)
        
        # Invert the reshape and transpose to recover the original ordering.
        state_tensor = cp.reshape(state_tensor, [2] * (n - 2) + [2, 2])
        # Compute the inverse permutation of new_order.
        inv_order = [0] * n
        for i, axis in enumerate(new_order):
            inv_order[axis] = i
        state_tensor = cp.transpose(state_tensor, inv_order)
        
        # Flatten back to 1D state vector.
        self.state_vec = cp.reshape(state_tensor, (-1,))
        return self.state_vec

    def apply_single_qubit_gate_inplace(self, gate_2x2, qubit):
        n = int(cp.log2(self.state_vec.size))
        # Reshape state vector into an n-dimensional tensor
        state_tensor = cp.reshape(self.state_vec, [2]*n)
        # Move the target qubit axis to the last position
        axes = list(range(n))
        axes.remove(n-1-qubit)
        axes.append(n-1-qubit)
        state_tensor = cp.transpose(state_tensor, axes)
        
        # Merge all other axes: now shape is (2**(n-1), 2)
        new_shape = (-1, 2)
        state_tensor = cp.reshape(state_tensor, new_shape)
        
        # Apply the gate in batch
        state_tensor = cp.dot(state_tensor, gate_2x2.T)
        
        # Reshape back and invert the transpose
        state_tensor = cp.reshape(state_tensor, [2]*n)
        # Compute the inverse permutation of axes
        inv_axes = cp.argsort(cp.array(axes))
        state_tensor = cp.transpose(state_tensor, inv_axes.tolist())
        
        self.state_vec = cp.reshape(state_tensor, (-1,))
        return self.state_vec

    
    ###########################################################
    # Class for circuit tuning and optimization
    ###########################################################
    class Tuner:
        # Functions to build two qubit gates from magnetic fields
        def build_hamiltonian(self, H1,H2,J):
            a= cp.array([[1,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,-1]], dtype=cp.complex128)
            b= cp.array([[0,0,0,0],
                        [0,1,0,0],
                        [0,0,-1,0],
                        [0,0,0,0]], dtype=cp.complex128)
            c= cp.array([[1,0,0,0],
                        [0,-1,2,0],
                        [0,2,-1,0],
                        [0,0,0,1]], dtype=cp.complex128)
            H_plus= H1+H2
            H_minus=H1-H2
            H= (H_plus*a)+(H_minus*b)+(J*c)
            return H

        def cost_function(self, params, U_target):
            # Move to GPU
            H1,H2,J,T= params
            H= self.build_hamiltonian(H1,H2,J)
            U_guess= cpx_linalg.expm(-1j*H*T)
            diff= U_guess- U_target
            cost_val= cp.linalg.norm(diff, ord='fro')
            return float(cost_val.get())

        def get_parameters(self, U_target):
            print("U_target:", U_target)
            params0= [1.0, 0.5, 0.2, 1.0]
            def wrap_cost(p):
                return self.cost_function(p, U_target)
            res= minimize(wrap_cost, params0,method='BFGS', options={'disp':True})
            print("Optimal:",res.x)
            print("Cost:",res.fun)
            return res.x
    ###########################################################
    # Helper: Round a complex CuPy matrix
    ###########################################################
    def round_complex_matrix(self, matrix, decimals=5):
        """
        Round a complex matrix (CuPy array) to 'decimals' decimal places 
        in both real and imaginary parts.
        Returns a new complex CuPy array (does not modify the input in place).
        """
        real_rounded = cp.around(matrix.real, decimals=decimals)
        imag_rounded = cp.around(matrix.imag, decimals=decimals)
        return real_rounded + 1j * imag_rounded

    ###########################################################
    # Helper: Build wavefunction from binary string
    ###########################################################
    def binary_to_state_vector(self, bin_str):
        """
        Given a binary string of length n, returns a 1D CuPy array 
        of length 2^n with zeros except a 1 at the decimal index of bin_str.
        """
        n = len(bin_str)
        index = int(bin_str, 2)
        vec = cp.zeros(2**n, dtype=cp.complex128)
        vec[index] = 1.0
        return vec

    ###########################################################
    # Helper: Print basis state if single amplitude is nonzero
    ###########################################################
    def print_basis_state(self, vec):
        """
        Checks if 'vec' (CuPy array) has exactly one nonzero amplitude, 
        and if so prints its index in binary.
        """
        # find nonzeros on GPU
        nonzeros = cp.flatnonzero(cp.abs(vec) > 1e-15).get()  # move to CPU
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
          
            self.matrix = cp.asarray(matrix, dtype=cp.complex128)
            self.q = q

        def act(self):
            self.circuit.apply_single_qubit_gate_inplace(self.matrix, self.q)

        def sqrt(self, matrix):
            U_cpu = cp.asnumpy(matrix)
            sqrtU_cpu = scipy.linalg.sqrtm(U_cpu)
            sqrtU = cp.asarray(sqrtU_cpu)
            return sqrtU
        def dagger(self, matrix):
            return matrix.conj().T
    class TwoQubitGate:
        def __init__(self, matrix, ctrl, target):
            self.circuit = Circuit._active_circuit
          
            self.matrix = cp.asarray(matrix, dtype=cp.complex128)
            self.U = self.matrix[2:, 2:]
            self.dagger = self.matrix.conj().T
            self.ctrl = ctrl
            self.target = target
        
        def sqrt(self):
            U_cpu = cp.asnumpy(self.matrix)
            sqrtU_cpu = scipy.linalg.sqrtm(U_cpu)
            sqrtU = cp.asarray(sqrtU_cpu)
            self.matrix = sqrtU
            return self

        def act(self):

                self.circuit.apply_two_qubit_gate_inplace(self.matrix, self.ctrl, self.target)

    


    class X(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[0,1],[1,0]], dtype=cp.complex128)
            self.U = mat
            super().__init__(mat, q)

    class SqZ(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[1,0],[0,1j]], dtype=cp.complex128)
            super().__init__(mat,q)

    class SqX(OneQubitGate):
        def __init__(self, q):
            # H, sqZ in CuPy
            self.q = q
            h = cp.array([[1/math.sqrt(2), 1/math.sqrt(2)],
                        [1/math.sqrt(2),-1/math.sqrt(2)]], dtype=cp.complex128)
            sqZ = cp.array([[1,0],[0,1j]], dtype=cp.complex128)
            mat = h @ sqZ @ h
            super().__init__(mat,q)

    class Y(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[0,-1j],[1j,0]], dtype=cp.complex128)
            super().__init__(mat,q)

    class Z(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[1,0],[0,-1]], dtype=cp.complex128)
            super().__init__(mat,q)

    class H(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[1/math.sqrt(2),1/math.sqrt(2)],
                            [1/math.sqrt(2),-1/math.sqrt(2)]], dtype=cp.complex128)
            super().__init__(mat,q)

    class I(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[1,0],[0,1]], dtype=cp.complex128)
            super().__init__(mat,q)

    class T(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[1,0],[0,cp.exp(1j*math.pi/4)]], dtype=cp.complex128)
            super().__init__(mat,q)

    class CNOT(TwoQubitGate):
        def __init__(self, q0, q1):
            self.q0, self.q1 = q0, q1
            self.U = cp.array([[0,1],[1,0]], dtype=cp.complex128)
            mat = cp.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,0,1],
                [0,0,1,0]], dtype=cp.complex128)
            super().__init__(mat, q0, q1)

    class V(OneQubitGate):
        def __init__(self, q):
            self.q = q
            mat = cp.array([[1,-1j],
                            [-1j,1]], dtype=cp.complex128)
            mat *= ((1j+1)/2)
            super().__init__(mat,q)


    #############################################################
    # controlU
    #############################################################
    class controlU(TwoQubitGate):
        def __init__(self, U, q0, q1):
            self.q0, self.q1 = q0, q1
            Ucp = cp.asarray(U)
            self.U = Ucp
            u00,u01= Ucp[0,0],Ucp[0,1]
            u10,u11= Ucp[1,0],Ucp[1,1]
            u00_ = u00.item()
            u01_ = u01.item()
            u10_ = u10.item()
            u11_ = u11.item()
       

            mat = cp.array([
                [1,   0,    0,    0   ],
                [0,   1,    0,    0   ],
                [0,   0,   u00_,  u01_],
                [0,   0,   u10_,  u11_]
                ], dtype=cp.complex128)
            super().__init__(mat, q0, q1)

    #############################################################
    # Example S class
    #############################################################
    class Module:
        def __init__(self):
            self.circuit = Circuit._active_circuit
            self.layers = []

        def makeControlled(self, ctrls, ancillas):
            if len(ancillas) < len(ctrls)-1:
                raise ValueError("Not enough ancillas to make this a controlled module!")
            if isinstance(self, self.circuit.GeneralToffoli):
                controls = ctrls.copy()
                controls.extend(self.controls)
                anc = ancillas.copy()
                anc.extend(self.ancillas)
                # print(self, controls, self.targets, anc)
                self = self.circuit.GeneralToffoli(controls, self.targets, anc)
                return self
            for i, layer in enumerate(self.layers):
                
                if isinstance(layer, self.circuit.OneQubitGate):
                    U = layer.matrix
                    self.layers[i] = self.circuit.GeneralToffoliU(ctrls, [layer.q], ancillas, U)
                  
                elif isinstance(layer, self.circuit.TwoQubitGate):
                    U = layer.U
                    controls = ctrls.copy()
                    controls.append((layer.q0, 1))
                    self.layers[i] = self.circuit.GeneralToffoliU(controls, [layer.q1], ancillas, U)
                elif isinstance(layer, self.circuit.GeneralToffoli):
                    controls = ctrls.copy()
                    controls.extend(self.controls)
                    anc = ancillas.copy()
                    anc.extend(self.ancillas)
                    # print(self, controls, self.targets, anc)
                    self.layers[i] = self.circuit.GeneralToffoli(controls, self.targets, anc)

                else: 
                    layer.makeControlled(ctrls, ancillas)
         
           
            return self
            
        def act(self):
            for layer in self.layers: 
                layer.act()
            return self.circuit.state_vec
        def decompose(self):
            for layer in self.layers: 
                # print(layer)
                if isinstance(layer, self.circuit.OneQubitGate):
                    print(layer, layer.q)
                    
                elif isinstance(layer, self.circuit.TwoQubitGate):
                    print(layer, layer.ctrl, layer.target)
                else:
                    layer.decompose()

        def sqrt(self, matrix):
            U_cpu = cp.asnumpy(matrix)
            sqrtU_cpu = scipy.linalg.sqrtm(U_cpu)
            sqrtU = cp.asarray(sqrtU_cpu)
            return sqrtU
        def dagger(self, matrix):
            return matrix.conj().T
        

    class S(Module):
        def __init__(self, p, zero, q):
            self.p, self.zero, self.q= p, zero, q
            super().__init__()
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
            self.layers.append(self.circuit.controlU(self.circuit.SqX(self.target).dagger(sqx_matrix), self.c1, self.target))
            self.layers.append(self.circuit.CNOT(self.c0, self.c1))
            self.layers.append(self.circuit.controlU(sqx_matrix, self.c1, self.target))

    class ToffoliU(Module):
        def __init__(self, c0, c1, target, U):
 
            super().__init__()
            self.U = U
            self.c0 = c0
            self.c1 = c1
            self.target = target
            self.layers.append(self.circuit.controlU(self.sqrt(U), self.c0, self.target))
            self.layers.append(self.circuit.CNOT(self.c0, self.c1))
            self.layers.append(self.circuit.controlU(self.dagger(self.sqrt(U)), self.c1, self.target))
            self.layers.append(self.circuit.CNOT(self.c0, self.c1))
            self.layers.append(self.circuit.controlU(self.sqrt(U), self.c1, self.target))


    #############################################################
    # ... (Similarly for other classes like nFoldX, etc.)
    #############################################################


    ###############################################################################
    # nFoldX Class (on GPU/CuPy)
    ###############################################################################
    class nFoldX(Module):
        """
        A general multi-controlled X gate on 1 target, given 'n' controls 
        plus some ancillas. If 'n-2=0' => normal Toffoli with 2 controls. 
        If 'n-2=1', we do a simpler chain, else we do a bigger chain approach.
        """
        def __init__(self, controls, target, ancillas):
            """
            n: total # of controls
            controls: the qubit indices for those controls
            target: single qubit index to flip
            ancillas: extra qubits used in chain
            state_vector: wavefunction
            """

            super().__init__()
            n = len(controls)
            if len(ancillas) < n-2:
                raise ValueError("Must have at least n-2 ancillas")
            
            if n-2 == 0:
                # directly do a standard Toffoli with 2 controls
                self.layers.append(self.circuit.Toffoli(controls[0], controls[1], target))
            elif n-2 == 1:
                # do a simpler approach with one ancilla
                # e.g. tX => toff => tX => toff
                tX = self.circuit.Toffoli(ancillas[0], controls[n-1], target)
                toff = self.circuit.Toffoli(controls[0], controls[1], ancillas[0])
 
                self.layers.append(tX)
                self.layers.append(toff)
                self.layers.append(tX)
                self.layers.append(toff)

           
            else:
          
                tX = self.circuit.Toffoli(ancillas[0], controls[n-1], target)

                toffoli_gates = []
                for i in range(n-3):
          
                    tgate = self.circuit.Toffoli(ancillas[i+1], controls[n-2-i], ancillas[i])
                 
                    toffoli_gates.append(tgate)
                middle_toffoli = self.circuit.Toffoli(controls[0], controls[1], ancillas[n-3])
      
                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))
                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))
    class nFoldU(Module):
        """
        A general multi-controlled X gate on 1 target, given 'n' controls 
        plus some ancillas. If 'n-2=0' => normal Toffoli with 2 controls. 
        If 'n-2=1', we do a simpler chain, else we do a bigger chain approach.
        """
        def __init__(self, controls, target, ancillas, U):
            """
            n: total # of controls
            controls: the qubit indices for those controls
            target: single qubit index to flip
            ancillas: extra qubits used in chain
            state_vector: wavefunction
            """

            super().__init__()
            self.U = U
            n = len(controls)
            if len(ancillas) < n-2:
                raise ValueError("Must have at least n-2 ancillas")
            
            if n-2 == 0:
                # directly do a standard Toffoli with 2 controls
                self.layers.append(self.circuit.ToffoliU(controls[0], controls[1], target, U))
            elif n-2 == 1:
                # do a simpler approach with one ancilla
                # e.g. tX => toff => tX => toff
                tX = self.circuit.ToffoliU(ancillas[0], controls[n-1], target, U)
                toff = self.circuit.ToffoliU(controls[0], controls[1], ancillas[0], U)
 
                self.layers.append(tX)
                self.layers.append(toff)
                self.layers.append(tX)
                self.layers.append(toff)

           
            else:
          
                tX = self.circuit.ToffoliU(ancillas[0], controls[n-1], target, U)

                toffoli_gates = []
                for i in range(n-3):
          
                    tgate = self.circuit.ToffoliU(ancillas[i+1], controls[n-2-i], ancillas[i], U)
                 
                    toffoli_gates.append(tgate)
                middle_toffoli = self.circuit.ToffoliU(controls[0], controls[1], ancillas[n-3], U)
      
                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))
                self.layers.append(tX)
                self.layers.extend(toffoli_gates)
                self.layers.append(middle_toffoli)
                self.layers.extend(reversed(toffoli_gates))

    ###############################################################################
    # GeneralToffoli Class (on GPU/CuPy)
    ###############################################################################
    class GeneralToffoli(Module):
        """
        For each (cqubit, cval) in 'controls', if cval=0 we apply X, 
        i.e. 'flip' => effectively we treat all controls as 1. Then do 
        nFoldX => flip back => that is the logic behind the gen_flip_zero_controls approach.
        We also allow multiple target qubits => repeat nFoldX for each target.
        """
        def __init__(self, controls, targets, ancillas):
            """
            dim: total # qubits => 2^dim length state
            controls: e.g. [(qA,0), (qB,1), ...]
            targets: list of target qubits
            ancillas: if needed for the chain approach
            state_vector: wavefunction
            """
            super().__init__()
           

            self.controls = controls
            self.targets = targets
            self.ancillas = ancillas
            ctrls  = [control[0] for control in controls]
            for (control, x) in self.controls: 
                if x == 0:
                    self.layers.append(self.circuit.X(control))
            for target in targets:
                self.layers.append(self.circuit.nFoldX(ctrls, target, ancillas))
            for (control, x) in self.controls: 
                if x == 0:
                    self.layers.append(self.circuit.X(control))

    class GeneralToffoliU(Module):
        """
        For each (cqubit, cval) in 'controls', if cval=0 we apply X, 
        i.e. 'flip' => effectively we treat all controls as 1. Then do 
        nFoldX => flip back => that is the logic behind the gen_flip_zero_controls approach.
        We also allow multiple target qubits => repeat nFoldX for each target.
        """
        def __init__(self, controls, targets, ancillas, U):
            """
            dim: total # qubits => 2^dim length state
            controls: e.g. [(qA,0), (qB,1), ...]
            targets: list of target qubits
            ancillas: if needed for the chain approach
            state_vector: wavefunction
            """
            super().__init__()
           
            self.U = U
            self.controls = controls
            self.targets = targets
            ctrls  = [control[0] for control in controls]
            for (control, x) in self.controls: 
                if x == 0:
                    self.layers.append(self.circuit.X(control))
            for target in targets:
                self.layers.append(self.circuit.nFoldU(ctrls, target, ancillas, U))
            for (control, x) in self.controls: 
                if x == 0:
                    self.layers.append(self.circuit.X(control))

    ###############################################################################
    # Comparator
    ###############################################################################
    class Comparator(Module): 
        """
        Uses a series of GeneralToffoli calls. Example logic from your code.
        """
        def __init__(self, a, b, comparator_zeros, ancillas, c1, c0):
            """
            a, b => lists of qubits
            comparator_zeros => some qubits
            ancillas => ...
            c1, c0 => single qubits?
            state_vector => wavefunction
            """
            super().__init__()
            if len(ancillas) != len(comparator_zeros):
                raise ValueError("ancillas, comparator_zeros mismatch")
            self.n = (len(a)+len(b)+ len(comparator_zeros)+2+len(ancillas))
            self.a = a
            self.b = b
            self.comparator_zeros= comparator_zeros
            self.ancillas= ancillas
            self.c1= c1
            self.c0= c0
            i=0
            for a_i,b_i in zip(self.a, self.b):
                controls = [(a_i,1),(b_i,0)]
                j=0
                while j< i:
                    controls.append((self.comparator_zeros[j], 0))
                    j+=1
                if i< len(self.comparator_zeros):
                    targets= [self.comparator_zeros[i], self.c1]
                else:
                    targets= [self.c1]
                T1 = self.circuit.GeneralToffoli(controls.copy(), targets.copy(), self.ancillas)
           
                self.layers.append(T1)
                print("T1", controls, targets, self.ancillas)
                i+=1
                # flip the first two controls
                controls[0]= (a_i,0)
                controls[1]= (b_i,1)
                if i< len(self.comparator_zeros):
                    targets= [self.comparator_zeros[i], self.c0]
                else:
                    targets= [self.c0]
                T2= self.circuit.GeneralToffoli(controls.copy(), targets.copy(), self.ancillas)
               
                self.layers.append(T2)
                print("T2", controls, targets, self.ancillas)
                i+=1
            for layer in self.layers: 
                print(layer.controls, layer.targets)

    class Carry(Module):
        def __init__(self, a, b, c, d):
            super().__init__()
            self.a,self.b,self.c,self.d= a, b, c, d
            self.layers.append(self.circuit.Toffoli(self.b, self.c, self.d))
            self.layers.append(self.circuit.CNOT(self.b, self.c))
            self.layers.append(self.circuit.Toffoli(self.a, self.c, self.d))
            


    class CarryBack(Module):
        def __init__(self, a, b, c, d):
            super().__init__()
            self.a,self.b,self.c,self.d= a, b, c, d
            self.layers.append(self.circuit.Toffoli(self.a, self.c, self.d))
            self.layers.append(self.circuit.CNOT(self.b, self.c))
            self.layers.append(self.circuit.Toffoli(self.b, self.c, self.d))
            

    class SUM(Module):
        def __init__(self, a, b, s):
            super().__init__()
            self.a,self.b,self.s = a, b, s
            self.layers.append(self.circuit.CNOT(self.b, self.s))
            self.layers.append(self.circuit.CNOT(self.a, self.s))


    class Adder(Module):
        """
        Ripple-carry adder that does:
        1) 'ripple down' with Carry gates
        2) 'ripple up' with CarryBack and some SUM gates
        The final big_U is the product of all submodule unitaries in reverse order.
        """
        def __init__(self, a, b, c):
            super().__init__()
            assert len(b) == len(a) + 1 and len(a) == len(c)
            self.a, self.b, self.c = a, b, c
     
    
      
            for i in range(len(self.a)):
                if i == len(self.a)-1:
                    # final carry
                    carryU = self.circuit.Carry(self.c[i], self.a[i], self.b[i], self.b[i+1])
                    # carryU = Carry([self.c[i], self.a[i], self.b[i], self.b[i+1]], self.state_vector).U
                    self.layers.append(carryU)
                else:
                    carryU = self.circuit.Carry(self.c[i], self.a[i], self.b[i], self.c[i+1])

                    # carryU = Carry([self.c[i], self.a[i], self.b[i], self.c[i+1]], self.state_vector).U
                    self.layers.append(carryU)

            for i in reversed(range(len(self.a))):
                if i == len(self.a)-1:
                    # The top carry fix
                    cnotU = self.circuit.CNOT(self.a[i], self.b[i])
                    # cnotU = Operator(self.state_vector, [self.a[i], self.b[i]], CNOT().op()).U
                    self.layers.append(cnotU)
                    sumU = self.circuit.SUM(self.c[i], self.a[i], self.b[i])
                    # sumU = SUM([self.c[i], self.a[i], self.b[i]], self.state_vector).U
                    self.layers.append(sumU)
                else:
                    cbackU = self.circuit.CarryBack(self.c[i], self.a[i], self.b[i], self.c[i+1])
                    # cbackU = CarryBack([self.c[i], self.a[i], self.b[i], self.c[i+1]], self.state_vector).U
                    self.layers.append(cbackU)
                    sumU = self.circuit.SUM(self.c[i], self.a[i], self.b[i])
                    # sumU = SUM([self.c[i], self.a[i], self.b[i]], self.state_vector).U
                    self.layers.append(sumU)




    ###############################################################################
    # AS class in CuPy
    ###############################################################################
    class AS(Module):
        """
        The 'AS' module with no explicit control, 
        always performing:
            - For each triple (a,b,zero) in zip(self.a[:-1], self.b[:-1], self.c): S(...)
            - Then a CNOT on [b[-1], a[-1]]
            - Then an Adder(...) on (a[:-1], b, c)
        """
        def __init__(self, a, b, c):
            super().__init__()
            self.a = a
            self.b = b
            self.c = c

            self.layers.append(self.circuit.Adder(self.a[:-1], self.b, self.c))
            self.layers.append(self.circuit.CNOT(self.b[-1], self.a[-1]))
      
            # For each triple, do big_U = big_U @ S([a_i, zero_i, b_i], self.state_vector).U
            for a_i, b_i, zero_i in zip(self.a[:-1], self.b[:-1], self.c):
                self.layers.append(self.circuit.S(a_i, zero_i, b_i))

    ###############################################################################
    # ASS class in CuPy
    ###############################################################################
    class ASS:
        """
        Some bigger module that does:
        - For i in range(len(self.inc)):
        * Adder(self.inc[i], self.y, ...)
        * Comparator(...)
        * AS_00(...) 
        We'll not finalize the logic, just rewrite to use CuPy.
        """
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
            # The code snippet is incomplete in your original snippet. 
            # We'll define 'layers', but the logic is up to you to finalize.
            layers = []
            for i in range(len(self.inc)):
                # 1) Adder(...).U
                AM_MODULE = Adder(self.inc[i], self.y, self.state_vector).U  # Possibly incomplete
                layers.append(AM_MODULE)

                # 2) Comparator(...).U
                # comparator = Comparator(self.x, self.y, self.zeros, self.c1, self.c0, self.state_vector).U
                # layers.append(comparator)

                # 3) AS_00(...) 
                # as_00_mod = AS_00(self.dim, [self.c1,self.c0], self.a_x, self.b_y, self.c, self.inc, self.state_vector).U
                # layers.append(as_00_mod)

                # You presumably chain them with matrix multiplications
                # We'll not do a final reduce(...) since it's not shown
                pass
            # Return an identity for now, or reduce over 'layers'
            if layers:
                return reduce(operator.matmul, layers)
            else:
                return cp.eye(len(self.state_vector), dtype=cp.complex128)


   



state_vector = binary_to_state_vector("00000100110")

circuit = Circuit(state_vector)
# module = circuit.GeneralToffoliU([(2, 1), (1, 1)], [0], [4], circuit.X(0))



print("=======================")
# module = circuit.GeneralToffoli([(2, 1), (1, 1)], [0], [5]).makeControlled([(3, 0), (4, 0)], [6])
# module = circuit.GeneralToffoli([(2, 1), (1, 1), (3, 0), (4, 0)], [0], [5, 6])
module = circuit.AS([5, 4, 3], [2, 1, 0], [7, 6]).makeControlled([(9, 0), (8, 0)], [10])
circuit.addLayer(module)
circuit.decompose()

# # gate = circuit.X(0)
# # print(gate.sqrt(gate.matrix))
# # print(gate.dagger(gate.sqrt(gate.matrix)))


circuit.run()

