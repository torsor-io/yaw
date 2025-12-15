"""
yaw: Quantum context management language
Production prototype using SymPy backend

This module provides a functional approach to quantum computing based on:
- Algebraic operator manipulation
- Expectation functionals for quantum states
- Symbolic normalization with context management
- Tensor product algebra for multi-qubit systems
"""

from sympy.physics.quantum import Operator, Dagger
from sympy import sqrt, expand, Mul, Add, Pow
from sympy.core.numbers import Number as SympyNumber
from typing import List, Tuple

import re
import random
import numpy as np

__version__ = "0.1.1"
__all__ = [
    'YawOperator', 'TensorProduct', 'Algebra', 'Context', 'qudit',
    'State', 'EigenState', 'TensorState', 'ConjugatedState', 
    'TransformedState', 'CollapsedState', 'TensorSum', 'SuperpositionState',
    'OpChannel', 'opChannel', 'StChannel', 'stChannel',
    'OpMeasurement', 'opMeasure', 'StMeasurement', 'stMeasure',
    'OpBranches', 'opBranches', 'StBranches', 'stBranches',
    'compose_st_branches', 'compose_op_branches',
    'tensor', 'tensor_power', 'char', 'conj_op', 'conj_state',
    'QFT', 'qft', 'proj', 'proj_general', 'ctrl', 'ctrl_spectral', 'ctrl_single',
    'Projector', 'proj_algebraic', 'qubit', 'Encoding', 'rep',
    'comm', 'acomm',
    'StabilizerCode', 'five_qubit_code', 'bit_flip_code',
    'gnsVec', 'gnsMat', 'spec', 'minimal_poly', 'MixedState', 'mixed'
]

# ============================================================================
# CONTEXT MANAGEMENT
# ============================================================================


def _clean_number(num, decimals=10):
    """Clean numerical noise from numbers.
    
    Rounds floats and complex numbers to remove numerical errors.
    Converts numbers very close to integers to exact integers.
    Converts complex numbers with zero imaginary part to real.
    
    Args:
        num: Number to clean (int, float, or complex)
        decimals: Number of decimal places to round to (default 10)
    
    Returns:
        Cleaned number
    """
    if isinstance(num, (int, bool, type(None))):
        return num
    
    if isinstance(num, float):
        # Round to remove noise
        rounded = round(num, decimals)
        # If very close to zero, make it exactly zero
        if abs(rounded) < 10**(-decimals):
            return 0.0
        # If very close to an integer, make it an integer
        if abs(rounded - round(rounded)) < 10**(-decimals):
            return int(round(rounded))
        return rounded
    
    if isinstance(num, complex):
        # Clean real and imaginary parts separately
        real_clean = _clean_number(num.real, decimals)
        imag_clean = _clean_number(num.imag, decimals)
        
        # If imaginary part is zero, return just the real part
        if imag_clean == 0:
            return real_clean
        # If real part is zero, return imaginary as complex
        if real_clean == 0:
            return complex(0, imag_clean)
        
        return complex(real_clean, imag_clean)
    
    return num


class Context:
    """Context management with variable linking and inheritance.
    
    Manages three namespaces:
    - temp: Temporary variables (_gens, _rels)
    - global_: Global variables ($gens, $rels, $alg, $state)
    - user: User-defined variables (H, psi0, etc.)
    
    Supports context graph (GCG) for inheritance:
    - link(parent, child): parent inherits from child
    - unlink(parent, child): stop inheritance
    
    Default behavior: $gens and $rels inherit from _gens and _rels
    """
    
    def __init__(self):
        """Initialize context with empty namespaces and default links."""
        self.temp = {}
        self.global_ = {}
        self.user = {}
        
        # Context graph: parent -> set of children
        self.links = {
            '$gens': {'_gens'},  # Default: $gens inherits from _gens
            '$rels': {'_rels'}
        }
        
        # Initialize empty generator/relation sets
        self.temp['_gens'] = set()
        self.temp['_rels'] = set()
        self.global_['$gens'] = set()
        self.global_['$rels'] = set()
    
    def link(self, parent, child):
        """Establish inheritance: parent inherits from child.
        
        Args:
            parent: Variable name (e.g., '$gens')
            child: Variable name to inherit from (e.g., '_gens')
        """
        if parent not in self.links:
            self.links[parent] = set()
        self.links[parent].add(child)
        return f"Linked: {parent} <- {child}"
    
    def unlink(self, parent, child):
        """Remove inheritance relationship.
        
        Args:
            parent: Variable name
            child: Variable name to stop inheriting from
        """
        if parent in self.links:
            self.links[parent].discard(child)
        return f"Unlinked: {parent} -/- {child}"
    
    def get(self, name):
        """Get variable value with inheritance.
        
        Resolution order:
        1. Direct lookup in appropriate namespace
        2. If not found and variable has links, check children
        
        Args:
            name: Variable name
            
        Returns:
            Variable value or None
        """
        # Temporary variables (_vars)
        if name.startswith('_'):
            if name in self.temp:
                return self.temp[name]
            return None
        
        # Global variables ($vars)
        elif name.startswith('$'):
            # Direct lookup
            if name in self.global_:
                result = self.global_[name]
                
                # If it's a set (like $gens), merge with children via links
                if isinstance(result, set) and name in self.links:
                    merged = set(result)
                    for child in self.links[name]:
                        child_val = self.get(child)
                        if child_val and isinstance(child_val, set):
                            merged |= child_val
                    return merged
                
                return result
            
            # Not found, check inheritance
            if name in self.links:
                for child in self.links[name]:
                    child_val = self.get(child)
                    if child_val is not None:
                        return child_val
            
            return None
        
        # User variables
        else:
            return self.user.get(name)
    
    def set(self, name, value):
        """Set variable value.
        
        Args:
            name: Variable name
            value: Value to store
        """
        if name.startswith('_'):
            self.temp[name] = value
        elif name.startswith('$'):
            self.global_[name] = value
        else:
            self.user[name] = value
    
    def has(self, name):
        """Check if variable exists."""
        return self.get(name) is not None
    
    def list_vars(self):
        """List all variables."""
        return (list(self.temp.keys()) + 
                list(self.global_.keys()) + 
                list(self.user.keys()))
    
    def add_to_temp_gens(self, gen_name):
        """Add generator to _gens (auto-accumulation).
        
        Args:
            gen_name: Generator name to add
        """
        if '_gens' not in self.temp:
            self.temp['_gens'] = set()
        self.temp['_gens'].add(gen_name)

# ============================================================================
# QUANTUM ERROR CORRECTION - ENCODINGS
# ============================================================================

class Encoding:
    """Functor from logical algebra to physical algebra.
    
    An encoding implements a quantum error correction code as an algebra
    homomorphism. It maps logical operators to physical operators while
    preserving algebraic structure:
    
        encode(A + B) = encode(A) + encode(B)
        encode(A * B) = encode(A) * encode(B)
        encode(c * A) = c * encode(A)
    
    This functional approach makes QEC codes first-class algebraic objects.
    
    Example:
        >>> # 3-qubit repetition code
        >>> code = rep(3)
        >>> X_L | code  # Returns X ⊗ X ⊗ X
        >>> (X + Z) | code  # Returns (X⊗X⊗X) + (Z⊗Z⊗Z)
    """
    
    def __init__(self, logical_algebra, physical_algebra, generator_map):
        """Create encoding functor.
        
        Args:
            logical_algebra: Source algebra (logical qubits/qudits)
            physical_algebra: Target algebra (physical qubits/qudits)
            generator_map: Dict mapping logical generator names to physical operators
                          e.g., {'X': X⊗X⊗X, 'Z': Z⊗Z⊗Z, 'I': I⊗I⊗I}
        """
        self.logical_algebra = logical_algebra
        self.physical_algebra = physical_algebra
        self.generator_map = generator_map
    
    def __call__(self, logical_op):
        """Apply encoding: logical_op | encoding"""

        if isinstance(logical_op, YawOperator):
            result = self._encode_operator(logical_op)
            return result
        elif isinstance(logical_op, Projector):
            # Encode projector by encoding its base operator
            base_encoded = self._encode_operator(logical_op.base_operator)
            # Return projector in physical space
            logical_proj = logical_op.expand()
            return self._encode_operator(logical_proj)
        else:
            raise TypeError(f"Cannot encode {type(logical_op)}")

    def _encode_operator(self, op):
        """Encode operator by traversing expression tree homomorphically."""
        from sympy import Add, Mul, Pow, Symbol

        expr = op._expr

        # Recursively encode the expression
        result = self._encode_expr(expr)
        return result

    def _encode_expr(self, expr):
        """Recursively encode a SymPy expression."""
        from sympy import Add, Mul, Pow, Symbol
        from sympy.physics.quantum.operator import Operator

        # Base case: Generator (Symbol or Quantum Operator)
        if isinstance(expr, Symbol) or isinstance(expr, Operator):
            gen_name = str(expr)

            if gen_name in self.generator_map:
                return self.generator_map[gen_name]
            else:
                # Unknown generator - return wrapped
                return YawOperator(expr, self.physical_algebra)

        # Base case: Numbers (scalars)
        if expr.is_number:
            return expr

        # Recursive case: Sum
        # encode(A + B) = encode(A) + encode(B)
        if isinstance(expr, Add):
            encoded_terms = [self._encode_expr(term) for term in expr.args]

            # Sum the encoded terms
            result = encoded_terms[0]
            for term in encoded_terms[1:]:
                result = result + term

            return result

        # Recursive case: Product
        # encode(A * B) = encode(A) * encode(B)
        if isinstance(expr, Mul):
            # Separate coefficients from operators
            coeff = 1
            operator_factors = []

            for arg in expr.args:
                if arg.is_number:
                    coeff *= arg
                else:
                    operator_factors.append(self._encode_expr(arg))

            # If no operators, just return coefficient
            if not operator_factors:
                return _clean_number(coeff)

            # Multiply encoded factors together
            result = operator_factors[0]
            for factor in operator_factors[1:]:
                result = result * factor

            # Apply coefficient
            if coeff != 1:
                result = coeff * result

            return result

        # Recursive case: Power
        # encode(A^n) = encode(A)^n
        if isinstance(expr, Pow):
            base_encoded = self._encode_expr(expr.base)
            exponent = expr.exp

            # Handle integer powers
            if exponent.is_integer:
                exp_int = int(exponent)
                if exp_int == 0:
                    # A^0 = I
                    if 'I' in self.generator_map:
                        return self.generator_map['I']
                    else:
                        return 1
                elif exp_int > 0:
                    result = base_encoded
                    for _ in range(exp_int - 1):
                        result = result * base_encoded
                    return result
                else:
                    raise ValueError(f"Negative powers not supported in encoding: {expr}")
            else:
                raise ValueError(f"Non-integer powers not supported in encoding: {expr}")

        # Default: wrap in YawOperator
        return YawOperator(expr, self.physical_algebra)

    def __ror__(self, operator):
        """Enable operator | encoding syntax.
        
        Args:
            operator: YawOperator in logical algebra
            
        Returns:
            Encoded operator in physical algebra
            
        Example:
            >>> X | code  # Calls code.__ror__(X)
        """
        return self(operator)
    
    def __str__(self):
        gen_names = list(self.generator_map.keys())
        return f"Encoding([{', '.join(gen_names)}]: logical ↦ physical)"
    
    def __repr__(self):
        return f"Encoding({self.logical_algebra} ↦ {self.physical_algebra})"

def rep(n, algebra=None):
    """n-fold repetition code encoding.
    
    Maps each logical operator to n tensor copies of itself:
        X_L ↦ X ⊗ X ⊗ ... ⊗ X  (n times)
        Z_L ↦ Z ⊗ Z ⊗ ... ⊗ Z  (n times)
        I_L ↦ I ⊗ I ⊗ ... ⊗ I  (n times)
    
    This is the simplest quantum error correction code, protecting against
    single bit flips (for odd n).
    
    Args:
        n: Number of repetitions (physical qubits per logical qubit)
        algebra: Logical algebra (if None, inferred from context when applied)
        
    Returns:
        Encoding functor from logical to physical algebra
        
    Example:
        >>> $alg = qudit(2)
        >>> code = rep(3)
        >>> X | code  # X ⊗ X ⊗ X
        >>> (X + Z) | code  # (X⊗X⊗X) + (Z⊗Z⊗Z)
        
        >>> # Stabilizers for 3-qubit repetition code (bit flip)
        >>> S1 = Z @ Z @ I
        >>> S2 = I @ Z @ Z
    """
    if n < 1:
        raise ValueError(f"Number of repetitions must be at least 1, got {n}")
    
    # If algebra not provided, return a partial encoding that will infer it
    if algebra is None:
        return _PartialRepEncoding(n)
    
    # Build generator map: each generator ↦ n-fold tensor product
    generator_map = {}
    
    for gen_name, gen_op in algebra.generators.items():
        # Create n-fold tensor product
        if n == 1:
            encoded = gen_op
        else:
            # Start with first copy
            encoded = gen_op
            # Tensor with n-1 more copies
            for _ in range(n - 1):
                encoded = encoded @ gen_op
        
        generator_map[gen_name] = encoded
    
    # Also map identity
    I_op = algebra.I
    if n == 1:
        encoded_I = I_op
    else:
        encoded_I = I_op
        for _ in range(n - 1):
            encoded_I = encoded_I @ I_op
    
    generator_map['I'] = encoded_I
    
    # Physical algebra is n copies of logical algebra
    # For now, leave as None (could construct tensor product algebra)
    physical_algebra = None
    
    return Encoding(algebra, physical_algebra, generator_map)


class _PartialRepEncoding:
    """Partial repetition encoding that infers algebra from operator.
    
    This allows rep(n) to be used without explicitly passing an algebra:
        X | rep(3)
    
    The algebra is inferred from X when the encoding is applied.
    """
    
    def __init__(self, n):
        self.n = n
    
    def __ror__(self, operator):
        """Enable operator | rep(n) syntax."""
        if isinstance(operator, YawOperator):
            # Infer algebra from operator
            if operator.algebra is None:
                raise ValueError(
                    f"Cannot infer algebra from operator {operator}. "
                    f"Please provide algebra explicitly: rep({self.n}, algebra)"
                )
            
            # Create full encoding with inferred algebra
            full_encoding = rep(self.n, operator.algebra)
            
            # Apply it
            return full_encoding(operator)
        else:
            raise TypeError(f"Cannot encode {type(operator)} with rep({self.n})")
    
    def __call__(self, operator):
        """Also support code(operator) syntax."""
        return self.__ror__(operator)


# ============================================================================
# STABILIZER CODES
# ============================================================================

class StabilizerCode:
    """Stabilizer quantum error correction code.
    
    A stabilizer code is defined by:
    - Stabilizer generators: Commuting operators that stabilize the code space
    - Logical operators: Operators that commute with stabilizers but aren't in the stabilizer group
    - Encoding: Maps logical operators to physical operators
    
    Attributes:
        stabilizers: List of stabilizer generators
        logical_ops: Dict mapping logical operator names to physical operators
        encoding: Encoding functor for logical → physical
        error_table: Dict mapping syndromes to recovery operations
    """
    
    def __init__(self, stabilizers, logical_ops, encoding, error_table=None, name="Stabilizer Code"):
        """Create stabilizer code.
        
        Args:
            stabilizers: List of stabilizer generator operators
            logical_ops: Dict like {'X_L': physical_X, 'Z_L': physical_Z}
            encoding: Encoding functor
            error_table: Optional dict mapping syndrome tuples to recovery operators
            name: Human-readable name for the code
        """
        self.stabilizers = stabilizers
        self.logical_ops = logical_ops
        self.encoding = encoding
        self.error_table = error_table or {}
        self.name = name
    
    def measure_syndrome(self, state):
        """Measure stabilizer syndrome on a state.
        
        Args:
            state: Quantum state (physical qubits)
            
        Returns:
            Tuple of stabilizer measurement outcomes (each ±1)
        """
        syndrome = []
        for stab in self.stabilizers:
            # Measure expectation value of stabilizer
            outcome = state.expect(stab)
            # Convert to ±1 (assuming eigenvalues are ±1)
            syndrome.append(outcome)
        return tuple(syndrome)
    
    def lookup_correction(self, syndrome):
        """Look up error correction operation for a syndrome.
        
        Args:
            syndrome: Tuple of stabilizer outcomes
            
        Returns:
            Correction operator, or None if syndrome not in table
        """
        return self.error_table.get(syndrome)
    
    def correct_error(self, state, syndrome=None):
        """Correct errors on a state.
        
        Args:
            state: Physical state (possibly with errors)
            syndrome: Optional syndrome (if None, measure it)
            
        Returns:
            Corrected state
        """
        if syndrome is None:
            syndrome = self.measure_syndrome(state)
        
        correction = self.lookup_correction(syndrome)
        if correction is None:
            raise ValueError(f"No correction found for syndrome {syndrome}")
        
        # Apply correction: U |ψ⟩
        return correction.conj_state(state)
    
    def __str__(self):
        return f"{self.name} ({len(self.stabilizers)} stabilizers)"
    
    def __repr__(self):
        return f"StabilizerCode(name='{self.name}', n_stab={len(self.stabilizers)})"


def five_qubit_code(algebra=None):
    """Five-qubit stabilizer code (smallest perfect code).
    
    The [[5,1,3]] code is the smallest quantum code that can correct
    an arbitrary single-qubit error. It encodes 1 logical qubit into
    5 physical qubits and has distance 3.
    
    Stabilizers (4 generators):
        S1 = X Z Z X I
        S2 = I X Z Z X
        S3 = X I X Z Z
        S4 = Z X I X Z
    
    Logical operators:
        X_L = X X X X X
        Z_L = Z Z Z Z Z
    
    Args:
        algebra: Logical qubit algebra (if None, created automatically)
        
    Returns:
        StabilizerCode instance with encoding, stabilizers, and error table
        
    Example:
        >>> code = five_qubit_code()
        >>> X_L = code.logical_ops['X']
        >>> Z_L = code.logical_ops['Z']
        >>> 
        >>> # Check stabilizers commute with logical ops
        >>> S1 = code.stabilizers[0]
        >>> comm(S1, X_L).normalize()  # Should be 0
        >>> 
        >>> # Syndrome measurement
        >>> psi = char(Z_L, 0)  # Logical |0⟩
        >>> syndrome = code.measure_syndrome(psi)  # All +1 for code space
        >>> 
        >>> # Error correction
        >>> noisy_psi = (X @ I @ I @ I @ I).conj_state(psi)  # Apply X error
        >>> syndrome = code.measure_syndrome(noisy_psi)
        >>> corrected = code.correct_error(noisy_psi, syndrome)
    """
    # Create logical algebra if not provided
    if algebra is None:
        algebra = Algebra(
            gens=['X', 'Z'],
            rels=['herm', 'unit', 'anti', 'pow(2)']
        )
    
    X = algebra.X
    Z = algebra.Z
    I = algebra.I
    
    # Define stabilizer generators on 5 physical qubits
    # S1 = X Z Z X I
    S1 = X @ Z @ Z @ X @ I
    # S2 = I X Z Z X
    S2 = I @ X @ Z @ Z @ X
    # S3 = X I X Z Z
    S3 = X @ I @ X @ Z @ Z
    # S4 = Z X I X Z
    S4 = Z @ X @ I @ X @ Z
    
    stabilizers = [S1, S2, S3, S4]
    
    # Define logical operators
    # X_L = X X X X X (all X)
    X_L = X @ X @ X @ X @ X
    # Z_L = Z Z Z Z Z (all Z)
    Z_L = Z @ Z @ Z @ Z @ Z
    
    logical_ops = {
        'X': X_L,
        'Z': Z_L,
        'I': I @ I @ I @ I @ I
    }
    
    # Create encoding functor
    generator_map = {
        'X': X_L,
        'Z': Z_L,
        'I': I @ I @ I @ I @ I
    }
    
    encoding = Encoding(algebra, None, generator_map)
    
    # Build error correction table
    # For single-qubit Pauli errors, map syndrome → correction
    error_table = {}
    
    # Identity (no error) → all stabilizers +1
    error_table[(1, 1, 1, 1)] = I @ I @ I @ I @ I
    
    # Single-qubit X errors
    # These can be computed by checking which stabilizers anticommute with X_i
    # For now, we'll leave the error table incomplete as a placeholder
    # A complete implementation would enumerate all 15 single-qubit Pauli errors
    
    # Example: X error on qubit 0 
    # X_0 anticommutes with S1 and S3 → flips their signs
    # Syndrome: (-1, 1, -1, 1) means apply X on qubit 0
    error_table[(-1, 1, -1, 1)] = X @ I @ I @ I @ I
    
    # More entries would go here...
    # In practice, compute this systematically or generate on the fly
    
    return StabilizerCode(
        stabilizers=stabilizers,
        logical_ops=logical_ops,
        encoding=encoding,
        error_table=error_table,
        name="[[5,1,3]] Five-Qubit Code"
    )


def bit_flip_code(algebra=None):
    """Three-qubit bit-flip code (repetition code for X errors).
    
    The [[3,1,1]] bit-flip code protects against single X (bit-flip) errors
    by encoding one logical qubit into three physical qubits. This is the
    quantum version of the classical repetition code.
    
    Stabilizers (2 generators):
        S1 = Z Z I  (checks qubits 0 and 1 have same Z value)
        S2 = I Z Z  (checks qubits 1 and 2 have same Z value)
    
    Logical operators:
        X_L = X X X  (flip all three bits)
        Z_L = Z I I  (measure any single Z)
    
    Code space: {|000⟩, |111⟩} encodes {|0⟩_L, |1⟩_L}
    
    Args:
        algebra: Logical qubit algebra (if None, created automatically)
        
    Returns:
        StabilizerCode instance with complete error correction
        
    Example:
        >>> code = bit_flip_code()
        >>> X_L = code.logical_ops['X']
        >>> Z_L = code.logical_ops['Z']
        >>> 
        >>> # Encode logical |0⟩ → |000⟩
        >>> psi_L = char(Z_L, 0)  # Logical |0⟩ 
        >>> # (In code space, this is |000⟩)
        >>> 
        >>> # Apply bit flip error on qubit 1
        >>> error = I @ X @ I
        >>> noisy = error.conj_state(psi_L)
        >>> 
        >>> # Measure syndrome
        >>> syndrome = code.measure_syndrome(noisy)
        >>> # Should be (-1, 1) indicating error on qubit 1
        >>> 
        >>> # Correct
        >>> corrected = code.correct_error(noisy, syndrome)
    """
    # Create logical algebra if not provided
    if algebra is None:
        algebra = Algebra(
            gens=['X', 'Z'],
            rels=['herm', 'unit', 'anti', 'pow(2)']
        )
    
    X = algebra.X
    Z = algebra.Z
    I = algebra.I
    
    # Define stabilizer generators on 3 physical qubits
    # S1 = Z Z I  (qubits 0,1 have same phase)
    S1 = Z @ Z @ I
    # S2 = I Z Z  (qubits 1,2 have same phase)
    S2 = I @ Z @ Z
    
    stabilizers = [S1, S2]
    
    # Define logical operators
    # X_L = X X X (flip all bits)
    X_L = X @ X @ X
    # Z_L = Z I I (measure first qubit's phase)
    # Could also use Z @ I @ I or I @ Z @ I - all equivalent in code space
    Z_L = Z @ I @ I
    
    logical_ops = {
        'X': X_L,
        'Z': Z_L,
        'I': I @ I @ I
    }
    
    # Create encoding functor
    generator_map = {
        'X': X_L,
        'Z': Z_L,
        'I': I @ I @ I
    }
    
    encoding = Encoding(algebra, None, generator_map)
    
    # Build complete error correction table
    # Syndromes are (S1, S2) where each is ±1
    error_table = {}
    
    # No error: both stabilizers measure +1
    error_table[(1, 1)] = I @ I @ I
    
    # X error on qubit 0: flips S1 (which checks 0,1)
    # Syndrome: S1=-1, S2=+1
    error_table[(-1, 1)] = X @ I @ I
    
    # X error on qubit 1: flips both S1 and S2 (1 is in both checks)
    # Syndrome: S1=-1, S2=-1
    error_table[(-1, -1)] = I @ X @ I
    
    # X error on qubit 2: flips S2 (which checks 1,2)
    # Syndrome: S1=+1, S2=-1
    error_table[(1, -1)] = I @ I @ X
    
    return StabilizerCode(
        stabilizers=stabilizers,
        logical_ops=logical_ops,
        encoding=encoding,
        error_table=error_table,
        name="[[3,1,1]] Bit-Flip Code"
    )
    
    def __str__(self):
        return f"rep({self.n})"
    
    def __repr__(self):
        return f"rep({self.n})"



# ============================================================================
# OPERATORS
# ============================================================================
    
class YawOperator:
    """Operator in a non-commutative algebra with context management.
    
    Wraps a SymPy expression with yaw semantics, including:
    - Non-commutative multiplication
    - Adjoint (Hermitian conjugate)
    - Context-aware normalization
    - Conjugation operations
    
    Attributes:
        _expr: Internal SymPy expression
        algebra: Associated Algebra for normalization rules
    """
    
    def __init__(self, sympy_expr, algebra=None):
        """Create operator from SymPy expression.
        
        Args:
            sympy_expr: SymPy quantum operator expression
            algebra: Associated Algebra (optional)
        """
        self._expr = sympy_expr
        self.algebra = algebra

    def __ror__(self, other):
        """Enable A | context syntax.
        
        Handles:
        - state | A  ↦ expectation value (existing)
        - A | encoding  ↦ encoded operator (new)
        """
        if isinstance(other, State):
            # Existing: expectation value
            return other.expect(self)
        elif isinstance(other, Encoding):
            # New: apply encoding
            return other(self)
        else:
            # Try to interpret as encoding if it's callable
            if callable(other):
                return other(self)
            raise TypeError(f"Cannot evaluate operator with {type(other)}")

    def __invert__(self):
        """Enable ~A syntax as shorthand for A.normalize()
        
        Example:
            >>> ~(X + Z)  # Same as (X + Z).normalize()
            >>> ~comm(X, Z)  # Same as comm(X, Z).normalize()
        """
        return self.normalize()
        
    def __mul__(self, other):
        """Non-commutative multiplication.
        
        Supports:
        - YawOperator * YawOperator
        - YawOperator * TensorSum (distributivity)
        - YawOperator * TensorProduct
        - YawOperator * scalar
        """
        if isinstance(other, YawOperator):
            return YawOperator(self._expr * other._expr, self.algebra)
        elif isinstance(other, TensorSum):
            # Distribute: A * (B + C) = A*B + A*C
            return TensorSum([self * term for term in other.terms])
        elif isinstance(other, TensorProduct):
            # Promote self to tensor product and multiply element-wise
            # A * (B⊗C) treated as (A⊗I⊗...) * (B⊗C)
            # For now, just multiply first factor
            new_factors = other.factors.copy()
            new_factors[0] = self * new_factors[0]
            return TensorProduct(new_factors)
        else:
            # Scalar or SymPy expression
            return YawOperator(self._expr * other, self.algebra)

    def __matmul__(self, other):
        """Tensor product using @ operator: A @ B = A ⊗ B

        Distributes over sums on both sides.
        """
        from sympy import Add, Mul

        # *** FIRST: Check if SELF (left operand) is a sum ***
        # Handle case where expr might be coeff * (A + B)
        expr_to_check = self._expr
        coeff = 1
        
        if isinstance(expr_to_check, Mul):
            # Extract numerical coefficient
            coeff_factors = []
            non_coeff_factors = []
            for arg in expr_to_check.args:
                if arg.is_number:
                    coeff_factors.append(arg)
                else:
                    non_coeff_factors.append(arg)
            
            if coeff_factors:
                from sympy import Mul as SympyMul
                coeff = SympyMul(*coeff_factors) if len(coeff_factors) > 1 else coeff_factors[0]
                if len(non_coeff_factors) == 1:
                    expr_to_check = non_coeff_factors[0]
                elif len(non_coeff_factors) > 1:
                    expr_to_check = SympyMul(*non_coeff_factors)
                # else: pure coefficient, handled below
        
        if isinstance(expr_to_check, Add):
            # Distribute: coeff * (A + B) @ C = coeff * (A @ C + B @ C)
            terms = []
            for term_expr in expr_to_check.args:
                term_op = YawOperator(term_expr, self.algebra)
                tensor_product = term_op @ other
                terms.append(tensor_product)
            result = TensorSum(terms)
            if coeff != 1:
                result = coeff * result
            return result

        # *** SECOND: Check if OTHER (right operand) is a sum ***
        if isinstance(other, YawOperator):
            other_expr = other._expr
            other_coeff = 1
            
            if isinstance(other_expr, Mul):
                # Extract numerical coefficient from other
                coeff_factors = []
                non_coeff_factors = []
                for arg in other_expr.args:
                    if arg.is_number:
                        coeff_factors.append(arg)
                    else:
                        non_coeff_factors.append(arg)
                
                if coeff_factors:
                    from sympy import Mul as SympyMul
                    other_coeff = SympyMul(*coeff_factors) if len(coeff_factors) > 1 else coeff_factors[0]
                    if len(non_coeff_factors) == 1:
                        other_expr = non_coeff_factors[0]
                    elif len(non_coeff_factors) > 1:
                        other_expr = SympyMul(*non_coeff_factors)
            
            if isinstance(other_expr, Add):
                # Distribute: A @ (coeff * (B + C)) = coeff * (A @ B + A @ C)
                terms = []
                for term_expr in other_expr.args:
                    term_op = YawOperator(term_expr, other.algebra)
                    tensor_product = self @ term_op
                    terms.append(tensor_product)
                result = TensorSum(terms)
                if other_coeff != 1:
                    result = other_coeff * result
                return result

        # *** THIRD: Handle simple cases ***
        if isinstance(other, YawOperator):
            # Simple tensor product
            return TensorProduct([self, other])
        elif isinstance(other, TensorProduct):
            # A @ (B ⊗ C) = A ⊗ B ⊗ C
            return TensorProduct([self] + other.factors)
        elif isinstance(other, TensorSum):
            # A @ (B + C + ...) = A @ B + A @ C + ...
            terms = [self @ term for term in other.terms]
            return TensorSum(terms)
        else:
            raise TypeError(f"Cannot tensor YawOperator with {type(other)}")
    
    def __rmatmul__(self, other):
        """Right tensor product: supports (A + B) @ C

        Distributes: (A + B) @ C = A @ C + B @ C
        """
        from sympy import Add, Mul

        if isinstance(other, YawOperator):
            # Check if other is a sum (possibly with coefficient)
            other_expr = other._expr
            other_coeff = 1
            
            if isinstance(other_expr, Mul):
                # Extract numerical coefficient
                coeff_factors = []
                non_coeff_factors = []
                for arg in other_expr.args:
                    if arg.is_number:
                        coeff_factors.append(arg)
                    else:
                        non_coeff_factors.append(arg)
                
                if coeff_factors:
                    from sympy import Mul as SympyMul
                    other_coeff = SympyMul(*coeff_factors) if len(coeff_factors) > 1 else coeff_factors[0]
                    if len(non_coeff_factors) == 1:
                        other_expr = non_coeff_factors[0]
                    elif len(non_coeff_factors) > 1:
                        other_expr = SympyMul(*non_coeff_factors)
            
            if isinstance(other_expr, Add):
                # Distribute
                terms = []
                for term in other_expr.args:
                    term_op = YawOperator(term, other.algebra)
                    tensor_product = TensorProduct([term_op, self])
                    terms.append(tensor_product)

                # Return TensorSum with coefficient if needed
                result = TensorSum(terms)
                if other_coeff != 1:
                    result = other_coeff * result
                return result
            else:
                return TensorProduct([other, self])
        else:
            # Fall back
            return TensorProduct([other, self])
    
    def __rmul__(self, other):
        """Right multiplication (for scalar * operator)."""
        return YawOperator(other * self._expr, self.algebra)
    
    def __add__(self, other):
        """Addition of operators."""
        if isinstance(other, YawOperator):
            return YawOperator(self._expr + other._expr, self.algebra)
        elif isinstance(other, TensorProduct):
            return TensorSum([self, other])
        elif isinstance(other, TensorSum):
            return TensorSum([self] + other.terms)
        else:
            return YawOperator(self._expr + other, self.algebra)

    def __radd__(self, other):
        return YawOperator(other + self._expr, self.algebra)

    def __sub__(self, other):
        """Subtraction of operators."""
        if isinstance(other, YawOperator):
            return YawOperator(self._expr - other._expr, self.algebra)
        return YawOperator(self._expr - other, self.algebra)
    
    def __rsub__(self, other):
        """Right subtraction (for scalar - operator)."""
        return YawOperator(other - self._expr, self.algebra)

    def __neg__(self):
        """Negation: -A"""
        return YawOperator(-self._expr, self.algebra)
    
    def __truediv__(self, other):
        """Division by scalar."""
        return YawOperator(self._expr / other, self.algebra)
    
    def __pow__(self, exp):
        """Exponentiation."""
        return YawOperator(self._expr ** exp, self.algebra)
    
    def adjoint(self):
        """Hermitian conjugate: A† or A*"""
        return YawOperator(Dagger(self._expr), self.algebra)
    
    @property
    def H(self):
        """Hermitian conjugate shortcut: A.H (numpy convention)
        
        Example:
            >>> X.H  # Returns X† (same as X.adjoint())
        """
        return self.adjoint()
    
    @property
    def dag(self):
        """Dagger shortcut: A.dag (alternative notation)
        
        Example:
            >>> X.dag  # Returns X† (same as X.adjoint())
        """
        return self.adjoint()
    
    @property
    def d(self):
        """Dagger shortcut: A.d (most concise)
        
        This is the recommended shortcut for adjoint/dagger operation.
        
        Example:
            >>> X.d  # Returns X† (same as X.adjoint())
            >>> (X * Z).d  # Returns Z† X† = Z X
        """
        return self.adjoint()
    
    @property
    def T(self):
        """Transpose shortcut: A.T (for quantum operators, same as adjoint)
        
        Note: In quantum mechanics, transpose typically means Hermitian conjugate.
        For real operators, this is the same as matrix transpose.
        
        Example:
            >>> X.T  # Returns X† (same as X.adjoint())
        """
        return self.adjoint()
    
    
    def lmul(self, state):
        """Left multiplication: Create functional φ where φ(B) = ψ(AB)
        
        This allows building coherent superpositions algebraically.
        
        Args:
            state: Base functional ψ
            
        Returns:
            LeftMultipliedState instance
            
        Example:
            >>> psi_00 = char(Z, 0) @ char(Z, 0)
            >>> X_X = X @ X
            >>> phi = X_X.lmul(psi_00)
            >>> # Now phi(A) computes ⟨00|(X⊗X)A|00⟩ = ⟨11|A|00⟩
        """
        return LeftMultipliedState(self, state)
    
    def rmul(self, state):
        """Right multiplication: Create functional φ where φ(B) = ψ(BA)
        
        Args:
            state: Base functional ψ
            
        Returns:
            RightMultipliedState instance
            
        Example:
            >>> psi_00 = char(Z, 0) @ char(Z, 0)
            >>> X_X = X @ X
            >>> phi = X_X.rmul(psi_00)
            >>> # Now phi(A) computes ⟨00|A(X⊗X)|00⟩ = ⟨00|A|11⟩
        """
        return RightMultipliedState(self, state)
    
    def __mod__(self, state):
        """Left multiplication using % operator: A % φ
        
        Syntactic sugar for A.lmul(φ).
        Creates functional where φ(B) = ψ(AB).
        
        Example:
            >>> phi = X_X % psi_00  # Same as X_X.lmul(psi_00)
        """
        return self.lmul(state)
    
    def __floordiv__(self, state):
        """Right multiplication using // operator: A // φ
        
        Syntactic sugar for A.rmul(φ).
        Creates functional where φ(B) = ψ(BA).
        
        Example:
            >>> phi = X_X // psi_00  # Same as X_X.rmul(psi_00)
        """
        return self.rmul(state)
    
    def normalize(self, verbose=False):
        """Apply algebra normalization rules.
        
        Args:
            verbose: If True, print normalization steps
            
        Returns:
            Normalized YawOperator
        """
        if self.algebra:
            return self.algebra.normalize(self, verbose=verbose)
        return self
    
    def conj_op(self, other):
        """Operator conjugation: self >> other = self† * other * self
        
        This is the 'conjugate by' operation for operators.
        For unitary U: U >> A = U† A U
        
        Args:
            other: Operator to conjugate
            
        Returns:
            Conjugated operator
        """
        return self.adjoint() * other * self
    
    def conj_state(self, state):
        """State conjugation: self << state
        
        Returns a new state transformed by this operator.
        For unitary U: U << |ψ⟩ is the transformed state.
        
        Args:
            state: State to transform
            
        Returns:
            ConjugatedState object
        """
        return ConjugatedState(self, state)
    
    def __str__(self):
        """String representation with Python complex notation.
        
        Converts SymPy's 'I' (imaginary unit) to Python's '1j' notation
        to avoid confusion with the identity operator I.
        
        Strategy: Only replace 'I' when it appears as part of a complex coefficient
        (i.e., when I is multiplied with other operators), which indicates it's
        the imaginary unit, not the identity operator.
        
        Key distinction:
        - "2*I" (2 times identity) -> keep as "2*I"
        - "2.0*I*X" (2i times X) -> convert to "2.0j*X"
        - "I*X*Z" (i times X times Z) -> convert to "1j*X*Z"
        - "I" (identity alone) -> keep as "I"
        - "(1.0 + 1.0*I)*X" (complex coefficient) -> "(1.0+1.0j)*X"
        """
        s = str(self._expr)
        
        import re
        
        # Replace imaginary unit I in various contexts
        
        # 1. Pattern: number*I*OPERATOR (e.g., "1.0*I*X" -> "1j*X")
        #    This is the most common pattern for complex coefficients
        s = re.sub(r'(\d+\.?\d*)\*I\*([A-Z])', r'\1j*\2', s)
        
        # 2. Pattern: I*OPERATOR at start (e.g., "I*X*Z" -> "1j*X*Z")
        s = re.sub(r'^I\*([A-Z])', r'1j*\1', s)
        
        # 3. Pattern: + I*OPERATOR (with space preserved)
        s = re.sub(r'\+\s*I\*([A-Z])', r'+ 1j*\1', s)
        
        # 4. Pattern: - I*OPERATOR (no space after minus)
        s = re.sub(r'\-\s*I\*([A-Z])', r'-1j*\1', s)
        
        # 5. Pattern: OPERATOR*I*OPERATOR (e.g., "X*I*Z" -> "X*1j*Z")
        s = re.sub(r'([A-Z][A-Za-z0-9_]*)\*I\*([A-Z])', r'\1*1j*\2', s)
        
        # 6. Pattern: (parenthesized)*I*OPERATOR (e.g., "(-1)*I*X" -> "(-1)*1j*X")
        s = re.sub(r'\)\*I\*([A-Z])', r')*1j*\1', s)
        
        # 7. Pattern: number*I within parentheses (e.g., "(1.0 + 2.0*I)" -> "(1.0+2.0j)")
        #    This handles complex coefficients like (1+1j)*X
        s = re.sub(r'(\d+\.?\d*)\*I(?!\*[A-Z])', r'\1j', s)
        
        # 8. Pattern: standalone I within numeric expressions (e.g., "0.5*I*(X + Z)")
        #    Match I that's part of a numeric expression (preceded by *, +, -, or parenthesis)
        s = re.sub(r'([\*\+\-\(])\s*I\s*\*\s*\(', r'\g<1>1j*(', s)
        
        # Clean up: 1.0j -> 1j, 0.0j -> 0j
        s = re.sub(r'\b1\.0j\b', '1j', s)
        s = re.sub(r'([^0-9])0\.0j\b', r'\g<1>0j', s)  # Keep leading digit
        s = re.sub(r'^0\.0j\b', '0j', s)  # At start of string
        
        # Clean up spacing around complex numbers in parentheses
        s = re.sub(r'\(\s*(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)j\s*\)', r'(\1+\2j)', s)
        s = re.sub(r'\(\s*(\d+\.?\d*)\s*\-\s*(\d+\.?\d*)j\s*\)', r'(\1-\2j)', s)
        
        return s
    
    def __repr__(self):
        return f"YawOp({self._expr})"
    
    def __eq__(self, other):
        if isinstance(other, YawOperator):
            return self.normalize()._expr == other.normalize()._expr
        return False

    def __rshift__(self, other):
        """Operator conjugation: U >> A means U >> A = U† A U
        
        Usage: H >> Z instead of H.conj_op(Z)
        """
        return self.conj_op(other)
    
    def __lshift__(self, state):
        """Apply operator to state: U << φ
        
        Implements the transformation: (U << φ)(A) = φ(U† A U)
        
        This is computed algebraically as: U << φ = U† % (U // φ)
        where % is left multiplication and // is right multiplication.
        
        Special case: Identity operator returns state unchanged.
        
        Args:
            state: State functional to transform
            
        Returns:
            Transformed state functional
            
        Example:
            >>> H << psi_0  # Apply Hadamard to |0⟩
        """
        # Check if this is identity
        if self.algebra:
            try:
                normalized = self.normalize()
                if str(normalized._expr) == 'I' or normalized._expr == 1:
                    return state
            except:
                pass
        
        if str(self._expr) == 'I':
            return state
        
        # Algebraic implementation: U << φ = U† % (U // φ)
        # This computes: (U << φ)(A) = φ(U† A U)
        return self.adjoint() % (self // state)

# ============================================================================
# COMMUTATORS AND ANTICOMMUTATORS
# ============================================================================

def comm(A, B):
    """Compute commutator: [A, B] = AB - BA
    
    The commutator measures how much two operators fail to commute.
    - [A, B] = 0 means A and B commute
    - [A, B] ≠ 0 means A and B don't commute
    
    Args:
        A: First operator (YawOperator, TensorProduct, or TensorSum)
        B: Second operator
        
    Returns:
        Commutator [A, B]
        
    Example:
        >>> comm(X, Z)  # Should be non-zero for Pauli operators
        >>> comm(X, X)  # Should be zero (X commutes with itself)
    """
    return A * B - B * A


def acomm(A, B):
    """Compute anticommutator: {A, B} = AB + BA
    
    The anticommutator measures symmetric multiplication.
    - {A, B} = 0 means A and B anticommute
    - {A, B} ≠ 0 means they don't anticommute
    
    For qubits with Pauli operators:
    - {X, Z} = 0 (anticommute)
    - {X, X} = 2I (don't anticommute)
    
    Args:
        A: First operator (YawOperator, TensorProduct, or TensorSum)
        B: Second operator
        
    Returns:
        Anticommutator {A, B}
        
    Example:
        >>> acomm(X, Z)  # Should be zero for anticommuting Paulis
        >>> acomm(X, X)  # Should be 2I
    """
    return A * B + B * A
        
# ============================================================================
# TENSOR PRODUCTS
# ============================================================================

class TensorSum:
    """Sum of tensor products: A⊗B + C⊗D + ..."""
    
    def __init__(self, terms, _skip_normalize=False):
        """Create sum of terms.
        
        Args:
            terms: List of terms to sum
            _skip_normalize: Internal flag to skip auto-normalization (prevents recursion)
        """
        self.terms = list(terms)
        
        # Flatten nested TensorSums
        flattened = []
        for term in self.terms:
            if isinstance(term, TensorSum):
                flattened.extend(term.terms)
            else:
                flattened.append(term)
        self.terms = flattened
        
        # Auto-normalize unless explicitly skipped (used internally by normalize())
        if not _skip_normalize and len(self.terms) > 1:
            normalized = self.normalize()
            # If normalize returns a TensorSum, use its terms
            # (avoid double-wrapping)
            if isinstance(normalized, TensorSum):
                self.terms = normalized.terms
            elif normalized == 0:
                # Everything cancelled - represent as empty sum
                self.terms = []
            else:
                # Single term - wrap it
                self.terms = [normalized]

    def __invert__(self):
        """Enable ~A syntax for normalization."""
        return self.normalize()
        
    def __lshift__(self, state):
        """Apply sum to state: (A + B) << φ
        
        Implements: ((A + B) << φ)(C) = φ((A + B)† C (A + B))
        
        This is computed algebraically as: (A + B) << φ = (A + B)† % ((A + B) // φ)
        
        Args:
            state: State functional to transform
            
        Returns:
            Transformed state functional
        """
        # Algebraic implementation: (A + B) << φ = (A + B)† % ((A + B) // φ)
        return self.adjoint() % (self // state)
        
    def __str__(self):
        """Display as sum."""
        if not self.terms:
            return "0"
        return " + ".join(str(t) for t in self.terms)
    
    def __repr__(self):
        """REPL display."""
        return self.__str__()
    
    def __add__(self, other):
        """Add another term or sum."""
        if isinstance(other, TensorSum):
            return TensorSum(self.terms + other.terms)
        elif isinstance(other, (TensorProduct, YawOperator)):
            return TensorSum(self.terms + [other])
        else:
            raise TypeError(f"Cannot add TensorSum with {type(other)}")
    
    def __radd__(self, other):
        """Right addition."""
        if isinstance(other, (TensorProduct, YawOperator)):
            return TensorSum([other] + self.terms)
        else:
            return self.__add__(other)
    
    def __rshift__(self, operator):
        """Conjugate: (A + B) >> C = (A + B)† C (A + B)
        
        This distributes as:
            (A† + B†) C (A + B) = A†CA + A†CB + B†CA + B†CB
        
        We use the distributivity of multiplication that's already implemented.
        """
        # Compute adjoint: (A + B)† = A† + B†
        adjoint_terms = []
        for term in self.terms:
            if hasattr(term, 'adjoint'):
                adjoint_terms.append(term.adjoint())
            else:
                # For terms without adjoint, assume self-adjoint
                adjoint_terms.append(term)
        
        adjoint_sum = TensorSum(adjoint_terms, _skip_normalize=True)
        
        # Conjugation: U† C U
        # Left multiply: U† * C
        left_result = adjoint_sum * operator
        
        # Right multiply: (U† * C) * U
        final_result = left_result * self
        
        return final_result

    def __mul__(self, other):
        """Multiplication: (A + B) * C = A*C + B*C (distributivity)"""
        if isinstance(other, (int, float, complex)):
            # Scalar multiplication
            return TensorSum([term * other for term in self.terms], _skip_normalize=True)
        
        if hasattr(other, 'is_number') and other.is_number:
            # SymPy scalar multiplication
            return TensorSum([term * other for term in self.terms], _skip_normalize=True)
        
        if isinstance(other, (TensorProduct, YawOperator, TensorSum)):
            # Distribute multiplication over sum: (A + B) * C = A*C + B*C
            return TensorSum([term * other for term in self.terms])
        
        raise TypeError(f"Cannot multiply TensorSum with {type(other)}")
    
    def __rmul__(self, other):
        """Right multiplication: C * (A + B) = C*A + C*B (distributivity)"""
        if isinstance(other, (int, float, complex)):
            # Scalar multiplication
            return TensorSum([other * term for term in self.terms], _skip_normalize=True)
        
        if hasattr(other, 'is_number') and other.is_number:
            # SymPy scalar multiplication
            return TensorSum([other * term for term in self.terms], _skip_normalize=True)
        
        if isinstance(other, (TensorProduct, YawOperator, TensorSum)):
            # Distribute multiplication over sum: C * (A + B) = C*A + C*B
            return TensorSum([other * term for term in self.terms])
        
        raise TypeError(f"Cannot multiply {type(other)} with TensorSum")
    
    def __sub__(self, other):
        """Subtraction: (A + B) - C"""
        if isinstance(other, (TensorProduct, YawOperator, TensorSum)):
            neg_other = other * (-1)
            return self + neg_other
        else:
            raise TypeError(f"Cannot subtract {type(other)} from TensorSum")
    
    def __rsub__(self, other):
        """Right subtraction: A - (B + C)"""
        if isinstance(other, (TensorProduct, YawOperator)):
            neg_self = self * (-1)
            return other + neg_self
        else:
            raise TypeError(f"Cannot subtract TensorSum from {type(other)}")
    
    def __neg__(self):
        """Negation: -(A + B) = -A + -B"""
        return TensorSum([term * (-1) for term in self.terms])

    def adjoint(self):
        """Hermitian conjugate of sum: (A + B)† = A† + B†
        
        Distributes adjoint over each term in the sum.
        
        Example:
            >>> (X@I + Z@I).adjoint()  # Returns X†@I† + Z†@I†
            >>> (H@H + X@Z).d  # Same using shortcut
        """
        adjoint_terms = []
        for term in self.terms:
            if hasattr(term, 'adjoint'):
                adj = term.adjoint()
                # Normalize each term if possible
                if hasattr(adj, 'normalize'):
                    adj = adj.normalize()
                adjoint_terms.append(adj)
            else:
                # If term doesn't have adjoint, keep it as is
                adjoint_terms.append(term)
        
        return TensorSum(adjoint_terms)
    
    @property
    def H(self):
        """Hermitian conjugate shortcut: (A + B).H
        
        Example:
            >>> (X@I + Z@I).H  # Returns X†@I† + Z†@I†
        """
        return self.adjoint()
    
    @property
    def dag(self):
        """Dagger shortcut: (A + B).dag
        
        Example:
            >>> (X@I + Z@I).dag  # Returns X†@I† + Z†@I†
        """
        return self.adjoint()
    
    @property
    def d(self):
        """Dagger shortcut: (A + B).d (most concise)
        
        This is the recommended shortcut for adjoint/dagger operation.
        
        Example:
            >>> (X@I + Z@I).d  # Returns X†@I† + Z†@I†
        """
        return self.adjoint()
    
    def lmul(self, state):
        """Left multiplication: Create functional φ where φ(B) = ψ(AB)
        
        This allows building coherent superpositions algebraically.
        
        Args:
            state: Base functional ψ
            
        Returns:
            LeftMultipliedState instance
            
        Example:
            >>> psi_00 = char(Z, 0) @ char(Z, 0)
            >>> X_X = X @ X
            >>> phi = X_X.lmul(psi_00)
            >>> # Now phi(A) computes ⟨00|(X⊗X)A|00⟩ = ⟨11|A|00⟩
        """
        return LeftMultipliedState(self, state)
    
    def rmul(self, state):
        """Right multiplication: Create functional φ where φ(B) = ψ(BA)
        
        Args:
            state: Base functional ψ
            
        Returns:
            RightMultipliedState instance
            
        Example:
            >>> psi_00 = char(Z, 0) @ char(Z, 0)
            >>> X_X = X @ X
            >>> phi = X_X.rmul(psi_00)
            >>> # Now phi(A) computes ⟨00|A(X⊗X)|00⟩ = ⟨00|A|11⟩
        """
        return RightMultipliedState(self, state)

    def __mod__(self, state):
        """Left multiplication using % operator: (A + B) % φ
        
        Syntactic sugar for (A + B).lmul(φ).
        
        Example:
            >>> phi = (X@I + Z@I) % psi_00
        """
        return self.lmul(state)
    
    def __floordiv__(self, state):
        """Right multiplication using // operator: (A + B) // φ
        
        Syntactic sugar for (A + B).rmul(φ).
        
        Example:
            >>> phi = (X@I + Z@I) // psi_00
        """
        return self.rmul(state)

    def normalize(self, verbose=False):
        """Normalize and simplify the sum by combining like terms."""
        from collections import defaultdict
        from sympy import Mul, Symbol

        # Step 1: Normalize each term
        normalized_terms = []
        for term in self.terms:
            if hasattr(term, 'normalize'):
                normalized_terms.append(term.normalize(verbose=verbose))
            else:
                normalized_terms.append(term)

        # Step 2: Extract coefficients and canonical structures
        canonical_terms = []

        for term in normalized_terms:
            if isinstance(term, TensorProduct):
                # Extract coefficient from entire tensor product
                total_coeff = 1
                canonical_factors = []

                for factor in term.factors:
                    # *** CRITICAL FIX: Preserve Projector type ***
                    if isinstance(factor, Projector):
                        # Projectors are already canonical - don't extract coefficients
                        canonical_factors.append(factor)
                    elif isinstance(factor, YawOperator):
                        # Extract coefficient from this factor
                        expr = factor._expr

                        if isinstance(expr, Mul):
                            factor_coeff = 1
                            non_coeff_parts = []

                            for arg in expr.args:
                                if arg.is_number:
                                    factor_coeff *= arg
                                else:
                                    non_coeff_parts.append(arg)

                            total_coeff *= factor_coeff

                            if non_coeff_parts:
                                from sympy import Mul as SympyMul
                                canonical_factors.append(
                                    YawOperator(SympyMul(*non_coeff_parts), factor.algebra)
                                )
                            else:
                                # Pure scalar factor - represents identity
                                # *** FIXED: Extract SymPy expression from I ***
                                I_expr = factor.algebra.I._expr if hasattr(factor.algebra.I, '_expr') else Symbol('I')
                                canonical_factors.append(
                                    YawOperator(I_expr, factor.algebra)
                                )
                        elif expr.is_number:
                            # Pure number
                            total_coeff *= expr
                            # Treat as identity
                            I_expr = factor.algebra.I._expr if hasattr(factor.algebra.I, '_expr') else Symbol('I')
                            canonical_factors.append(
                                YawOperator(I_expr, factor.algebra)
                            )
                        else:
                            # No coefficient
                            canonical_factors.append(factor)
                    else:
                        canonical_factors.append(factor)

                # Create canonical structure
                if canonical_factors:
                    canonical_structure = TensorProduct(canonical_factors)
                else:
                    canonical_structure = None

                canonical_terms.append((total_coeff, canonical_structure))

            elif isinstance(term, YawOperator):
                # Single operator
                expr = term._expr

                if isinstance(expr, Mul):
                    coeff = 1
                    non_coeff_parts = []

                    for arg in expr.args:
                        if arg.is_number:
                            coeff *= arg
                        else:
                            non_coeff_parts.append(arg)

                    if non_coeff_parts:
                        from sympy import Mul as SympyMul
                        canonical_terms.append((coeff, YawOperator(SympyMul(*non_coeff_parts), term.algebra)))
                    else:
                        canonical_terms.append((coeff, None))
                elif expr.is_number:
                    canonical_terms.append((expr, None))
                else:
                    canonical_terms.append((1, term))
            else:
                canonical_terms.append((1, term))

        # Step 3: Group by canonical structure
        structure_groups = defaultdict(list)

        for coeff, structure in canonical_terms:
            # Create key from structure
            if structure is None:
                key = "scalar"
            elif isinstance(structure, TensorProduct):
                # Key from normalized factor strings
                factor_strs = []
                for f in structure.factors:
                    if isinstance(f, YawOperator):
                        f_norm = f.normalize() if hasattr(f, 'normalize') else f
                        factor_strs.append(str(f_norm._expr))
                    else:
                        factor_strs.append(str(f))
                key = " @ ".join(factor_strs)
            else:
                # YawOperator
                s_norm = structure.normalize() if hasattr(structure, 'normalize') else structure
                key = str(s_norm._expr)

            structure_groups[key].append((coeff, structure))

        # Step 4: Combine coefficients for each structure
        combined = []

        for key, coeff_struct_pairs in structure_groups.items():
            # Sum coefficients
            total_coeff = sum(c for c, _ in coeff_struct_pairs)

            # *** FIXED: Better zero check ***
            try:
                coeff_value = complex(total_coeff)
                if abs(coeff_value) < 1e-10:
                    continue
            except:
                # Can't convert to number, keep it
                pass

            # Get structure
            _, structure = coeff_struct_pairs[0]

            # Create result term
            if structure is None:
                # Pure scalar - skip if we don't have algebra
                continue
            elif abs(complex(total_coeff) - 1) < 1e-10:
                # Coefficient is 1
                combined.append(structure)
            elif abs(complex(total_coeff) + 1) < 1e-10:
                # Coefficient is -1
                combined.append((-1) * structure)
            else:
                # Apply coefficient using * operator (not constructor)
                combined.append(total_coeff * structure)

        # Step 5: Return simplified result
        if len(combined) == 0:
            # Everything cancelled
            return 0
        elif len(combined) == 1:
            return combined[0]
        else:
            # Pass _skip_normalize=True to prevent infinite recursion
            return TensorSum(combined, _skip_normalize=True)

    def flatten(self):
        """Flatten tensor structure by linearity: (A + B) @ C → A @ C + B @ C
        
        Applies flattening to each term and returns a new TensorSum.
        
        Example:
            >>> sum_op = (X @ Y) + (Z @ I)
            >>> flat = sum_op.flatten()  # Each term flattened independently
        """
        flattened_terms = []
        for term in self.terms:
            if hasattr(term, 'flatten'):
                flattened_terms.append(term.flatten())
            else:
                flattened_terms.append(term)
        
        return TensorSum(flattened_terms, _skip_normalize=True)
 
    def __matmul__(self, other):
        """Tensor product: (A + B + ...) @ C

        Distributes: (A + B) @ C = A @ C + B @ C
        """
        if isinstance(other, YawOperator):
            # Check if other is a sum
            if isinstance(other._expr, Add):
                # Double distribution: (A + B) @ (C + D)
                new_terms = []
                for self_term in self.terms:
                    for other_term_expr in other._expr.args:
                        other_term = YawOperator(other_term_expr, other.algebra)
                        new_terms.append(self_term @ other_term)
                return TensorSum(new_terms)
            else:
                # Simple distribution: (A + B) @ C
                terms = [term @ other for term in self.terms]
                return TensorSum(terms)

        elif isinstance(other, TensorProduct):
            # (A + B) @ (C ⊗ D) - distribute left over right
            terms = [term @ other for term in self.terms]
            return TensorSum(terms)

        elif isinstance(other, TensorSum):
            # (A + B) @ (C + D) - full distribution
            new_terms = []
            for self_term in self.terms:
                for other_term in other.terms:
                    new_terms.append(self_term @ other_term)
            return TensorSum(new_terms)

        else:
            raise TypeError(f"Cannot tensor TensorSum with {type(other)}")
        
    def _extract_coefficient(self, tensor_prod):
        """Extract coefficient from a tensor product.
        
        Returns (coefficient, structure) where structure is the tensor
        product without the coefficient.
        """
        if not isinstance(tensor_prod, TensorProduct):
            return (1, tensor_prod)
        
        # Check if first factor has a scalar coefficient
        first_factor = tensor_prod.factors[0]
        
        if isinstance(first_factor, YawOperator):
            expr = first_factor._expr
            
            # Check if it's a scalar multiple
            from sympy import Mul
            if isinstance(expr, Mul):
                coeff = 1
                non_coeff_args = []
                
                for arg in expr.args:
                    if arg.is_number:
                        coeff *= arg
                    else:
                        non_coeff_args.append(arg)
                
                if non_coeff_args:
                    # Reconstruct first factor without coefficient
                    from sympy import Mul as SympyMul
                    new_first = YawOperator(SympyMul(*non_coeff_args), first_factor.algebra)
                    new_factors = [new_first] + tensor_prod.factors[1:]
                    return (coeff, TensorProduct(new_factors))
                else:
                    # Pure scalar - return rest of factors
                    if len(tensor_prod.factors) > 1:
                        return (coeff, TensorProduct(tensor_prod.factors[1:]))
                    else:
                        return (coeff, None)
            
            # No coefficient
            return (1, tensor_prod)
        
        return (1, tensor_prod)
    
    def _extract_coefficient_op(self, op):
        """Extract coefficient from a YawOperator."""
        from sympy import Mul
        
        expr = op._expr
        
        if isinstance(expr, Mul):
            coeff = 1
            non_coeff_args = []
            
            for arg in expr.args:
                if arg.is_number:
                    coeff *= arg
                else:
                    non_coeff_args.append(arg)
            
            if non_coeff_args:
                from sympy import Mul as SympyMul
                return (coeff, YawOperator(SympyMul(*non_coeff_args), op.algebra))
            else:
                return (coeff, None)
        
        return (1, op)
    
    def _get_structure_key(self, structure):
        """Get a string key representing the structure of a term."""
        if structure is None:
            return "scalar"
        
        if isinstance(structure, TensorProduct):
            # Create key from normalized factors
            factor_keys = []
            for factor in structure.factors:
                if isinstance(factor, YawOperator):
                    # Normalize and get string
                    norm_factor = factor.normalize() if hasattr(factor, 'normalize') else factor
                    factor_keys.append(str(norm_factor._expr))
                else:
                    factor_keys.append(str(factor))
            
            return " @ ".join(factor_keys)
        
        elif isinstance(structure, YawOperator):
            norm = structure.normalize() if hasattr(structure, 'normalize') else structure
            return str(norm._expr)
        
        return str(structure)

class TensorProduct:
    """Tensor product of operators: A ⊗ B ⊗ C"""
    
    def __init__(self, factors):
        """Create tensor product.
        
        Args:
            factors: List of YawOperator instances
        """
        # Ensure factors is a list
        if not isinstance(factors, list):
            factors = list(factors)
        
        self.factors = factors
        
        if not self.factors:
            raise ValueError("TensorProduct requires at least one factor")
        
        # *** FIX: Don't try to access algebra on the list ***
        # Infer algebra from first factor (if any have algebra)
        self.algebra = None
        for factor in self.factors:
            if isinstance(factor, YawOperator) and hasattr(factor, 'algebra') and factor.algebra is not None:
                self.algebra = factor.algebra
                break

    def __invert__(self):
        """Enable ~A syntax for normalization."""
        return self.normalize()
            
    def __lshift__(self, state):
        """Apply tensor product to state: (A⊗B) << φ
        
        Implements: ((A⊗B) << φ)(C) = φ((A⊗B)† C (A⊗B))
        
        This is computed algebraically as: (A⊗B) << φ = (A⊗B)† % ((A⊗B) // φ)
        
        Args:
            state: State functional to transform
            
        Returns:
            Transformed state functional
        """
        # Algebraic implementation: (A⊗B) << φ = (A⊗B)† % ((A⊗B) // φ)
        return self.adjoint() % (self // state)
            
    def __matmul__(self, other):
        """Extend tensor product: (A ⊗ B) @ C = A ⊗ B ⊗ C

        Distributes over sums: (A ⊗ B) @ (C + D) = A ⊗ B ⊗ C + A ⊗ B ⊗ D
        """
        from sympy import Add

        if isinstance(other, YawOperator):
            # Check if other is a sum
            if isinstance(other._expr, Add):
                # Distribute
                terms = []
                for term in other._expr.args:
                    term_op = YawOperator(term, other.algebra)
                    extended = TensorProduct(self.factors + [term_op])
                    terms.append(extended)

                return TensorSum(terms)
            else:
                return TensorProduct(self.factors + [other])

        elif isinstance(other, TensorProduct):
            return TensorProduct(self.factors + other.factors)

        elif isinstance(other, TensorSum):
            # (A ⊗ B) @ (C + D + ...) 
            distributed_terms = [self @ term for term in other.terms]
            return TensorSum(distributed_terms)

        else:
            raise TypeError(f"Cannot tensor TensorProduct with {type(other)}")
    
    def __add__(self, other):
        """Addition creates a sum of tensors."""
        if isinstance(other, TensorProduct):
            return TensorSum([self, other])
        elif isinstance(other, TensorSum):
            return TensorSum([self] + other.terms)
        elif isinstance(other, YawOperator):
            return TensorSum([self, other])
        else:
            raise TypeError(f"Cannot add TensorProduct with {type(other)}")
    
    def __radd__(self, other):
        """Right addition."""
        if isinstance(other, (YawOperator, TensorProduct)):
            return TensorSum([other, self])
        elif isinstance(other, TensorSum):
            return TensorSum(other.terms + [self])
        else:
            raise TypeError(f"Cannot add {type(other)} with TensorProduct")
    
    def __truediv__(self, other):
        """Division by scalar."""
        if isinstance(other, (int, float, complex)) or (hasattr(other, 'is_number') and other.is_number):
            new_factors = self.factors.copy()
            new_factors[0] = new_factors[0] / other
            return TensorProduct(new_factors)
        else:
            raise TypeError(f"Cannot divide TensorProduct by {type(other)}")
    
    def adjoint(self):
        """Hermitian conjugate of tensor product: (A⊗B)† = A†⊗B†
        
        Distributes adjoint over each factor and automatically normalizes.
        
        Example:
            >>> (X@Y).adjoint()  # Returns X†⊗Y† (normalized)
            >>> (H@H).H  # Same as (H@H).adjoint()
        """
        adjoint_factors = []
        for factor in self.factors:
            if hasattr(factor, 'adjoint'):
                adj = factor.adjoint()
                # Normalize each factor
                if hasattr(adj, 'normalize'):
                    adj = adj.normalize()
                adjoint_factors.append(adj)
            else:
                # If factor doesn't have adjoint, keep it as is
                adjoint_factors.append(factor)
        
        return TensorProduct(adjoint_factors)
    
    @property
    def H(self):
        """Hermitian conjugate shortcut: (A⊗B).H
        
        Example:
            >>> (X@Y).H  # Returns X†⊗Y†
        """
        return self.adjoint()
    
    @property
    def dag(self):
        """Dagger shortcut: (A⊗B).dag
        
        Example:
            >>> (X@Y).dag  # Returns X†⊗Y†
        """
        return self.adjoint()
    
    @property
    def d(self):
        """Dagger shortcut: (A⊗B).d (most concise)
        
        Example:
            >>> (X@Y).d  # Returns X†⊗Y†
        """
        return self.adjoint()
    
    def lmul(self, state):
        """Left multiplication: Create functional φ where φ(B) = ψ(AB)
        
        This allows building coherent superpositions algebraically.
        
        Args:
            state: Base functional ψ
            
        Returns:
            LeftMultipliedState instance
            
        Example:
            >>> psi_00 = char(Z, 0) @ char(Z, 0)
            >>> X_X = X @ X
            >>> phi = X_X.lmul(psi_00)
            >>> # Now phi(A) computes ⟨00|(X⊗X)A|00⟩ = ⟨11|A|00⟩
        """
        return LeftMultipliedState(self, state)
    
    def rmul(self, state):
        """Right multiplication: Create functional φ where φ(B) = ψ(BA)
        
        Args:
            state: Base functional ψ
            
        Returns:
            RightMultipliedState instance
            
        Example:
            >>> psi_00 = char(Z, 0) @ char(Z, 0)
            >>> X_X = X @ X
            >>> phi = X_X.rmul(psi_00)
            >>> # Now phi(A) computes ⟨00|A(X⊗X)|00⟩ = ⟨00|A|11⟩
        """
        return RightMultipliedState(self, state)
    
    def __mod__(self, state):
        """Left multiplication using % operator: (A⊗B) % φ
        
        Syntactic sugar for (A⊗B).lmul(φ).
        
        Example:
            >>> phi = X_X % psi_00  # Same as X_X.lmul(psi_00)
        """
        return self.lmul(state)
    
    def __floordiv__(self, state):
        """Right multiplication using // operator: (A⊗B) // φ
        
        Syntactic sugar for (A⊗B).rmul(φ).
        
        Example:
            >>> phi = X_X // psi_00  # Same as X_X.rmul(psi_00)
        """
        return self.rmul(state)
    
    @property
    def T(self):
        """Transpose shortcut: (A⊗B).T
        
        Example:
            >>> (X@Y).T  # Returns X†⊗Y†
        """
        return self.adjoint()
    
    def normalize(self, verbose=False):
        """Normalize each factor separately."""
        normalized_factors = []
        for factor in self.factors:
            if isinstance(factor, YawOperator) and hasattr(factor, 'algebra') and factor.algebra:
                normalized_factors.append(factor.normalize(verbose=verbose))
            else:
                normalized_factors.append(factor)
        return TensorProduct(normalized_factors)
    
    def flatten(self):
        """Flatten nested tensor products: (A @ B) @ C → A @ B @ C
        
        Returns a new TensorProduct with all nested TensorProducts expanded
        into a flat list of factors.
        
        Example:
            >>> A = X @ Y
            >>> B = Z @ I  
            >>> nested = A @ B  # 2 factors: [A, B]
            >>> flat = nested.flatten()  # 4 factors: [X, Y, Z, I]
        """
        flat_factors = []
        for factor in self.factors:
            if isinstance(factor, TensorProduct):
                # Recursively flatten nested tensor products
                nested_flat = factor.flatten()
                flat_factors.extend(nested_flat.factors)
            else:
                flat_factors.append(factor)
        
        return TensorProduct(flat_factors)
    
    def __rshift__(self, other):
        """Operator conjugation for tensor products.
        
        Implements: (U⊗V) >> (A⊗B) = (U†AU)⊗(V†BV)
        
        This distributes conjugation across factors:
        - Each U_i conjugates the corresponding A_i
        - Preserves tensor product structure
        - Automatically normalizes each factor
        
        Args:
            other: Operator to conjugate (YawOperator, TensorProduct, or TensorSum)
            
        Returns:
            Conjugated operator (automatically normalized)
            
        Example:
            >>> (H@H) >> (X@X)  # Returns Z@Z (auto-normalized)
            >>> (H@I) >> (X@Y)  # Returns Z@Y (auto-normalized)
        
        Note:
            This only works when self and other have compatible tensor structure.
            For non-tensor operators, falls back to standard conjugation.
        """
        # Case 1: Conjugating a TensorProduct
        if isinstance(other, TensorProduct):
            if len(self.factors) != len(other.factors):
                raise ValueError(
                    f"Cannot conjugate: tensor products have different lengths "
                    f"({len(self.factors)} vs {len(other.factors)})"
                )
            
            # Apply conjugation factor-by-factor: (U_i >> A_i)
            conjugated_factors = []
            for u_factor, a_factor in zip(self.factors, other.factors):
                # Each factor conjugates: u_factor >> a_factor = u_factor† * a_factor * u_factor
                if hasattr(u_factor, 'conj_op'):
                    conjugated = u_factor.conj_op(a_factor)
                elif hasattr(u_factor, '__rshift__'):
                    conjugated = u_factor >> a_factor
                else:
                    # Manual conjugation: Uâ€  A U
                    u_adj = u_factor.adjoint() if hasattr(u_factor, 'adjoint') else u_factor
                    conjugated = u_adj * a_factor * u_factor
                
                # Normalize each factor before adding to result
                if hasattr(conjugated, 'normalize'):
                    conjugated = conjugated.normalize()
                
                conjugated_factors.append(conjugated)
            
            return TensorProduct(conjugated_factors)
        
        # Case 2: Conjugating a YawOperator
        elif isinstance(other, YawOperator):
            # For single operators, we need to interpret this as acting on first factor
            # This is ambiguous - for now, raise error
            raise ValueError(
                "Cannot conjugate single operator with tensor product. "
                "Convert to tensor product first (e.g., A@I)"
            )
        
        # Case 3: Conjugating a TensorSum
        elif isinstance(other, TensorSum):
            # Distribute conjugation over sum: U >> (A + B) = (U >> A) + (U >> B)
            conjugated_terms = []
            for term in other.terms:
                conjugated_terms.append(self >> term)
            
            return TensorSum(conjugated_terms)
        
        else:
            raise TypeError(f"Cannot conjugate TensorProduct with {type(other)}")
    
    def conj_op(self, other):
        """Operator conjugation: self >> other
        
        This is the method version of __rshift__.
        Implements (U⊗V) >> (A⊗B) = (U†AU)⊗(V†BV)
        
        Args:
            other: Operator to conjugate
            
        Returns:
            Conjugated operator
        """
        return self >> other

    def __mul__(self, other):
        """Multiplication of tensor products or scalars.
        
        For scalars: c * (A ⊗ B) = (cA) ⊗ B
        For tensor products: (A⊗B) * (C⊗D) = (A*C) ⊗ (B*D)
        For tensor sums: (A⊗B) * (C + D) = (A⊗B)*C + (A⊗B)*D (distributivity)
        For single operators: Promote to tensor product if needed
        """
        # Scalar multiplication
        if isinstance(other, (int, float, complex)):
            new_factors = self.factors.copy()
            new_factors[0] = new_factors[0] * other
            return TensorProduct(new_factors)
        
        # Check for SymPy numbers
        if hasattr(other, 'is_number') and other.is_number:
            new_factors = self.factors.copy()
            new_factors[0] = new_factors[0] * other
            return TensorProduct(new_factors)
        
        # TensorSum: distribute multiplication
        elif isinstance(other, TensorSum):
            # (A⊗B) * (C + D) = (A⊗B)*C + (A⊗B)*D
            return TensorSum([self * term for term in other.terms])
        
        # Tensor product multiplication (element-wise)
        elif isinstance(other, TensorProduct):
            if len(self.factors) != len(other.factors):
                raise ValueError(
                    f"Cannot multiply tensor products of different lengths: "
                    f"{len(self.factors)} vs {len(other.factors)}"
                )
            
            # Element-wise multiplication: (A⊗B) * (C⊗D) = (A*C) ⊗ (B*D)
            new_factors = []
            for f1, f2 in zip(self.factors, other.factors):
                product = f1 * f2
                # Normalize each factor
                if hasattr(product, 'normalize'):
                    product = product.normalize()
                new_factors.append(product)
            
            return TensorProduct(new_factors)
        
        # Single YawOperator: treat as element-wise multiplication on first factor
        elif isinstance(other, YawOperator):
            if len(self.factors) == 1:
                # Single factor case
                return TensorProduct([self.factors[0] * other])
            else:
                # Multi-factor: multiply first factor only
                new_factors = self.factors.copy()
                new_factors[0] = new_factors[0] * other
                return TensorProduct(new_factors)
        
        else:
            raise TypeError(f"Cannot multiply TensorProduct with {type(other)}")
    
    def __rmul__(self, other):
        """Right multiplication: c * (A ⊗ B)
        
        This is called when the left operand doesn't know how to multiply
        with TensorProduct (e.g., int * TensorProduct).
        """
        # *** FIXED: Handle scalar on the left ***
        if isinstance(other, (int, float, complex)):
            new_factors = self.factors.copy()
            new_factors[0] = other * new_factors[0]
            return TensorProduct(new_factors)
        
        # Check for SymPy numbers
        if hasattr(other, 'is_number') and other.is_number:
            new_factors = self.factors.copy()
            new_factors[0] = other * new_factors[0]
            return TensorProduct(new_factors)
        
        # For other types, try regular multiplication
        return self.__mul__(other)
    
    def __sub__(self, other):
        """Subtraction: A - B"""
        # *** FIXED: More explicit approach ***
        if isinstance(other, TensorProduct):
            neg_other = other * (-1)
            return self + neg_other
        elif isinstance(other, YawOperator):
            neg_other = other * (-1)
            return self + neg_other
        elif isinstance(other, TensorSum):
            neg_other = other * (-1)
            return self + neg_other
        else:
            raise TypeError(f"Cannot subtract {type(other)} from TensorProduct")
    
    def __rsub__(self, other):
        """Right subtraction: B - self"""
        if isinstance(other, (TensorProduct, YawOperator, TensorSum)):
            neg_self = self * (-1)
            return other + neg_self
        else:
            raise TypeError(f"Cannot subtract TensorProduct from {type(other)}")
    
    def __neg__(self):
        """Negation: -A = (-1) * A"""
        return self * (-1)
    
    def __str__(self):
        factor_strs = [str(f) for f in self.factors]
        return " @ ".join(factor_strs)
    
    def __repr__(self):
        # *** FIXED: Return string directly ***
        return self.__str__()

# ============================================================================
# ALGEBRA
# ============================================================================

class Algebra:
    """Algebraic presentation <G|R> with generators and relations.
    
    Manages a non-commutative algebra defined by:
    - Generators: Named operators (e.g., X, Z for Pauli algebra)
    - Relations: Rules like hermiticity, unitarity, anticommutation
    
    Provides symbolic normalization based on rewrite rules compiled
    from the relations.
    
    Attributes:
        generator_names: List of generator names
        generators: Dict mapping names to YawOperator instances
        relation_specs: List of relation specifiers
        rules: Compiled rewrite rules
        I: Identity operator
    """
    
    def __init__(self, gens: List[str], rels: List[str]):
        """Create algebra from generators and relations.
        
        Args:
            gens: List of generator names (e.g., ['X', 'Z'])
            rels: List of relation specifiers (e.g., ['herm', 'unit', 'anti'])
        """
        self.generator_names = gens
        self.relation_specs = rels
        
        # Create SymPy operators
        self.generators = {}
        for name in gens:
            sympy_op = Operator(name)
            self.generators[name] = YawOperator(sympy_op, self)
        
        # Identity operator
        self.I = YawOperator(Operator('I'), self)
        
        # Compile relations to rewrite rules
        self.rules = self._compile_relations(rels)
    
    def __getattr__(self, name):
        """Allow convenient access: algebra.X instead of algebra.generators['X']"""
        if name in self.generators:
            return self.generators[name]
        raise AttributeError(f"Generator {name} not in algebra")
    
    def _compile_relations(self, rels: List[str]) -> List[Tuple]:
        """Convert relation specifiers to SymPy rewrite rules."""
        import re

        rules = []
        sympy_gens = {name: self.generators[name]._expr 
                      for name in self.generator_names}
        sympy_I = self.I._expr

        # Store braiding phase (default to None)
        self.braid_phase = None

        # *** NEW: Store power modulus (default to None) ***
        self.power_mod = None

        # Track if we have herm and unit (implies pow(2))
        has_herm = 'herm' in rels
        has_unit = 'unit' in rels

        for rel in rels:
            if rel == 'herm':
                # Hermitian: X† = X
                for name, op_expr in sympy_gens.items():
                    rules.append((Dagger(op_expr), op_expr))

            elif rel == 'unit':
                # Unitary: X†X = I
                for name, op_expr in sympy_gens.items():
                    rules.append((op_expr * Dagger(op_expr), sympy_I))
                    rules.append((Dagger(op_expr) * op_expr, sympy_I))

            elif rel == 'anti':
                # Anticommutation: XY = -YX (special case of braid(-1))
                self.braid_phase = -1

            elif rel.startswith('braid('):
                # Braiding relation: XY = ω YX
                match = re.match(r'braid\((.*)\)', rel)
                if match:
                    phase_expr = match.group(1)
                    # Evaluate the phase expression
                    from sympy import sympify, exp, pi, I as sympy_I_const
                    try:
                        # Create namespace for evaluation
                        phase_namespace = {
                            'exp': exp,
                            'pi': pi,
                            'I': sympy_I_const,
                            'i': sympy_I_const,
                        }
                        self.braid_phase = sympify(phase_expr, locals=phase_namespace)
                    except Exception as e:
                        print(f"Warning: Could not parse braid phase '{phase_expr}': {e}")
                        self.braid_phase = sympify(phase_expr)

            elif rel.startswith('pow('):
                # Power relation: X^n = I
                match = re.match(r'pow\((\d+)\)', rel)
                if match:
                    n = int(match.group(1))
                    self.power_mod = n  # *** Store the modulus ***
                    for name, op_expr in sympy_gens.items():
                        rules.append((op_expr**n, sympy_I))

        # Derive pow(2) from herm + unit
        has_explicit_pow = any(rel.startswith('pow(') for rel in rels)
        if has_herm and has_unit and not has_explicit_pow:
            # Hermitian + Unitary ⟹ X₂ = I
            self.power_mod = 2  # *** Store derived modulus ***
            for name, op_expr in sympy_gens.items():
                rules.append((op_expr**2, sympy_I))

        # Add identity simplification rules
        for name, op_expr in sympy_gens.items():
            rules.append((op_expr * sympy_I, op_expr))
            rules.append((sympy_I * op_expr, op_expr))

        rules.append((Dagger(sympy_I), sympy_I))
            
        # Add identity power rules
        max_ident_power = 100
        for n in range(2, max_ident_power):
            rules.append((sympy_I**n, sympy_I))

        return rules
    
    def _simplify_identity(self, expr):
        """Remove identity operators from products."""
        sympy_I = self.I._expr
        
        if isinstance(expr, Mul):
            args = list(expr.args)
            new_args = [arg for arg in args if arg != sympy_I]
            
            if not new_args:
                return sympy_I
            elif len(new_args) == 1:
                return new_args[0]
            else:
                return Mul(*new_args)
        
        elif isinstance(expr, Add):
            return Add(*[self._simplify_identity(term) for term in expr.args])
        
        return expr
    
    def _simplify_projector_conjugates(self, expr):
        """Simplify projector expressions using projector algebra.
        
        Rules:
        - conjugate(proj(...)) → proj(...) (self-adjoint)
        - proj(A, k)^n → proj(A, k) for n ≥ 1 (idempotence)
        - proj(A, k) * proj(A, j) → 0 if k ≠ j (orthogonality)
        - proj(A, k) * proj(A, k) → proj(A, k) (idempotence)
        """
        from sympy import conjugate, Pow, Mul, Symbol
        
        def is_projector(expr):
            """Check if expression is a projector symbol."""
            return hasattr(expr, 'name') and str(expr).startswith('proj(')
        
        def parse_projector(proj_expr):
            """Parse proj(op, k) to extract operator and index.
            Returns (operator_str, index) or None if not a projector.
            """
            if not is_projector(proj_expr):
                return None
            
            proj_str = str(proj_expr)
            # Format is "proj(op, k)" - extract op and k
            try:
                # Remove "proj(" prefix and ")" suffix
                inner = proj_str[5:-1]  # "op, k"
                parts = inner.split(', ')
                if len(parts) == 2:
                    return (parts[0], int(parts[1]))
            except:
                pass
            return None
        
        def simplify_term(term):
            # Handle conjugate(proj(...))
            if isinstance(term, conjugate):
                arg = term.args[0]
                if is_projector(arg):
                    return arg  # Remove conjugate (self-adjoint)
                # Recursively simplify
                simplified_arg = simplify_term(arg)
                if simplified_arg == arg:
                    return term
                else:
                    return conjugate(simplified_arg)
            
            # Handle proj(...)^n for n ≥ 1 (idempotence)
            elif isinstance(term, Pow):
                base = term.base
                exp = term.exp
                
                if is_projector(base):
                    # Check if exponent is a positive integer (works with both Python int and SymPy Integer)
                    try:
                        exp_val = int(exp)
                        if exp_val >= 1:
                            return base  # proj^n = proj for n ≥ 1
                        elif exp_val == 0:
                            return self.I._expr  # proj^0 = I
                    except (TypeError, ValueError):
                        pass  # Not an integer exponent
                
                # Recursively simplify base
                simplified_base = simplify_term(base)
                if simplified_base != base:
                    return Pow(simplified_base, exp)
                return term
            
            # Handle products - this is the critical case!
            elif isinstance(term, Mul):
                args = list(term.args)
                
                # Separate coefficients from projectors/operators
                coeff = 1
                proj_factors = []
                other_factors = []
                
                for arg in args:
                    if isinstance(arg, (int, float, complex)) or (hasattr(arg, 'is_number') and arg.is_number):
                        coeff *= arg
                    elif is_projector(arg):
                        proj_factors.append(arg)
                    elif isinstance(arg, Pow) and is_projector(arg.base):
                        # Handle proj^n in products
                        try:
                            exp_val = int(arg.exp)
                            if exp_val >= 1:
                                proj_factors.append(arg.base)  # Reduce proj^n to proj
                            elif exp_val == 0:
                                coeff *= 1  # proj^0 = I
                            else:
                                proj_factors.append(arg)
                        except (TypeError, ValueError):
                            proj_factors.append(arg)  # Non-integer exponent
                    else:
                        other_factors.append(arg)
                
                # Simplify consecutive projectors: proj(A,k) * proj(A,j)
                simplified_projs = []
                i = 0
                while i < len(proj_factors):
                    current_proj = proj_factors[i]
                    current_info = parse_projector(current_proj)
                    
                    if current_info is None:
                        simplified_projs.append(current_proj)
                        i += 1
                        continue
                    
                    current_op, current_idx = current_info
                    
                    # Look ahead for consecutive projectors of the same operator
                    j = i + 1
                    absorbed = False
                    while j < len(proj_factors):
                        next_proj = proj_factors[j]
                        next_info = parse_projector(next_proj)
                        
                        if next_info is None:
                            break
                        
                        next_op, next_idx = next_info
                        
                        # Check if they're projectors of the same operator
                        if current_op == next_op:
                            if current_idx == next_idx:
                                # proj(A, k) * proj(A, k) = proj(A, k)
                                # Keep current, skip next
                                proj_factors.pop(j)
                                absorbed = True
                                # Don't increment j, check the new element at position j
                            else:
                                # proj(A, k) * proj(A, j) = 0 for k ≠ j
                                return 0
                        else:
                            break
                    
                    if not absorbed:
                        simplified_projs.append(current_proj)
                    i += 1
                
                # Reconstruct
                all_factors = []
                if coeff != 1:
                    all_factors.append(coeff)
                all_factors.extend(simplified_projs)
                all_factors.extend([simplify_term(f) for f in other_factors])
                
                if not all_factors:
                    return 1
                elif len(all_factors) == 1:
                    return all_factors[0]
                else:
                    return Mul(*all_factors)
            
            # Handle sums recursively
            elif isinstance(term, Add):
                new_args = [simplify_term(arg) for arg in term.args]
                return Add(*new_args)
            
            return term
        
        return simplify_term(expr)
    
    def _reduce_powers(self, expr):
        """Reduce higher powers according to pow(n) relations.

        If generators satisfy X^n = I, then X^m = X^(m mod n).
        """
        if self.power_mod is None:
            return expr

        gen_exprs = [self.generators[name]._expr for name in self.generator_names]
        sympy_I = self.I._expr

        def reduce_term(term):
            if isinstance(term, Pow):
                # Identity powers: I^n = I
                if term.base == sympy_I and term.exp > 1:
                    return sympy_I
                # Generator powers: reduce modulo power_mod
                if term.base in gen_exprs and term.exp >= self.power_mod:
                    new_exp = term.exp % self.power_mod
                    return sympy_I if new_exp == 0 else term.base**new_exp
            elif isinstance(term, Mul):
                new_args = [reduce_term(arg) for arg in term.args]
                return Mul(*new_args)
            return term

        if isinstance(expr, Add):
            return Add(*[reduce_term(term) for term in expr.args])
        else:
            return reduce_term(expr)
    
    def _apply_braiding(self, expr):
        """Apply braiding relations: XY = ω YX

        Uses bubble sort to put operators in canonical order,
        accumulating the braiding phase for each swap.

        For anticommutation: ω = -1 (XY = -YX)
        For general braiding: ω = exp(2πi/d) or other phase
        """
        # Only apply if we have a braiding phase
        if self.braid_phase is None:
            return expr

        gen_exprs = [self.generators[name]._expr for name in self.generator_names]

        if isinstance(expr, Mul):
            args = list(expr.args)

            # Separate coefficient from operators
            coeff = 1
            ops = []
            for arg in args:
                if isinstance(arg, SympyNumber) or arg.is_number:
                    coeff *= arg
                else:
                    ops.append(arg)

            # Bubble sort into canonical order, accumulating phase
            changed = True
            while changed:
                changed = False
                for i in range(len(ops) - 1):
                    if ops[i] in gen_exprs and ops[i+1] in gen_exprs:
                        idx_i = gen_exprs.index(ops[i])
                        idx_j = gen_exprs.index(ops[i+1])

                        if idx_i > idx_j:  # Wrong order - need to swap
                            ops[i], ops[i+1] = ops[i+1], ops[i]
                            coeff *= self.braid_phase  # Accumulate phase
                            changed = True
                            break

            # Reconstruct
            if ops:
                return _clean_number(coeff) * Mul(*ops) if coeff != 1 else Mul(*ops)
            else:
                return _clean_number(coeff)

        elif isinstance(expr, Add):
            return Add(*[self._apply_braiding(term) for term in expr.args])

        return expr
    
    def normalize(self, yaw_op: YawOperator, verbose=False) -> YawOperator:
        """Apply rewrite rules until convergence."""
        expr = expand(yaw_op._expr)

        if verbose:
            print(f"\n{'='*60}")
            print(f"NORMALIZING: {expr}")
            print(f"{'='*60}")

        max_iterations = 50
        for iteration in range(max_iterations):
            old_expr = expr

            # Apply standard rewrite rules
            for rule in self.rules:
                if isinstance(rule, tuple) and len(rule) == 2:
                    pattern, replacement = rule
                    new_expr = expr.replace(pattern, replacement)
                    if new_expr != expr:
                        if verbose:
                            print(f"  → Rule applied: {expr} ↦ {new_expr}")
                        expr = new_expr

            # Reduce higher powers
            new_expr = self._reduce_powers(expr)
            if new_expr != expr:
                if verbose:
                    print(f"  → Powers reduced: {expr} ↦ {new_expr}")
                expr = new_expr

            # Apply braiding (replaces old anticommutation)
            if self.braid_phase is not None:
                new_expr = self._apply_braiding(expr)
                if new_expr != expr:
                    if verbose:
                        phase_str = str(self.braid_phase)
                        print(f"  • Braiding applied (ω={phase_str}): {expr}")
                    expr = new_expr

            # Simplify identities
            new_expr = self._simplify_identity(expr)
            if new_expr != expr:
                if verbose:
                    print(f"  → Identity simplified: {expr} ↦ {new_expr}")
                expr = new_expr
            
            # Simplify conjugate(proj(...)) → proj(...) since projectors are self-adjoint
            new_expr = self._simplify_projector_conjugates(expr)
            if new_expr != expr:
                if verbose:
                    print(f"  → Projector conjugates simplified: {expr} ↦ {new_expr}")
                expr = new_expr

            expr = expand(expr)

            if expr == old_expr:
                if verbose:
                    print(f"  • Converged")
                break

        if verbose:
            print(f"FINAL: {expr}")
            print(f"{'='*60}\n")

        return YawOperator(expr, self)

    def __str__(self):
        """Human-readable string representation."""
        gens = ', '.join(self.generators.keys())

        rels = []
        # *** FIXED: Use hasattr to safely check attributes ***
        if hasattr(self, 'hermitian') and self.hermitian:
            rels.append('herm')
        if hasattr(self, 'unitary') and self.unitary:
            rels.append('unit')
        if hasattr(self, 'power_mod') and self.power_mod:
            rels.append(f'pow({self.power_mod})')
        if hasattr(self, 'anticommuting'):
            if isinstance(self.anticommuting, list):
                rels.append('anti')
            elif self.anticommuting:
                rels.append('anti')
        if hasattr(self, 'braiding_phase') and self.braiding_phase is not None:
            rels.append(f'braid({self.braiding_phase})')

        rels_str = ', '.join(rels) if rels else '...'
        return f"<{gens} | {rels_str}>"

    def __repr__(self):
        """Detailed representation."""
        return self.__str__()

def qudit(d = 2):
    """Create d-level qudit algebra.
    
    For Pd-level systems, generators satisfy:
    - X^d = Z^d = I (power relation)
    - XZ = ω ZX where ω = exp(2πi/d) (braiding)
    
    Args:
        d: Dimension of the qudit (d=2 for qubit, d=3 for qutrit, etc.)
        
    Returns:
        Algebra instance configured for d-level system
    """
    from sympy import pi, I as sympy_I, exp
    
    # Compute braiding phase
    omega = exp(2*pi*sympy_I/d)

    if d == 2:
        # Qubits: use anticommutation
        return Algebra(
            gens=['X', 'Z'],
            rels=['herm', 'unit', 'anti']
        )
    else:
        return Algebra(
            gens=['X', 'Z'],
            rels=['herm', 'unit', f'pow({d})', f'braid(exp(2*pi*I/{d}))']
        )

def qubit():
    return qudit(2)
    
# ============================================================================
# STATES
# ============================================================================

class _SimpleState:
    """Minimal state for spectrum computation, avoiding circular dependencies.
    
    This is an internal helper class used by EigenState.__init__ to compute
    spectra without triggering infinite recursion through char().
    
    It provides a simple eigenstate of a given observable with eigenvalue +1,
    which is sufficient for GNS matrix construction via gnsMat().
    """
    
    def __init__(self, observable, algebra):
        """Create simple eigenstate with eigenvalue +1.
        
        Args:
            observable: Observable to be eigenstate of
            algebra: Associated algebra
        """
        self.observable = observable
        self.algebra = algebra
        self.eigenvalue = 1.0  # Simple choice: always use +1 eigenvalue
    
    def expect(self, op, _depth=0):
        """Compute expectation value via simple algebraic rules.
        
        Uses the fact that this is an eigenstate with eigenvalue +1.
        """
        if _depth > 10:
            return 0.0
        
        # Normalize operator
        op_norm = self.algebra.normalize(op)
        op_expr = op_norm._expr
        obs_expr = self.observable._expr
        
        # Scalar
        if isinstance(op_expr, SympyNumber) or (hasattr(op_expr, 'is_number') and op_expr.is_number):
            return _clean_number(complex(op_expr))
        
        # Identity
        if str(op_expr) == 'I':
            return 1.0
        
        # The observable itself (eigenvalue = 1)
        if op_expr == obs_expr:
            return _clean_number(self.eigenvalue)
        
        # Power of observable
        if isinstance(op_expr, Pow) and op_expr.base == obs_expr:
            return _clean_number(self.eigenvalue) ** op_expr.exp
        
        # Sum (linearity)
        if isinstance(op_expr, Add):
            total = 0.0
            for term in op_expr.args:
                term_op = YawOperator(term, self.algebra)
                total += self.expect(term_op, _depth=_depth+1)
            return _clean_number(total)
        
        # Product with coefficient
        if isinstance(op_expr, Mul):
            coeff = 1.0
            operator_parts = []
            
            for arg in op_expr.args:
                if isinstance(arg, SympyNumber) or (hasattr(arg, 'is_number') and arg.is_number):
                    coeff *= complex(arg)
                else:
                    operator_parts.append(arg)
            
            if not operator_parts:
                return _clean_number(coeff)
            
            # Reconstruct operator without coefficient
            from sympy import Mul as SympyMul
            op_part = SympyMul(*operator_parts)
            op_part_yaw = YawOperator(op_part, self.algebra)
            
            return coeff * self.expect(op_part_yaw, _depth=_depth+1)
        
        # Default: assume off-diagonal or unknown (return 0)
        return 0.0

class State:
    """Base class for quantum states as functionals."""
    
    def expect(self, operator, _depth=0):
        """Compute expectation value."""
        raise NotImplementedError("Subclasses must implement expect()")
    
    def __rmul__(self, scalar):
        """Scalar multiplication: scalar * state.
        
        Creates a weighted state component for building mixed states.
        
        Args:
            scalar: Probability weight (typically 0 ≤ scalar ≤ 1)
        
        Returns:
            MixedState with single component
        
        Example:
            >>> psi = char(Z, 0)
            >>> weighted = 0.7 * psi  # MixedState([(0.7, psi)])
        """
        return MixedState([(scalar, self)])
    
    def __add__(self, other):
        """Addition: state + state.
        
        Combines states into a mixed state (with automatic normalization).
        
        Args:
            other: Another State or MixedState
        
        Returns:
            MixedState combining both states
        
        Example:
            >>> rho = 0.7*psi0 + 0.3*psi1
            >>> # Equivalent to: mixed([(0.7, psi0), (0.3, psi1)])
        """
        if isinstance(other, MixedState):
            # Pure state + MixedState
            # Convert self to MixedState and add
            return MixedState([(1.0, self)]) + other
        elif isinstance(other, State):
            # Two pure states - combine with equal weight, then normalize
            return MixedState([(1.0, self), (1.0, other)])
        else:
            raise TypeError(f"Cannot add State and {type(other)}")
    
    def __ror__(self, operator):
        """Enable A | state syntax."""
        return self.expect(operator)
    
    def __matmul__(self, other):
        """Tensor product of states: |ψ⟩ @ |φ⟩ = |ψ⟩ ⊗ |φ⟩"""
        if isinstance(other, State):
            # Both are states
            if isinstance(self, TensorState):
                if isinstance(other, TensorState):
                    # Both tensor states: flatten
                    return TensorState(self.states + other.states)
                else:
                    # Self is tensor, other is simple
                    return TensorState(self.states + [other])
            else:
                if isinstance(other, TensorState):
                    # Self is simple, other is tensor
                    return TensorState([self] + other.states)
                else:
                    # Both simple states
                    return TensorState([self, other])
        else:
            raise TypeError(f"Cannot tensor State with {type(other)}")
    
    def __rmatmul__(self, other):
        """Right tensor product: supports (A + B) @ C

        Distributes: (A + B) @ C = A @ C + B @ C
        """
        from sympy import Add

        if isinstance(other, YawOperator):
            # Check if other is a sum
            if isinstance(other._expr, Add):
                # Distribute
                terms = []
                for term in other._expr.args:
                    term_op = YawOperator(term, other.algebra)
                    terms.append(term_op @ self)

                result = terms[0]
                for term in terms[1:]:
                    result = result + term
                return result
            else:
                return other.__matmul__(self)
        else:
            return self.__matmul__(other)

    def __rmul__(self, scalar):
        """Scalar multiplication: scalar * state.
        
        Creates a weighted state for convex combinations.
        
        Args:
            scalar: Probability weight (should be in [0,1])
        
        Returns:
            MixedState with single component [(scalar, self)]
        
        Example:
            >>> psi0 = char(Z, 0)
            >>> weighted = 0.7 * psi0  # MixedState([(0.7, psi0)])
            >>> rho = 0.7*psi0 + 0.3*psi1  # Convex combination!
        """
        return MixedState([(scalar, self)])
    
    def __add__(self, other):
        """Add states to create convex combinations.
        
        Pure states added together create mixed states (with automatic
        normalization). This enables natural syntax:
            rho = 0.7*psi0 + 0.3*psi1
        
        Args:
            other: Another State instance
        
        Returns:
            MixedState combining the states
        
        Example:
            >>> psi0 = char(Z, 0)
            >>> psi1 = char(Z, 1)
            >>> rho = 0.7*psi0 + 0.3*psi1  # MixedState
            >>> rho.expect(Z)  # 0.4
        """
        if isinstance(other, MixedState):
            # self is pure, other is mixed
            # Treat self as MixedState([(1.0, self)])
            return MixedState([(1.0, self)]) + other
        elif isinstance(other, State):
            # Both are pure states (or at least non-MixedState)
            # Combine into MixedState
            return MixedState([(1.0, self), (1.0, other)])
        else:
            raise TypeError(f"Cannot add State and {type(other)}")
    
    def __radd__(self, other):
        """Right addition (for sum() to work)."""
        if other == 0:
            # This handles sum([states]) which starts with 0
            return self
        return self.__add__(other)
        
class EigenState(State):
    """Eigenstate of an observable: |λ⟩ such that A|λ⟩ = λ|λ⟩.
    
    The eigenvalue λ is determined by the index into the spectrum:
        eigenvalue = spec(observable)[index]
    
    where spec() returns eigenvalues in descending order.
    
    Attributes:
        observable: The operator this is an eigenstate of
        index: Position in descending spectrum (0 = largest eigenvalue)
        algebra: Associated algebra
        eigenvalue: The eigenvalue from spec(observable)[index]
    
    Example:
        >>> alg = qubit()
        >>> psi_0 = EigenState(alg.Z, 0, alg)  # Largest eigenvalue (+1)
        >>> psi_1 = EigenState(alg.Z, 1, alg)  # Smallest eigenvalue (-1)
        >>> psi_0.eigenvalue  # 1.0
        >>> psi_1.eigenvalue  # -1.0
    """
    
    def __init__(self, observable: YawOperator, index: int, algebra):
        """Create eigenstate of observable.
        
        Args:
            observable: Operator to be eigenstate of
            index: Spectrum position (0 for largest eigenvalue)
            algebra: Associated algebra
        """
        self.observable = observable
        self.index = index
        self.algebra = algebra
        
        # Compute eigenvalue from spectrum
        # To avoid circular dependency (spec() might try to create char() states),
        # we create a simple state to pass to spec()
        # Use a different generator than the observable if possible
        basis = _get_operator_basis(algebra)
        
        # Find a generator different from our observable for the state
        state_generator = None
        obs_str = str(observable._expr)
        for gen in basis:
            gen_str = str(gen._expr)
            if gen_str != 'I' and gen_str != obs_str:
                state_generator = gen
                break
        
        # If we can't find a different generator, use the first non-identity one
        if state_generator is None:
            for gen in basis:
                if str(gen._expr) != 'I':
                    state_generator = gen
                    break
        
        # Create a simple eigenstate without using char() to avoid recursion
        # We'll create a minimal state directly
        if state_generator is not None:
            # Create a temporary simple state for spectrum computation
            # This state just needs to define the GNS inner product
            temp_state = _SimpleState(state_generator, algebra)
            eigenvalues = spec(observable, temp_state)
        else:
            # Fallback: use no state and let spec handle it
            # This might recurse, but only if there's truly no way to create a state
            eigenvalues = spec(observable)
        
        if index < 0 or index >= len(eigenvalues):
            raise ValueError(
                f"Index {index} out of range for spectrum with "
                f"{len(eigenvalues)} eigenvalues: {eigenvalues}"
            )
        
        self.eigenvalue = eigenvalues[index]
    
    def expect(self, op: YawOperator, _depth=0) -> complex:
        """Compute expectation value via algebraic simplification.
        
        Uses linearity of expectation and handles:
        - Scalars
        - Identity
        - Powers of observable
        - Sums (via linearity)
        - Products with coefficients
        - Off-diagonal terms (returns 0)
        """
        if _depth > 10:
            return _clean_number(0.0)
        
        # Handle TensorSum: ⟨ψ|(A + B)|ψ⟩ = ⟨ψ|A|ψ⟩ + ⟨ψ|B|ψ⟩
        if isinstance(op, TensorSum):
            result = sum(self.expect(term, _depth=_depth+1) for term in op.terms)
            return _clean_number(result)

        if isinstance(op, Projector):
            if self.observable == op.base_operator:
                # Duality: ⟨char(A,j) | proj(A,k) | char(A,j)⟩ = δ_{jk}
                if self.index == op.eigenspace_index:
                    return _clean_number(1.0)
                else:
                    return _clean_number(0.0)
            # Different operator - fall through to general case
        
        # *** CRITICAL FIX: Detect YawOperator wrappers around projector symbols ***
        # If op is a YawOperator with a projector symbol expression,
        # handle it like a Projector object for eigenstate duality
        if isinstance(op, YawOperator) and not isinstance(op, Projector):
            expr_str = str(op._expr)
            if expr_str.startswith('proj('):
                # Parse the projector: proj(Observable, index)
                try:
                    import re
                    match = re.match(r'proj\((.+),\s*(\d+)\)', expr_str)
                    if match:
                        obs_str = match.group(1)
                        proj_index = int(match.group(2))
                        
                        # Check if this matches our observable
                        if str(self.observable) == obs_str:
                            # Duality: ⟨char(A,j) | proj(A,k) | char(A,j)⟩ = δ_{jk}
                            if self.index == proj_index:
                                return _clean_number(1.0)
                            else:
                                return _clean_number(0.0)
                except:
                    # If parsing fails, fall through to general case
                    pass
        
        # Normalize operator first
        op_norm = self.algebra.normalize(op)
        op_expr = op_norm._expr
        obs_expr = self.observable._expr
        
        # Case 1: Pure scalar
        if isinstance(op_expr, SympyNumber) or op_expr.is_number:
            return _clean_number(complex(op_expr))
        
        # Case 2: Identity
        if str(op_expr) == 'I':
            return _clean_number(1.0)
        
        # Case 3: The observable itself
        if op_expr == obs_expr:
            return _clean_number(self.eigenvalue)
        
        # Case 4: Power of observable
        if isinstance(op_expr, Pow) and op_expr.base == obs_expr:
            return _clean_number(self.eigenvalue) ** op_expr.exp
        
        # Case 5: Sum (linearity)
        if isinstance(op_expr, Add):
            total = 0.0
            for term in op_expr.args:
                term_op = YawOperator(term, self.algebra)
                total += self.expect(term_op, _depth=_depth+1)
            return _clean_number(total)
        
        # Case 6: Product with coefficient
        if isinstance(op_expr, Mul):
            coeff = 1.0
            operator_parts = []
            
            for arg in op_expr.args:
                if isinstance(arg, SympyNumber) or arg.is_number:
                    coeff *= complex(arg)
                else:
                    operator_parts.append(arg)
            
            if not operator_parts:
                return _clean_number(coeff)
            
            if len(operator_parts) == 1:
                op_part = YawOperator(operator_parts[0], self.algebra)
                return _clean_number(coeff) * self.expect(op_part, _depth=_depth+1)
            
            # Multiple operators - check if equal to observable
            product_expr = Mul(*operator_parts)
            if product_expr == obs_expr:
                return _clean_number(coeff) * self.eigenvalue
            
            # Otherwise off-diagonal
            return _clean_number(0.0)
        
        # Case 7: Other operators (off-diagonal)
        return _clean_number(0.0)
    
    def __str__(self):
        return f"char({self.observable}, {self.index})"

    def __ror__(self, operator):
        """Enable A | state syntax."""
        return self.expect(operator)

    def __repr__(self):
        return f"char({obs_str}, {self.index})"
    
    def flatten(self):
        """EigenStates are already atomic - return self."""
        return self
    
class TensorState(State):
    """Tensor product of states: |ψ⟩ ⊗ |φ⟩"""
    
    def __init__(self, states):
        self.states = list(states)
        
        if not self.states:
            raise ValueError("TensorState requires at least one state")
        
        # Flatten nested TensorStates
        flattened = []
        for s in self.states:
            if isinstance(s, TensorState):
                flattened.extend(s.states)
            else:
                flattened.append(s)
        self.states = flattened
    
    def expect(self, operator, _depth=0):
        """Compute expectation value on tensor product state."""
        if _depth > 10:
            return 0.0
        
        # Handle tensor sum: ⟨ψ|(A + B)|ψ⟩ = ⟨ψ|A|ψ⟩ + ⟨ψ|B|ψ⟩
        if isinstance(operator, TensorSum):
            result = sum(self.expect(term, _depth=_depth+1) for term in operator.terms)
            return _clean_number(result)
        
        # Handle tensor product operators
        if isinstance(operator, TensorProduct):
            if len(operator.factors) != len(self.states):
                raise ValueError(
                    f"Operator has {len(operator.factors)} factors "
                    f"but state has {len(self.states)} factors"
                )
            
            # Compute product of individual expectations
            result = 1
            for op, state in zip(operator.factors, self.states):
                result *= state.expect(op, _depth=_depth+1)
            
            return result
        
        # Single operator - assume it acts on first subsystem
        elif isinstance(operator, YawOperator):
            if len(self.states) == 1:
                return self.states[0].expect(operator, _depth=_depth+1)
            else:
                # Trace out other subsystems (partial trace)
                # For now, just apply to first subsystem
                return self.states[0].expect(operator, _depth=_depth+1)
        
        # Scalar
        return operator
    
    def __matmul__(self, other):
        """Extend tensor product of states"""
        if isinstance(other, State):
            if isinstance(other, TensorState):
                return TensorState(self.states + other.states)
            else:
                return TensorState(self.states + [other])
        else:
            raise TypeError(f"Cannot tensor TensorState with {type(other)}")
    
    def __str__(self):
        """Display as tensor product."""
        state_strs = [str(s) for s in self.states]
        return " ⊗ ".join(state_strs)
    
    def __repr__(self):
        """Display representation."""
        return f"TensorState({', '.join(str(s) for s in self.states)})"
    
    def flatten(self):
        """Flatten nested tensor structures: A @ (B @ C) → A @ B @ C
        
        Returns a new TensorState with all nested TensorStates expanded into
        a flat list of component states.
        
        Example:
            >>> state1 = char(Z, 0)
            >>> state2 = char(Z, 0) @ char(Z, 1)
            >>> nested = state1 @ state2  # 2 factors: [state1, state2]
            >>> flat = nested.flatten()    # 3 factors: [char(Z,0), char(Z,0), char(Z,1)]
        """
        flat_states = []
        for state in self.states:
            if isinstance(state, TensorState):
                # Recursively flatten nested tensor states
                nested_flat = state.flatten()
                flat_states.extend(nested_flat.states)
            else:
                flat_states.append(state)
        
        return TensorState(flat_states)
    
class LeftMultipliedState(State):
    """State with left multiplication: (A <<<) φ
    
    Creates a functional where (A <<<) φ (B) = φ(AB)
    
    This allows building coherent superpositions algebraically:
    φ_Bell = (1 + X⊗X <<<) char(Z,0)⊗char(Z,0) + (1 + X⊗X <<<) char(Z,1)⊗char(Z,1)
    
    Attributes:
        operator: Left multiplication operator
        state: Base functional
        algebra: Inherited algebra
    """
    
    def __init__(self, operator, state):
        """Create left-multiplied functional.
        
        Args:
            operator: Operator to left-multiply with
            state: Base functional
        """
        self.operator = operator
        self.state = state
        self.algebra = getattr(operator, 'algebra', None)
    
    def expect(self, op, _depth=0):
        """Compute expectation: (A <<<) φ (B) = φ(AB)"""
        if _depth > 10:
            return 0.0
        
        # Left multiply: measure A*op in original state
        product = self.operator * op
        if hasattr(product, 'normalize'):
            product = product.normalize()
        
        return self.state.expect(product, _depth=_depth+1)
    
    def __str__(self):
        return f"({self.operator} <<<) {self.state}"
    
    def __ror__(self, operator):
        """Enable A | state syntax."""
        return self.expect(operator)
    
    def __add__(self, other):
        """Add left-multiplied states."""
        if isinstance(other, (LeftMultipliedState, RightMultipliedState, State)):
            # Create a sum of functionals
            return SumState([self, other])
        else:
            raise TypeError(f"Cannot add LeftMultipliedState with {type(other)}")
    
    def __radd__(self, other):
        """Right addition."""
        if other == 0:
            return self
        return other + self
    
    def __mul__(self, scalar):
        """Scalar multiplication."""
        return ScaledState(scalar, self)
    
    def __rmul__(self, scalar):
        """Right scalar multiplication."""
        return ScaledState(scalar, self)
    
    def __truediv__(self, scalar):
        """Division by scalar."""
        return ScaledState(1/scalar, self)
    
    def flatten(self):
        """Flatten by recursively flattening the base state."""
        if hasattr(self.state, 'flatten'):
            return LeftMultipliedState(self.operator, self.state.flatten())
        return self


class RightMultipliedState(State):
    """State with right multiplication: (>>> A) φ
    
    Creates a functional where (>>> A) φ (B) = φ(BA)
    
    Attributes:
        operator: Right multiplication operator
        state: Base functional
        algebra: Inherited algebra
    """
    
    def __init__(self, operator, state):
        """Create right-multiplied functional.
        
        Args:
            operator: Operator to right-multiply with
            state: Base functional
        """
        self.operator = operator
        self.state = state
        self.algebra = getattr(operator, 'algebra', None)
    
    def expect(self, op, _depth=0):
        """Compute expectation: (>>> A) φ (B) = φ(BA)"""
        if _depth > 10:
            return 0.0
        
        # Right multiply: measure op*A in original state
        product = op * self.operator
        if hasattr(product, 'normalize'):
            product = product.normalize()
        
        return self.state.expect(product, _depth=_depth+1)
    
    def __str__(self):
        return f"(>>> {self.operator}) {self.state}"
    
    def __ror__(self, operator):
        """Enable A | state syntax."""
        return self.expect(operator)
    
    def __add__(self, other):
        """Add right-multiplied states."""
        if isinstance(other, (LeftMultipliedState, RightMultipliedState, State)):
            return SumState([self, other])
        else:
            raise TypeError(f"Cannot add RightMultipliedState with {type(other)}")
    
    def __radd__(self, other):
        """Right addition."""
        if other == 0:
            return self
        return other + self
    
    def __mul__(self, scalar):
        """Scalar multiplication."""
        return ScaledState(scalar, self)
    
    def __rmul__(self, scalar):
        """Right scalar multiplication."""
        return ScaledState(scalar, self)
    
    def __truediv__(self, scalar):
        """Division by scalar."""
        return ScaledState(1/scalar, self)
    
    def flatten(self):
        """Flatten by recursively flattening the base state."""
        if hasattr(self.state, 'flatten'):
            return RightMultipliedState(self.operator, self.state.flatten())
        return self


class ScaledState(State):
    """State scaled by a constant: c * φ
    
    Implements (c * φ)(A) = c * φ(A)
    """
    
    def __init__(self, scalar, state):
        self.scalar = scalar
        self.state = state
        self.algebra = getattr(state, 'algebra', None)
    
    def expect(self, op, _depth=0):
        """Compute expectation: (c * φ)(A) = c * φ(A)"""
        return self.scalar * self.state.expect(op, _depth=_depth)
    
    def __str__(self):
        return f"{self.scalar} * {self.state}"
    
    def __ror__(self, operator):
        return self.expect(operator)
    
    def __add__(self, other):
        if isinstance(other, State):
            return SumState([self, other])
        else:
            raise TypeError(f"Cannot add ScaledState with {type(other)}")
    
    def __radd__(self, other):
        if other == 0:
            return self
        return other + self
    
    def __mul__(self, scalar):
        return ScaledState(self.scalar * scalar, self.state)
    
    def __rmul__(self, scalar):
        return ScaledState(scalar * self.scalar, self.state)
    
    def __truediv__(self, scalar):
        return ScaledState(self.scalar / scalar, self.state)
    
    def flatten(self):
        """Flatten by recursively flattening the base state."""
        if hasattr(self.state, 'flatten'):
            return ScaledState(self.scalar, self.state.flatten())
        return self


class SumState(State):
    """Sum of state functionals: φ₁ + φ₂ + ...
    
    Implements (φ₁ + φ₂)(A) = φ₁(A) + φ₂(A)
    """
    
    def __init__(self, states):
        self.states = list(states)
        self.algebra = getattr(states[0], 'algebra', None) if states else None
    
    def expect(self, op, _depth=0):
        """Compute expectation: (φ₁ + φ₂)(A) = φ₁(A) + φ₂(A)"""
        result = sum(state.expect(op, _depth=_depth) for state in self.states)
        return _clean_number(result)
    
    def __str__(self):
        return " + ".join(str(s) for s in self.states)
    
    def __ror__(self, operator):
        return self.expect(operator)
    
    def __add__(self, other):
        if isinstance(other, SumState):
            return SumState(self.states + other.states)
        elif isinstance(other, State):
            return SumState(self.states + [other])
        else:
            raise TypeError(f"Cannot add SumState with {type(other)}")
    
    def __radd__(self, other):
        if other == 0:
            return self
        return other + self
    
    def __mul__(self, scalar):
        return ScaledState(scalar, self)
    
    def __rmul__(self, scalar):
        return ScaledState(scalar, self)
    
    def __truediv__(self, scalar):
        return ScaledState(1/scalar, self)
    
    def flatten(self):
        """Flatten by recursively flattening each component state."""
        flattened_states = []
        for state in self.states:
            if hasattr(state, 'flatten'):
                flattened_states.append(state.flatten())
            else:
                flattened_states.append(state)
        return SumState(flattened_states)
    
class ConjugatedState(State):
    """State transformed by unitary: U << |ψ⟩.
    
    Implements the Heisenberg picture: instead of transforming the state,
    transform the operators measured against it.
    
    Property: (U << ψ)(A) = ψ(U† A U)
    
    Attributes:
        unitary: Transformation operator
        state: Original state
        algebra: Inherited algebra
    """
    
    def __init__(self, unitary, state):
        """Create conjugated state.
        
        Args:
            unitary: Operator to transform by
            state: Original state
        """
        self.unitary = unitary
        self.state = state
        self.algebra = unitary.algebra
    
    def expect(self, op, _depth=0):
        """Compute expectation: ⟨U|ψ⟩|A|U|ψ⟩⟩ = ⟨ψ|U†AU|ψ⟩"""
        if _depth > 10:
            return 0.0
        
        # Transform operator instead of state
        transformed_op = self.unitary.conj_op(op)
        transformed_op_norm = transformed_op.normalize()
        
        # Measure in original state
        return self.state.expect(transformed_op_norm, _depth=_depth+1)
    
    def __str__(self):
        return f"({self.unitary} << {self.state})"

    def __ror__(self, operator):
        """Enable A | state syntax."""
        return self.expect(operator)

class MixedState(State):
    """Mixed state (classical probability distribution over pure states).
    """
    
    def __init__(self, components):
        """Create mixed state from probability-state pairs.
        
        Args:
            components: List of (probability, state) tuples
        """
        self.components = components
    
    def __call__(self, operator):
        """Expectation value: Tr(ρ A) = ∑_i p_i ⟨psi_i|A|psi_i⟩"""
        return sum(prob * state(operator) 
                   for prob, state in self.components)
    
    def expect(self, operator, _depth=0):
        """Expectation value with normalized probabilities.
        
        For mixed state ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|:
            ⟨A⟩_ρ = Tr(ρ A) = Σᵢ pᵢ ⟨ψᵢ|A|ψᵢ⟩
        """
        # Normalize probabilities
        total = sum(p for p, _ in self.components)
        if total > 1e-10:
            normalized_probs = [p/total for p, _ in self.components]
        else:
            normalized_probs = [p for p, _ in self.components]
        
        result = sum(prob * state.expect(operator) 
                   for prob, (_, state) in zip(normalized_probs, self.components))
        return _clean_number(result)
    
    def __repr__(self):
        """String representation with normalized probabilities."""
        # Normalize for display
        total = sum(p for p, _ in self.components)
        if total > 1e-10:
            normalized = [(p/total, s) for p, s in self.components]
        else:
            normalized = self.components
        
        terms = [f"{prob:.3f}*{state}" for prob, state in normalized]
        return " + ".join(terms)
    
    def __str__(self):
        return self.__repr__()
    
    def __add__(self, other):
        """Add mixed states with normalization."""
        if isinstance(other, MixedState):
            # Combine components
            all_components = self.components + other.components
            # Normalize
            total = sum(p for p, _ in all_components)
            if total > 1e-10:  # Avoid division by zero
                normalized = [(p/total, s) for p, s in all_components]
                return MixedState(normalized)
            else:
                return MixedState(all_components)
        elif isinstance(other, State):
            return self + MixedState([(1.0, other)])
        else:
            raise TypeError(f"Cannot add MixedState and {type(other)}")
    
    def __rmul__(self, scalar):
        """Scalar multiplication (for weighted combinations)."""
        # Note: This breaks normalization!
        # Need to handle carefully or prohibit
        scaled = [(scalar * p, s) for p, s in self.components]
        return MixedState(scaled)


class SuperpositionState(State):
    """Coherent quantum superposition: |ψ⟩ = Σᵢ αᵢ|ψᵢ⟩
    
    Unlike MixedState (classical mixture), this represents a coherent 
    quantum superposition with complex amplitudes.
    
    This is used when applying operator sums to states:
        (A + B)|ψ⟩ = A|ψ⟩ + B|ψ⟩
    
    The amplitudes are automatically extracted from the operator algebra.
    """
    
    def __init__(self, components):
        """Create superposition from amplitude-state pairs.
        
        Args:
            components: List of (amplitude, state) tuples
                       where amplitude is a complex number or YawOperator coefficient
        """
        self.components = components
    
    def __call__(self, operator):
        """Expectation value: ⟨ψ|A|ψ⟩ where |ψ⟩ = Σᵢ αᵢ|ψᵢ⟩
        
        For normalized superposition:
            ⟨ψ|A|ψ⟩ = Σᵢⱼ αᵢ* αⱼ ⟨ψᵢ|A|ψⱼ⟩
        
        For simplicity, we compute diagonally:
            ⟨ψ|A|ψ⟩ ≈ Σᵢ |αᵢ|² ⟨ψᵢ|A|ψᵢ⟩
        """
        # Compute normalization
        norm_sq = sum(abs(complex(amp))**2 for amp, _ in self.components)
        
        if norm_sq < 1e-10:
            return 0.0
        
        # Diagonal approximation
        result = sum(abs(complex(amp))**2 * state.expect(operator) 
                    for amp, state in self.components)
        
        return _clean_number(result / norm_sq)
    
    def expect(self, operator, _depth=0):
        """Expectation value (same as __call__ for pure states)."""
        return self(operator)
    
    def __repr__(self):
        """String representation showing amplitudes."""
        # Format with amplitudes
        terms = []
        for amp, state in self.components:
            amp_val = complex(amp) if not isinstance(amp, (int, float, complex)) else amp
            
            # Format amplitude nicely
            if abs(amp_val.imag) < 1e-10:
                # Real amplitude
                amp_str = f"{amp_val.real:.3f}" if abs(amp_val.real - 1.0) > 1e-10 else ""
            else:
                # Complex amplitude
                amp_str = f"({amp_val.real:.3f}+{amp_val.imag:.3f}j)"
            
            if amp_str and amp_str != "1.000":
                terms.append(f"{amp_str}*{state}")
            else:
                terms.append(str(state))
        
        return " + ".join(terms) if terms else "0"
    
    def __str__(self):
        return self.__repr__()
    
# ============================================================================
# QUANTUM CHANNELS
# ============================================================================

class OpChannel:
    """Quantum channel as a completely positive map on operators.
    
    Given Kraus operators {K_i}, implements the superoperator:
        E(A) = ∑ᵢ Kᵢ A Kᵢ†
    
    Channels are:
    - Linear: E(ₖ±A + βB) = ₖ±E(A) + βE(B)
    - Completely positive (CP)
    - Trace preserving (if ∑ᵢ Kᵢ†Kᵢ = I)
    - Unital (if ∑ᵢ KᵢKᵢ† = I)
    
    Attributes:
        kraus_ops: List of Kraus operators
    """
    
    def __init__(self, kraus_ops):
        """Create channel from Kraus operators.
        
        Args:
            kraus_ops: List of YawOperator instances (Kraus operators)
        """
        self.kraus_ops = list(kraus_ops)
        
        if not self.kraus_ops:
            raise ValueError("Channel must have at least one Kraus operator")
    
    def __call__(self, operator):
        """Apply channel to operator: E(A) = ∑ᵢ Kᵢ A Kᵢ†
        
        Args:
            operator: YawOperator to transform
            
        Returns:
            Transformed operator
        """
        result = None
        
        for K in self.kraus_ops:
            # Apply Kraus operator: Kᵢ A Kᵢ†
            term = K.conj_op(operator)  # K >> A
            
            if result is None:
                result = term
            else:
                result = result + term
        
        return result

    def __ror__(self, operator):
        """Enable operator | channel syntax (Unix pipe style).
        
        Allows composition: op | channel1 | channel2 | channel3
        
        Args:
            operator: YawOperator to transform
            
        Returns:
            Transformed operator
            
        Example:
            >>> noise = opChannel([K0, K1])
            >>> X | noise  # Apply noise channel to X
            >>> X | noise | another_channel  # Compose channels
        """
        return self(operator)
    
    def __mul__(self, other):
        """Compose channels: (E₁ ∘ E₂)(A) = E₁(E₂(A))
        
        Args:
            other: Another OpChannel
            
        Returns:
            Composed channel
        """
        if isinstance(other, OpChannel):
            return ComposedChannel(self, other)
        raise TypeError(f"Cannot compose OpChannel with {type(other)}")
    
    def is_trace_preserving(self):
        """Check if channel is trace preserving: ∑ᵢ Kᵢ†Kᵢ = I
        
        Returns:
            YawOperator (should normalize to I if TP)
        """
        result = None
        for K in self.kraus_ops:
            term = K.adjoint() * K
            if result is None:
                result = term
            else:
                result = result + term
        return result
    
    def is_unital(self):
        """Check if channel is unital: ∑ᵢ KᵢKᵢ† = I
        
        Returns:
            YawOperator (should normalize to I if unital)
        """
        result = None
        for K in self.kraus_ops:
            term = K * K.adjoint()
            if result is None:
                result = term
            else:
                result = result + term
        return result
    
    def __str__(self):
        return f"OpChannel({len(self.kraus_ops)} Kraus ops)"
    
    def __repr__(self):
        return f"OpChannel(kraus_ops={self.kraus_ops})"

class ComposedChannel(OpChannel):
    """Composition of two channels: (E₁ ∘ E₂)(A) = E₁(E₂(A))"""
    
    def __init__(self, first, second):
        """Compose two channels.
        
        Args:
            first: Applied second (outer)
            second: Applied first (inner)
        """
        self.first = first
        self.second = second
        # Don't store kraus_ops directly (would require computing composition)
        self._kraus_ops = None
    
    @property
    def kraus_ops(self):
        """Lazy computation of composed Kraus operators."""
        if self._kraus_ops is None:
            # E₁∘E₂ has Kraus ops {Kᵢᵢ½₁ᵢ¾K₁ᵢ½₂ᵢ¾}
            composed = []
            for K1 in self.first.kraus_ops:
                for K2 in self.second.kraus_ops:
                    composed.append(K1 * K2)
            self._kraus_ops = composed
        return self._kraus_ops
    
    def __call__(self, operator):
        """Apply composed channel."""
        return self.first(self.second(operator))
    
    def __str__(self):
        return f"({self.first} ∘ {self.second})"

def opChannel(kraus_ops):
    """Create operator channel from Kraus operators.
    
    Convenience constructor for OpChannel.
    
    Args:
        kraus_ops: List of Kraus operators
        
    Returns:
        OpChannel instance
        
    Example:
        >>> K0 = (I + Z) / 2
        >>> K1 = (I - Z) / 2
        >>> channel = opChannel([K0, K1])
        >>> channel(X)  # Dephasing channel applied to X
    """
    return OpChannel(kraus_ops)

class StChannel:
    """Quantum channel acting on states (dual to OpChannel).
    
    Given Kraus operators {K_i}, transforms states via:
        E*(|ψ⟩)(A) = ⟨ψ|E(A)|ψ⟩ = ⟨ψ|∑ᵢ Kᵢ A Kᵢ†|ψ⟩
    
    This is the Schrödinger picture: states evolve, operators fixed.
    Dual to OpChannel (Heisenberg picture).
    
    Attributes:
        kraus_ops: List of Kraus operators
        op_channel: Corresponding operator channel
    """
    
    def __init__(self, kraus_ops):
        """Create state channel from Kraus operators.
        
        Args:
            kraus_ops: List of YawOperator instances (Kraus operators)
        """
        self.kraus_ops = list(kraus_ops)
        
        if not self.kraus_ops:
            raise ValueError("Channel must have at least one Kraus operator")
        
        # Store dual operator channel for convenience
        self.op_channel = OpChannel(kraus_ops)
    
    def __call__(self, state):
        """Apply channel to state: E*(|ψ⟩)
        
        Args:
            state: State to transform
            
        Returns:
            TransformedState instance
        """
        return TransformedState(self, state)

    def __ror__(self, state):
        """Enable state | channel syntax (Unix pipe style).
        
        Allows composition: psi | channel1 | channel2 | channel3
        
        Args:
            state: State to transform
            
        Returns:
            Transformed state
            
        Example:
            >>> noise = stChannel([K0, K1])
            >>> psi0 | noise  # Apply noise to state
            >>> psi0 | noise | decay  # Compose channels
        """
        return self(state)
    
    def __mul__(self, other):
        """Compose state channels: (E₁ ∘ E₂)*(|ψ⟩) = E₁*(E₂*(|ψ⟩))
        
        Args:
            other: Another StChannel
            
        Returns:
            Composed state channel
        """
        if isinstance(other, StChannel):
            return ComposedStChannel(self, other)
        raise TypeError(f"Cannot compose StChannel with {type(other)}")
    
    def __str__(self):
        return f"StChannel({len(self.kraus_ops)} Kraus ops)"
    
    def __repr__(self):
        return f"StChannel(kraus_ops={self.kraus_ops})"

class TransformedState(State):
    """State transformed by a quantum channel.
    
    Implements: E*(|ψ⟩)(A) = ⟨ψ|E(A)|ψ⟩
    
    This is the key duality: to measure A on the transformed state
    is the same as measuring E(A) on the original state.
    
    Attributes:
        channel: StChannel that transformed the state
        state: Original state
    """
    
    def __init__(self, channel, state):
        """Create transformed state.
        
        Args:
            channel: StChannel to apply
            state: Original state
        """
        self.channel = channel
        self.state = state
    
    def expect(self, operator, _depth=0):
        """Compute expectation: ⟨E*(ψ)|A|E*(ψ)⟩ = ⟨ψ|E(A)|ψ⟩
        
        Args:
            operator: Operator to measure
            _depth: Recursion depth counter
            
        Returns:
            Expectation value
        """
        if _depth > 10:
            return 0.0
        
        # Apply channel to operator
        transformed_op = self.channel.op_channel(operator)
        
        # Normalize
        if hasattr(transformed_op, 'normalize'):
            transformed_op = transformed_op.normalize()
        
        # Measure on original state
        return self.state.expect(transformed_op, _depth=_depth+1)
    
    def __str__(self):
        return f"{self.channel}({self.state})"
    
    def __ror__(self, operator):
        """Enable A | state syntax."""
        return self.expect(operator)
    
    def flatten(self):
        """Flatten by recursively flattening the base state."""
        if hasattr(self.state, 'flatten'):
            return TransformedState(self.channel, self.state.flatten())
        return self

class ComposedStChannel(StChannel):
    """Composition of two state channels."""
    
    def __init__(self, first, second):
        """Compose two state channels.
        
        Args:
            first: Applied second (outer)
            second: Applied first (inner)
        """
        self.first = first
        self.second = second
        # Kraus ops are composition of individual Kraus ops
        self._kraus_ops = None
        self._op_channel = None
    
    @property
    def kraus_ops(self):
        """Lazy computation of composed Kraus operators."""
        if self._kraus_ops is None:
            composed = []
            for K1 in self.first.kraus_ops:
                for K2 in self.second.kraus_ops:
                    composed.append(K1 * K2)
            self._kraus_ops = composed
        return self._kraus_ops
    
    @property
    def op_channel(self):
        """Lazy computation of dual operator channel."""
        if self._op_channel is None:
            self._op_channel = OpChannel(self.kraus_ops)
        return self._op_channel
    
    def __call__(self, state):
        """Apply composed channel."""
        return TransformedState(self, state)
    
    def __str__(self):
        return f"({self.first} ∘ {self.second})"

def stChannel(kraus_ops):
    """Create state channel from Kraus operators.
    
    Convenience constructor for StChannel.
    
    Args:
        kraus_ops: List of Kraus operators
        
    Returns:
        StChannel instance
        
    Example:
        >>> K0 = (I + Z) / 2
        >>> K1 = (I - Z) / 2
        >>> channel = stChannel([K0, K1])
        >>> psi0 = char(Z, 0)
        >>> channel(psi0)  # Dephasing channel applied to |0⟩
    """
    return StChannel(kraus_ops)

# ============================================================================
# MEASUREMENT (STOCHASTIC UPDATE)
# ============================================================================

class CollapsedState(State):
    """State after measurement collapse: K|ψ⟩/ψÅ¡p normalized as functional.
    
    For density matrix semantics: ᵢ' = (KᵢK†)/p where p = Tr(KᵢK†)
    
    Since states are functionals:
        ψ'(A) = ψ(K†AK) / p
    
    Attributes:
        kraus_op: Kraus operator that collapsed the state
        state: Original state before measurement
        probability: Probability p of this outcome
    """
    
    def __init__(self, kraus_op, state, probability):
        """Create collapsed state.
        
        Args:
            kraus_op: Kraus operator K
            state: Original state |ψ⟩
            probability: p = ⟨ψ|K†K|ψ⟩
        """
        self.kraus_op = kraus_op
        self.state = state
        self.probability = probability
        
        # if probability < 1e-10:
        #    raise ValueError("Cannot collapse with zero probability")
    
    def expect(self, operator, _depth=0):
        """Compute expectation: ⟨ψ'|A|ψ'⟩ = ⟨ψ|K†AK|ψ⟩ / p
        
        Args:
            operator: Operator to measure
            _depth: Recursion depth counter
            
        Returns:
            Expectation value
        """
        if _depth > 10:
            return 0.0

        # Handle zero probability
        if self.probability < 1e-10:
            return 0.0  # Convention: zero-probability branch has zero expectation
        
        # Transform operator: K† A K
        transformed = self.kraus_op.conj_op(operator)
        
        # Normalize
        if hasattr(transformed, 'normalize'):
            transformed = transformed.normalize()
        
        # Measure on original state and normalize by probability
        return self.state.expect(transformed, _depth=_depth+1) / self.probability
    
    def __str__(self):
        return f"Collapsed[{self.kraus_op}, p={self.probability:.3f}]({self.state})"
    
    def __ror__(self, operator):
        """Enable A | state syntax."""
        return self.expect(operator)

class OpMeasurement:
    """Measurement device for operators (Heisenberg picture).
    
    Encapsulates a measurement with Kraus operators and state.
    Each call samples a random outcome according to Born rule.
    
    Usage:
        >>> measure = opMeasure([K0, K1], state, seed=42)
        >>> updated_op, prob = measure(A)
        # Returns (Kᵢ >> A, p(i)) for random outcome i
    
    Attributes:
        kraus_ops: List of Kraus operators
        state: State being measured
        seed: Random seed for reproducibility
    """
    
    def __init__(self, kraus_ops, state, seed=None):
        """Create measurement device.
        
        Args:
            kraus_ops: List of Kraus operators
            state: State to measure
            seed: Optional random seed
        """
        self.kraus_ops = list(kraus_ops)
        self.state = state
        self.seed = seed
        
        if not self.kraus_ops:
            raise ValueError("Must provide at least one Kraus operator")
    
    def __call__(self, operator):
        """Sample measurement outcome and return transformed operator.
        
        Samples outcome i with probability p(i) = ⟨ψ|Kᵢ†Kᵢ|ψ⟩
        
        Args:
            operator: Operator to transform
            
        Returns:
            (Kᵢ >> operator, p(i))
        """
        # Compute probabilities
        probs = []
        for K in self.kraus_ops:
            prob = self.state.expect(K.adjoint() * K)
            # Convert to float, handling complex numbers
            prob_float = float(prob.real) if hasattr(prob, 'real') else float(prob)
            probs.append(max(0.0, prob_float))  # Ensure non-negative
        
        # For trace-preserving channels, probabilities sum to 1
        # But check for numerical issues
        total = sum(probs)
        if total < 1e-10:
            raise ValueError("All measurement outcomes have zero probability")
        
        # Sample outcome
        if self.seed is not None:
            random.seed(self.seed)
        i = random.choices(range(len(self.kraus_ops)), weights=probs)[0]
        
        # Apply Kraus operator: Kᵢ >> A = Kᵢ† A Kᵢ
        K_i = self.kraus_ops[i]
        transformed = K_i.conj_op(operator)
        
        return (transformed, probs[i])
    
    def __str__(self):
        return f"OpMeasurement({len(self.kraus_ops)} outcomes)"
    
    def __repr__(self):
        return f"OpMeasurement(kraus_ops={len(self.kraus_ops)}, seed={self.seed})"

class StMeasurement:
    """Measurement device for states (Schrödinger picture).
    
    Encapsulates a measurement with Kraus operators and state.
    Each call samples a random outcome according to Born rule.
    
    Usage:
        >>> measure = stMeasure([K0, K1], state, seed=42)
        >>> collapsed_state, outcome_index = measure()
        # Returns (K_i << state / p(i), i) for random outcome i
    
    Attributes:
        kraus_ops: List of Kraus operators
        state: State being measured
        seed: Random seed for reproducibility
    """
    
    def __init__(self, kraus_ops, state, seed=None):
        """Create measurement device.
        
        Args:
            kraus_ops: List of Kraus operators
            state: State to measure
            seed: Optional random seed
        """
        self.kraus_ops = list(kraus_ops)
        self.state = state
        self.seed = seed
        
        if not self.kraus_ops:
            raise ValueError("Must provide at least one Kraus operator")
    
    def __call__(self):
        """Sample measurement outcome and return collapsed state.
        
        Samples outcome i with probability p(i) = ⟨ψ|Kᵢ†Kᵢ|ψ⟩
        Returns collapsed state: ψ'(A) = ψ(Kᵢ†AKᵢ) / p(i)
        
        Returns:
            (collapsed_state, i) - tuple of collapsed state and outcome index
        """
        # Compute probabilities
        probs = []
        for K in self.kraus_ops:
            prob = self.state.expect(K.adjoint() * K)
            # Convert to float, handling complex numbers
            prob_float = float(prob.real) if hasattr(prob, 'real') else float(prob)
            probs.append(max(0.0, prob_float))  # Ensure non-negative
        
        # For trace-preserving channels, probabilities sum to 1
        # But check for numerical issues
        total = sum(probs)
        if total < 1e-10:
            raise ValueError("All measurement outcomes have zero probability")
        
        # Sample outcome
        if self.seed is not None:
            random.seed(self.seed)
        i = random.choices(range(len(self.kraus_ops)), weights=probs)[0]
        
        # Collapse state: Kᵢ << state, normalized by p(i)
        K_i = self.kraus_ops[i]
        collapsed = CollapsedState(K_i, self.state, probs[i])
        
        return (collapsed, i)
    
    def __str__(self):
        return f"StMeasurement({len(self.kraus_ops)} outcomes)"
    
    def __repr__(self):
        return f"StMeasurement(kraus_ops={len(self.kraus_ops)}, seed={self.seed})"

def opMeasure(kraus_ops, state, seed=None):
    """Create operator measurement device.
    
    Args:
        kraus_ops: List of Kraus operators
        state: State to measure
        seed: Random seed for reproducibility (optional)
        
    Returns:
        OpMeasurement object (callable on operators)
        
    Example:
        >>> K0 = (I + Z) / 2  # Project to |0⟩
        >>> K1 = (I - Z) / 2  # Project to |1⟩
        >>> measure = opMeasure([K0, K1], psi0, seed=42)
        >>> updated_X, prob = measure(X)
    """
    return OpMeasurement(kraus_ops, state, seed)

def stMeasure(kraus_ops, state, seed=None):
    """Create state measurement device.
    
    Args:
        kraus_ops: List of Kraus operators
        state: State to measure
        seed: Random seed for reproducibility (optional)
        
    Returns:
        StMeasurement object (callable, returns collapsed state)
        
    Example:
        >>> K0 = (I + Z) / 2  # Project to |0⟩
        >>> K1 = (I - Z) / 2  # Project to |1⟩
        >>> measure = stMeasure([K0, K1], psi0, seed=42)
        >>> collapsed_state, outcome_index = measure()
    """
    return StMeasurement(kraus_ops, state, seed)

# ============================================================================
# MEASUREMENT BRANCHES (ENSEMBLE SEMANTICS)
# ============================================================================

class OpBranches:
    """All measurement branches in operator picture (Heisenberg).
    
    Returns complete list of all possible measurement outcomes with
    their probabilities. No sampling - shows full ensemble.
    
    Usage:
        >>> branches = opBranches([K0, K1], state)
        >>> outcomes = branches(A)
        # Returns [(K₀ >> A, p(0)), (K₁ >> A, p(1))]
    
    Attributes:
        kraus_ops: List of Kraus operators
        state: State being measured
    """
    
    def __init__(self, kraus_ops, state):
        """Create branching measurement.
        
        Args:
            kraus_ops: List of Kraus operators
            state: State to measure
        """
        self.kraus_ops = list(kraus_ops)
        self.state = state
        
        if not self.kraus_ops:
            raise ValueError("Must provide at least one Kraus operator")
    
    def __call__(self, operator):
        """Return all measurement branches.
        
        Args:
            operator: Operator to transform
            
        Returns:
            List of (transformed_operator, probability) tuples
        """
        branches = []
        
        for K in self.kraus_ops:
            prob = self.state.expect(K.adjoint() * K)
            prob_float = float(prob.real) if hasattr(prob, 'real') else float(prob)
            prob_float = max(0.0, prob_float)  # Ensure non-negative
            
            # Apply Kraus operator: K >> A
            transformed = K.conj_op(operator)
            
            branches.append((transformed, prob_float))
        
        return branches
    
    def __str__(self):
        return f"OpBranches({len(self.kraus_ops)} branches)"
    
    def __repr__(self):
        return f"OpBranches(kraus_ops={len(self.kraus_ops)})"

class StBranches:
    """All measurement branches in state picture (Schrödinger).
    
    Returns complete list of all possible measurement outcomes with
    their probabilities. No sampling - shows full ensemble.
    
    Usage:
        >>> branches = stBranches([K0, K1], state)
        >>> outcomes = branches()
        # Returns [(K₀ << state / p(0), p(0)), (K₁ << state / p(1), p(1))]
    
    Attributes:
        kraus_ops: List of Kraus operators
        state: State being measured
    """
    
    def __init__(self, kraus_ops, state):
        """Create branching measurement.
        
        Args:
            kraus_ops: List of Kraus operators
            state: State to measure
        """
        self.kraus_ops = list(kraus_ops)
        self.state = state
        
        if not self.kraus_ops:
            raise ValueError("Must provide at least one Kraus operator")
    
    def __call__(self):
        """Return all measurement branches.
        
        Returns:
            List of (collapsed_state, probability) tuples
        """
        branches = []
        
        for K in self.kraus_ops:
            prob = self.state.expect(K.adjoint() * K)
            prob_float = float(prob.real) if hasattr(prob, 'real') else float(prob)
            prob_float = max(0.0, prob_float)  # Ensure non-negative
            
            # Include zero prob branches for full ensemble view
            try:
                collapsed = CollapsedState(K, self.state, prob_float)
                branches.append((collapsed, prob_float))
            except ValueError:
                # Zero probability - include marker
                branches.append((None, prob_float))
        
        return branches
    
    def __str__(self):
        return f"StBranches({len(self.kraus_ops)} branches)"
    
    def __repr__(self):
        return f"StBranches(kraus_ops={len(self.kraus_ops)})"

def opBranches(kraus_ops, state):
    """Create operator branching measurement.
    
    Returns all possible measurement outcomes (full ensemble).
    
    Args:
        kraus_ops: List of Kraus operators
        state: State to measure
        
    Returns:
        OpBranches object (callable on operators)
        
    Example:
        >>> K0 = (I + Z) / 2
        >>> K1 = (I - Z) / 2
        >>> branches = opBranches([K0, K1], psi0)
        >>> outcomes = branches(X)
        # Returns [(K₀ >> X, 1.0), (K₁ >> X, 0.0)]
    """
    return OpBranches(kraus_ops, state)

def stBranches(kraus_ops, state):
    """Create state branching measurement.
    
    Returns all possible measurement outcomes (full ensemble).
    
    Args:
        kraus_ops: List of Kraus operators
        state: State to measure
        
    Returns:
        StBranches object (callable, returns list of branches)
        
    Example:
        >>> K0 = (I + Z) / 2
        >>> K1 = (I - Z) / 2
        >>> branches = stBranches([K0, K1], psi0)
        >>> outcomes = branches()
        # Returns [(collapsed₀, 1.0), (collapsed₁, 0.0)]
    """
    return StBranches(kraus_ops, state)

def compose_st_branches(new_kraus_ops, branch_list):
    """Compose measurements: apply new measurement to each branch.
    
    Branches grow exponentially: n branches × m outcomes = n*m branches.
    
    Args:
        new_kraus_ops: List of Kraus operators for second measurement
        branch_list: List of (state, prob) from previous measurement
        
    Returns:
        Combined list of (state, prob) for all paths
        
    Example:
        >>> # First measurement
        >>> branches1 = stBranches([K0, K1], psi0)()
        >>> # Second measurement on all branches
        >>> branches2 = compose_st_branches([L0, L1], branches1)
        # Now have 2*2 = 4 branches (all measurement histories)
    """
    all_branches = []
    
    for state, prob in branch_list:
        # Measure each branch
        measure = stBranches(new_kraus_ops, state)
        sub_branches = measure()
        
        # Accumulate with probability multiplication
        for sub_state, sub_prob in sub_branches:
            all_branches.append((sub_state, prob * sub_prob))
    
    return all_branches

def compose_op_branches(new_kraus_ops, state, branch_list):
    """Compose operator measurements (requires state for probabilities).
    
    Args:
        new_kraus_ops: List of Kraus operators for second measurement
        state: State (needed to compute probabilities)
        branch_list: List of (operator, prob) from previous measurement
        
    Returns:
        Combined list of (operator, prob) for all paths
    """
    # For operator branches, we need to track how operators evolve
    # This is more complex - may want to return list of (op, state_branch, prob)
    # For now, simpler to just use stBranches and track states
    raise NotImplementedError("Use compose_st_branches for sequential measurements")

# ============================================================================
# QUANTUM FOURIER TRANSFORM
# ============================================================================

class QFT:
    """Quantum Fourier Transform as algebraic automorphism.
    
    The QFT is defined purely algebraically by its action on generators:
        qft(X, Z) >> Z = X
        qft(X, Z) >> X = Z†
    
    This extends to arbitrary operators by the automorphism property:
        qft >> (AB) = (qft >> A)(qft >> B)
    
    For qudits with pow(d), we use Z† = Z^(d-1).
    
    Attributes:
        gen_x: X generator (shift operator)
        gen_z: Z generator (clock operator)
        algebra: Algebra containing X and Z
    
    Example:
        >>> X, Z = algebra.X, algebra.Z
        >>> W = qft(X, Z)
        >>> (W >> Z).normalize()  # Returns X
        >>> (W >> X).normalize()  # Returns Z^(d-1)
    """
    
    def __init__(self, gen_x, gen_z, algebra=None):
        """Create QFT operator.
        
        Args:
            gen_x: X generator
            gen_z: Z generator
            algebra: Optional algebra (inferred from operators if None)
        """
        self.gen_x = gen_x
        self.gen_z = gen_z
        
        # Infer algebra
        if algebra is None:
            if hasattr(gen_x, 'algebra') and gen_x.algebra is not None:
                algebra = gen_x.algebra
            elif hasattr(gen_z, 'algebra') and gen_z.algebra is not None:
                algebra = gen_z.algebra
        
        self.algebra = algebra
        
        # Get SymPy representations
        self.x_expr = gen_x._expr
        self.z_expr = gen_z._expr
        
        # Compute Z† (conjugate of Z)
        if algebra and algebra.power_mod:
            # For pow(d): Z† = Z^(d-1)
            d = algebra.power_mod
            self.z_dag_expr = self.z_expr ** (d - 1)
        else:
            # Generic: Z†
            self.z_dag_expr = Dagger(self.z_expr)
        
        # Similarly for X†
        if algebra and algebra.power_mod:
            d = algebra.power_mod
            self.x_dag_expr = self.x_expr ** (d - 1)
        else:
            self.x_dag_expr = Dagger(self.x_expr)
    
    def conj_op(self, operator):
        """Apply QFT via conjugation: W >> A = W A W†
        
        Uses the defining relations:
            W >> Z = X
            W >> X = Z†
        
        Args:
            operator: Operator to transform
            
        Returns:
            Transformed operator
        """
        if isinstance(operator, YawOperator):
            transformed_expr = self._transform_expr(operator._expr)
            result = YawOperator(transformed_expr, self.algebra)
            
            # Normalize if we have an algebra
            if self.algebra:
                return result.normalize()
            return result
        else:
            # Scalar
            return operator
    
    def _transform_expr(self, expr):
        """Recursively transform expression using QFT rules.
        
        Transformation rules:
            Z ↦ X
            X ↦ Z†
            I ↦ I
            AB ↦ (W >> A)(W >> B)  (automorphism)
            A + B ↦ (W >> A) + (W >> B)  (linearity)
            A + B â†¦ (W >> A) + (W >> B)  (linearity)
        """
        from sympy import symbols, Mul, Add, Pow
        
        # Base cases
        if expr == self.z_expr:
            return self.x_expr
        
        if expr == self.x_expr:
            return self.z_dag_expr
        
        # Daggers
        if isinstance(expr, Dagger):
            if expr.args[0] == self.z_expr:
                # Z† ↦ X†
                return self.x_dag_expr
            if expr.args[0] == self.x_expr:
                # X† ↦ Z
                return self.z_expr
        
        # Identity
        if self.algebra and expr == self.algebra.I._expr:
            return expr
        
        # Powers: X^n ↦ (Z†)^n, Z^n ↦ X^n
        if isinstance(expr, Pow):
            base = expr.args[0]
            exp = expr.args[1]
            
            if base == self.x_expr:
                return self.z_dag_expr ** exp
            if base == self.z_expr:
                return self.x_expr ** exp
            
            # Recurse on base
            return self._transform_expr(base) ** exp
        
        # Products: use automorphism
        if isinstance(expr, Mul):
            transformed_args = [self._transform_expr(arg) for arg in expr.args]
            return Mul(*transformed_args)
        
        # Sums: use linearity
        if isinstance(expr, Add):
            transformed_args = [self._transform_expr(arg) for arg in expr.args]
            return Add(*transformed_args)
        
        # Scalars and unknown expressions: pass through
        return expr
    
    def __rshift__(self, operator):
        """Enable W >> A syntax."""
        return self.conj_op(operator)
    
    def adjoint(self):
        """QFT is self-adjoint (up to phase): W† = W^(d-1)
        
        For qubits (d=2): W† = W
        For general qudits: W^d = I, so W† = W^(d-1)
        """
        if self.algebra and self.algebra.power_mod:
            d = self.algebra.power_mod
            if d == 2:
                return self  # Self-adjoint for qubits
            else:
                # Would need to implement QFT^n
                raise NotImplementedError("QFT^(d-1) for d>2 not yet implemented")
        return self  # Assume self-adjoint
    
    def __str__(self):
        return "qft"
    
    def __repr__(self):
        return f"QFT({self.gen_x}, {self.gen_z})"

def qft(gen_x, gen_z):
    """Create Quantum Fourier Transform operator.
    
    The QFT is defined algebraically by:
        qft(X, Z) >> Z = X
        qft(X, Z) >> X = Z†
    
    For qudits with X^d = Z^d = I, we have Z† = Z^(d-1).
    
    Args:
        gen_x: X generator (shift operator)
        gen_z: Z generator (clock operator)
        
    Returns:
        QFT operator
        
    Example:
        >>> # Qubit QFT (Hadamard)
        >>> alg = Algebra(['X', 'Z'], ['herm', 'unit', 'anti'])
        >>> W = qft(alg.X, alg.Z)
        >>> (W >> alg.Z).normalize()  # Returns X
        
        >>> # Qutrit QFT
        >>> alg3 = qudit(3)
        >>> W3 = qft(alg3.X, alg3.Z)
        >>> (W3 >> alg3.Z).normalize()  # Returns X
        >>> (W3 >> alg3.X).normalize()  # Returns Z^2
    """
    return QFT(gen_x, gen_z)

# ============================================================================
# PROJECTORS (MINIMAL POLYNOMIAL METHOD)
# ============================================================================

def proj_general(operator, eigenvalue, state=None, tolerance=1e-10):
    """Construct projector onto eigenspace using minimal polynomial.
    
    This implements the general recipe from spectral theory:
    Given an operator N with minimal polynomial p_min(x) = ∏ᵢ(x - λᵢ),
    the projector onto the eigenspace of λⱼ is:
    
        Πⱼ = pⱼ(N) / pⱼ(λⱼ)
    
    where pⱼ(x) = ∏ᵢ≠ⱼ(x - λᵢ) is the minimal polynomial with (x - λⱼ) factored out.
    
    This works because:
    - pⱼ(N)² = pⱼ(N) pⱼ(λⱼ) (from p_min(N) = 0)
    - Therefore Πⱼ² = Πⱼ (idempotence)
    - And N Πⱼ = λⱼ Πⱼ (eigenspace property)
    
    Args:
        operator: YawOperator to construct projector for
        eigenvalue: Eigenvalue to project onto
        state: Optional state for GNS construction (default: auto-detect)
        tolerance: Threshold for matching eigenvalues (default: 1e-10)
        
    Returns:
        YawOperator representing the projector
        
    Example:
        >>> alg = qubit()
        >>> X, Z = alg.X, alg.Z
        >>> P_plus = proj_general(X, 1.0)   # Project onto +1 eigenspace of X
        >>> P_minus = proj_general(X, -1.0) # Project onto -1 eigenspace of X
        >>> ~(P_plus + P_minus)  # Should give I
        >>> ~(P_plus * P_minus)  # Should give 0
    
    Notes:
        This is the algebraic/symbolic version. For numerical projectors,
        use proj() which returns a Projector object with built-in duality.
    """
    from sympy import symbols, expand, Poly
    
    # Get the minimal polynomial and eigenvalues
    poly, distinct_eigenvalues = minimal_poly(operator, state, tolerance)
    
    # Find which eigenvalue matches the target
    target_index = None
    for i, ev in enumerate(distinct_eigenvalues):
        # Check if eigenvalues match within tolerance
        if isinstance(eigenvalue, complex) or isinstance(ev, complex):
            diff = abs(complex(eigenvalue) - complex(ev))
        else:
            diff = abs(eigenvalue - ev)
        
        if diff < tolerance:
            target_index = i
            break
    
    if target_index is None:
        raise ValueError(
            f"Eigenvalue {eigenvalue} not found in spectrum {distinct_eigenvalues}. "
            f"Available eigenvalues: {distinct_eigenvalues}"
        )
    
    lambda_j = distinct_eigenvalues[target_index]
    
    # Construct pⱼ(x) = ∏ᵢ≠ⱼ(x - λᵢ)
    # This is the minimal polynomial with (x - λⱼ) factored out
    x = symbols('x')
    p_j_expr = 1
    for i, lambda_i in enumerate(distinct_eigenvalues):
        if i != target_index:
            p_j_expr *= (x - lambda_i)
    
    p_j_expr = expand(p_j_expr)
    
    # Evaluate pⱼ(λⱼ) (scalar)
    p_j_at_lambda = complex(p_j_expr.subs(x, lambda_j))
    
    if abs(p_j_at_lambda) < tolerance:
        raise ValueError(
            f"Denominator pⱼ(λⱼ) is zero - eigenvalue may have multiplicity > 1"
        )
    
    # Construct pⱼ(N) (operator)
    # Extract coefficients from polynomial
    p_j_poly = Poly(p_j_expr, x)
    coeffs = p_j_poly.all_coeffs()  # Highest degree first
    
    # Evaluate polynomial at operator N
    # pⱼ(N) = c₀N^d + c₁N^(d-1) + ... + c_d I
    N = operator
    I = operator.algebra.I if hasattr(operator, 'algebra') else None
    
    if I is None:
        raise ValueError("Operator must have an associated algebra with identity I")
    
    # Horner's method: pⱼ(N) = (...((c₀N + c₁)N + c₂)N + ... + c_d)
    p_j_N = I * coeffs[0]  # Start with c₀ * I
    for coeff in coeffs[1:]:
        p_j_N = p_j_N * N + I * coeff
    
    # Form projector: Πⱼ = pⱼ(N) / pⱼ(λⱼ)
    projector = p_j_N / p_j_at_lambda
    
    # Normalize (this should already be normalized due to the construction)
    return projector.normalize()


def proj(operator, k):
    """Create projector onto k-th eigenspace.
    
    Returns a Projector object that maintains duality with eigenstates.
    
    Args:
        operator: Operator with pow(d) relation
        k: Eigenspace index (0 to d-1)
        
    Returns:
        Projector object
        
    Example:
        >>> P0 = proj(Z, 0)
        >>> P0 | char(Z, 0)  # Returns 1.0
        >>> P0 | char(Z, 1)  # Returns 0.0
    """
    if not hasattr(operator, 'algebra') or operator.algebra is None:
        raise ValueError("Operator must have an associated algebra")
    
    algebra = operator.algebra
    
    if not hasattr(algebra, 'power_mod') or algebra.power_mod is None:
        raise ValueError("Operator must satisfy pow(d) relation")
    
    d = algebra.power_mod
    
    if k < 0 or k >= d:
        raise ValueError(f"Eigenspace index must be in range [0, {d-1}]")
    
    return Projector(operator, k, algebra)

def proj_algebraic(operator, k):
    """Convert projector to algebraic form (for compatibility).
    
    This expands projectors using minimal polynomials.
    """
    if not hasattr(operator, 'algebra') or operator.algebra is None:
        raise ValueError("Operator must have an associated algebra")
    
    algebra = operator.algebra
    
    if not hasattr(algebra, 'power_mod') or algebra.power_mod is None:
        raise ValueError("Operator must satisfy pow(d) relation")
    
    d = algebra.power_mod
    
    if k < 0 or k >= d:
        raise ValueError(f"Eigenspace index must be in range [0, {d-1}]")
    
    # Extract SymPy expressions
    I_expr = _get_sympy_expr(algebra.I)
    op_expr = _get_sympy_expr(operator)
    
    if d == 2:
        if k == 0:
            projector_expr = (I_expr + op_expr) / 2
        else:
            projector_expr = (I_expr - op_expr) / 2
        
        return YawOperator(projector_expr, algebra)
    
    # For d > 2: need braiding phase
    if algebra.braiding_phase is None:
        raise ValueError(f"Need braiding phase for d={d} > 2")
    
    omega = algebra.braiding_phase
    eigenvalue = omega**k
    
    # Minimal polynomial approach
    from sympy import prod
    numerator = prod(operator._expr - omega**j for j in range(d) if j != k)
    denominator = prod(eigenvalue - omega**j for j in range(d) if j != k)
    
    projector_expr = numerator / denominator
    projector_expr = projector_expr.expand()
    
    return YawOperator(projector_expr, algebra)

class Projector(YawOperator):
    """Projector onto eigenspace: proj(A, k)
    
    Special operator that knows its duality with eigenstates:
        proj(A, k) | char(A, j) = δ_{kj}
    """
    
    def __init__(self, operator, eigenspace_index, algebra=None):
        """Create projector.
        
        Args:
            operator: The operator whose eigenspace to project onto
            eigenspace_index: Which eigenspace (0 to d-1)
            algebra: Associated algebra (inherited from operator)
        """
        self.base_operator = operator
        self.eigenspace_index = eigenspace_index
        self._algebra = algebra or (operator.algebra if hasattr(operator, 'algebra') else None)
        
        # For compatibility with YawOperator interface, store symbolic expression
        # but we'll override the key methods
        # Use a symbolic representation
        from sympy import Symbol
        self._expr = Symbol(f"proj({operator}, {eigenspace_index})")
    
    @property
    def algebra(self):
        return self._algebra
    
    def expect(self, state, _depth=0):
        """Expectation value: ⟨state | proj | state⟩
        
        For eigenstates: proj(A, k) | char(A, j) = δ_{kj}
        """
        if isinstance(state, EigenState):
            # Check if state is an eigenstate of the same operator
            if state.operator == self.base_operator:
                # Orthogonality: proj(A, k) | char(A, j) = δ_{kj}
                if state.eigenspace_index == self.eigenspace_index:
                    return 1.0
                else:
                    return 0.0
            else:
                # Different operator - need to compute overlap
                # For now, fall back to algebraic computation
                return super().expect(state, _depth)
        else:
            return super().expect(state, _depth)
    
    def __lshift__(self, state):
        """Apply projector to state: proj << psi
        
        For eigenstates: proj(A, k) << char(A, j) = δ_{kj} char(A, j)
        """
        if isinstance(state, EigenState):
            # *** FIXED: Use state.observable and state.index ***
            if state.observable == self.base_operator:
                if state.index == self.eigenspace_index:
                    # Projector matches state: returns state
                    return state
                else:
                    # Projector annihilates state
                    raise ValueError(f"Projector proj({self.base_operator}, {self.eigenspace_index}) "
                                   f"annihilates char({state.observable}, {state.index})")
            else:
                # Different operator basis
                return TransformedState(state, self)
        else:
            return TransformedState(state, self)
    
    def expand(self):
        """Expand to algebraic form using minimal polynomial method.
        
        For qubits (d=2):
            proj(A, 0) = (I + A)/2
            proj(A, 1) = (I - A)/2
        
        For qudits (d>2):
            Uses minimal polynomial interpolation with braiding phase
        """
        return proj_algebraic(self.base_operator, self.eigenspace_index)
    
    # Keep old name for compatibility
    def to_algebraic(self):
        """Deprecated: use expand() instead."""
        return self.expand()
    
    @property
    def e(self):
        """Shorthand for .expand() - returns algebraic form.
        
        Usage: proj(Z, 0).e instead of proj(Z, 0).expand()
        
        Example:
            >>> CNOT = proj(Z, 0).e @ I + proj(Z, 1).e @ X
        """
        return self.expand()
    
    def adjoint(self):
        """Projectors are self-adjoint: P† = P"""
        return self
    
    def normalize(self, verbose=False):
        """Projectors are already in canonical form.
        
        Returns self to preserve Projector type (important for expectation values).
        """
        return self
    
    def __rmul__(self, other):
        """Right multiplication: other * proj
        
        Preserves Projector type when possible.
        """
        # Handle plain Python scalars
        if isinstance(other, (int, float, complex)):
            # Scalar multiplication - fall back to YawOperator
            from sympy import sympify
            return YawOperator(sympify(other) * self._expr, self._algebra)
        
        # If multiplying by identity, return self
        if hasattr(other, '_expr') and (str(other._expr) == 'I' or other._expr == 1):
            return self
        
        # If multiplying by a symbolic scalar, return scaled projector
        if hasattr(other, '_expr') and other._expr.is_number:
            # For now, fall back to YawOperator for scaled projectors
            return YawOperator(other._expr * self._expr, self._algebra)
        
        # Otherwise fall back to standard multiplication
        if hasattr(other, '_expr'):
            return YawOperator(other._expr * self._expr, self._algebra)
        else:
            # Unknown type - try converting to YawOperator
            from sympy import sympify
            return YawOperator(sympify(other) * self._expr, self._algebra)
    
    def __mul__(self, other):
        """Projector multiplication.
        
        Properties:
        - P² = P (idempotence)
        - PₖPⱼ = 0 if k ≠ j (orthogonality)
        - PₖPⱼ = Pₖ if k = j (idempotence again)
        """
        if isinstance(other, Projector):
            # Check if same base operator
            if self.base_operator == other.base_operator:
                if self.eigenspace_index == other.eigenspace_index:
                    # Same projector: P * P = P (idempotence)
                    return self
                else:
                    # Different projectors: Pₖ * Pⱼ = 0 (orthogonality)
                    # Return zero in the algebra
                    if self._algebra and hasattr(self._algebra, 'I'):
                        return self._algebra.I * 0
                    else:
                        return YawOperator(0, self._algebra)
            else:
                # Different operators - fall back to default multiplication
                # Expand both to algebraic form and multiply
                return self.expand() * other.expand()
        else:
            # Fall back to default YawOperator multiplication
            return super().__mul__(other)
    
    def __pow__(self, exponent):
        """Projector powers: P^n = P for n ≥ 1 (idempotence)"""
        if isinstance(exponent, int) and exponent >= 1:
            return self
        elif exponent == 0:
            # P^0 = I
            if self._algebra and hasattr(self._algebra, 'I'):
                return self._algebra.I
            else:
                return YawOperator(1, self._algebra)
        else:
            # Fractional or negative powers - expand and use default
            return self.expand() ** exponent
    
    def __str__(self):
        return f"proj({self.base_operator}, {self.eigenspace_index})"
    
    def __repr__(self):
        return f"Projector({self.base_operator}, {self.eigenspace_index})"

# ============================================================================
# CONTROLLED OPERATIONS
# ============================================================================

def ctrl_spectral(control_op, controlled_ops, state=None, tolerance=1e-10):
    """Create controlled operation using spectral decomposition.
    
    Given control operator N = ∑ᵢλᵢΠᵢ and operators [U₀, U₁, ..., Uₐ₋₁],
    constructs the controlled operation:
    
        CU = ∑ᵢ Πᵢ ⊗ Uᵢ
    
    This applies Uᵢ when the control is in the i-th eigenspace.
    
    This is more general than ctrl() as it works for ANY operator with
    a discrete spectrum, not just those with pow(d) relations.
    
    Args:
        control_op: Control operator (any operator with discrete spectrum)
        controlled_ops: List of operators to apply conditionally.
                       Length must match number of distinct eigenvalues.
        state: Optional state for GNS construction (default: auto-detect)
        tolerance: Eigenvalue matching tolerance (default: 1e-10)
        
    Returns:
        Controlled operation as sum of tensor products
        
    Example:
        >>> # Standard CNOT using spectral decomposition
        >>> CNOT = ctrl_spectral(Z, [I, X])
        
        >>> # Control with arbitrary Hermitian operator
        >>> H = (X + Z) / sqrt(2)
        >>> eigenvalues = spec(H)  # [1, -1]
        >>> # Apply X when H has eigenvalue +1, Y when H has eigenvalue -1
        >>> custom_ctrl = ctrl_spectral(H, [X, Y])
        
        >>> # For operators with non-standard spectra
        >>> A = X + 0.5*Z  # eigenvalues: [±√1.25]
        >>> ctrl_A = ctrl_spectral(A, [U0, U1])
    
    Notes:
        - Automatically uses proj_general() to construct projectors
        - Works for any operator, not just pow(d)
        - Eigenvalues are ordered by spec() (descending real part)
    """
    # Get spectrum to determine number of eigenspaces
    eigenvalues = spec(control_op, state, tolerance)
    
    # Check dimension match
    num_eigenspaces = len(eigenvalues)
    if len(controlled_ops) != num_eigenspaces:
        raise ValueError(
            f"Expected {num_eigenspaces} controlled operators for control "
            f"operator with {num_eigenspaces} distinct eigenvalues, "
            f"got {len(controlled_ops)}. Eigenvalues: {eigenvalues}"
        )
    
    # Build sum: ∑ᵢ Πᵢ ⊗ Uᵢ
    terms = []
    
    for i, eigenvalue in enumerate(eigenvalues):
        # Construct projector onto i-th eigenspace
        projector = proj_general(control_op, eigenvalue, state, tolerance)
        
        # Create tensor product: Πᵢ ⊗ Uᵢ
        term = projector @ controlled_ops[i]
        terms.append(term)
    
    # Return sum (TensorSum auto-normalizes)
    if len(terms) == 1:
        return terms[0]
    else:
        return TensorSum(terms)


def ctrl(control_op, controlled_ops):
    """Create controlled operation using projector decomposition.
    
    For control operator A with pow(d), constructs:
        Σ_k proj(A, k) ⊗ controlled_ops[k]
    
    This implements conditional application: if A is in eigenspace k,
    apply controlled_ops[k] to the target system.
    
    Args:
        control_op: Control operator (must have pow(d) relation)
        controlled_ops: List of operators to apply conditionally
                       Length must match dimension d
        
    Returns:
        Controlled operation as sum of tensor products
        
    Example:
        >>> CNOT = ctrl(Z, [I, X])
        >>> CCNOT = ctrl(Z, [I@I, CNOT])  # Nested control
    """
    # Check if control operator has algebra with pow(d)
    if not hasattr(control_op, 'algebra') or control_op.algebra is None:
        raise ValueError("Control operator must have an associated algebra")
    
    algebra = control_op.algebra
    
    if not hasattr(algebra, 'power_mod') or algebra.power_mod is None:
        raise ValueError("Control operator must satisfy pow(d) relation")
    
    d = algebra.power_mod
    
    # Check dimension match
    if len(controlled_ops) != d:
        raise ValueError(f"Expected {d} controlled operators for dimension-{d} "
                        f"control, got {len(controlled_ops)}")
    
    # Build sum: Σ_k proj(control, k) ⊗ controlled_ops[k]
    result = None
    
    for k in range(d):
        # Get projector for k-th eigenspace
        projector = proj(control_op, k).e
        
        # *** USE @ OPERATOR instead of tensor() function ***
        # This handles TensorSum distribution automatically
        term = projector @ controlled_ops[k]
        
        # Add to sum
        if result is None:
            result = term
        else:
            result = result + term
    
    return result

def ctrl_single(control_op, k, target_op):
    """Create controlled operation that acts only for specific control value.
    
    Applies target_op when control is in eigenspace k, identity otherwise.
    
    Constructs:
        proj(control, k) ⊗ target_op + Σ_{j≠k} proj(control, j) ⊗ I
    
    Args:
        control_op: Control operator
        k: Control eigenspace index (0 to d-1)
        target_op: Operation to apply when control is in state k
        
    Returns:
        Controlled operation
        
    Example:
        >>> # Apply X only when Z is in |1⟩ state
        >>> CX_on_1 = ctrl_single(Z, 1, X)
    """
    if not hasattr(control_op, 'algebra') or control_op.algebra is None:
        raise ValueError("Control operator must have an associated algebra")
    
    algebra = control_op.algebra
    d = algebra.power_mod
    
    if k < 0 or k >= d:
        raise ValueError(f"Control index k must be in range [0, {d-1}]")
    
    # Get identity for target system (infer from target_op)
    if hasattr(target_op, 'algebra') and target_op.algebra is not None:
        target_I = target_op.algebra.I
    else:
        # Assume target_op has same structure as control
        target_I = algebra.I
    
    # Build controlled ops list: I everywhere except position k
    controlled_ops = [target_I] * d
    controlled_ops[k] = target_op
    
    return ctrl(control_op, controlled_ops)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def tensor(*factors):
    """Create tensor product of operators or states.
    
    For operators: A ⊗ B
    For states: |ψ⟩ ⊗ |φ⟩
    Mixed tensors not currently supported.
    
    Args:
        *factors: Operators or States to tensor together
        
    Returns:
        TensorProduct (for operators) or TensorState (for states)
    """
    if not factors:
        raise ValueError("tensor() requires at least one argument")
    
    # Check if all factors are states
    all_states = all(isinstance(f, State) for f in factors)
    all_operators = all(isinstance(f, YawOperator) for f in factors)
    
    if all_states:
        # Tensor product of states
        if len(factors) == 1:
            return factors[0]
        return TensorState(list(factors))
    
    elif all_operators:
        # Tensor product of operators
        if len(factors) == 1:
            return factors[0]
        
        result = TensorProduct(list(factors))
        
        # *** REMOVED: Don't call normalize on tensor products ***
        # Tensor products may not have a well-defined algebra to normalize against
        return result
    
    else:
        raise TypeError("Cannot mix operators and states in tensor product")
    
def tensor_power(op, n):
    """Create n-fold tensor product: tensor_power(X, 3) = X⊗X⊗X.
    
    Args:
        op: Operator to repeat
        n: Number of copies
        
    Returns:
        TensorProduct instance
    """
    return TensorProduct(*[op for _ in range(n)])

def char(observable: YawOperator, index: int) -> State:
    """Create eigenstate (characteristic state) of observable.
    
    The eigenstate corresponds to the eigenvalue at position `index` in the
    spectrum returned by spec(observable), which lists eigenvalues in 
    descending order.
    
    Args:
        observable: Operator to be eigenstate of
        index: Position in descending spectrum (0 = largest eigenvalue)
               For example, if spec(X) = [1.0, -1.0], then:
               - char(X, 0) is the +1 eigenstate
               - char(X, 1) is the -1 eigenstate
        
    Returns:
        EigenState instance with eigenvalue = spec(observable)[index]
        
    Raises:
        ValueError: If observable has no associated algebra
        ValueError: If index is out of range for the spectrum
        
    Example:
        >>> alg = qubit()
        >>> spec(alg.Z)  # [1.0, -1.0]
        >>> psi_0 = char(alg.Z, 0)  # |0⟩ state (eigenvalue +1)
        >>> psi_1 = char(alg.Z, 1)  # |1⟩ state (eigenvalue -1)
        >>> psi_0.expect(alg.Z)  # Returns 1.0
        >>> psi_1.expect(alg.Z)  # Returns -1.0
    """
    if not observable.algebra:
        raise ValueError("Observable must be associated with an algebra")
    return EigenState(observable, index, observable.algebra)


def mixed(probabilities_and_states):
    """Create mixed state.
    
    Args:
        probabilities_and_states: List of (prob, state) tuples
    
    Example:
        >>> rho = mixed([(0.7, psi0), (0.3, psi1)])
    """
    return MixedState(probabilities_and_states)

def conj_op(U, A):
    """Operator conjugation: U >> A = U† A U.
    
    Transforms operator A by unitary U.
    
    Args:
        U: Unitary operator
        A: Operator to transform
        
    Returns:
        Conjugated operator
    """
    return U.conj_op(A)

def conj_state(U, state):
    """State conjugation: U << |ψ⟩.
    
    Transforms state by unitary U.
    
    Args:
        U: Unitary operator
        state: State to transform
        
    Returns:
        ConjugatedState instance
    """
    return U.conj_state(state)

def _get_sympy_expr(obj):
    """Extract SymPy expression from YawOperator or return as-is."""
    if hasattr(obj, '_expr'):
        return obj._expr
    elif isinstance(obj, Symbol):
        return obj
    else:
        return obj

# ============================================================================
# GNS CONSTRUCTION - Gelfand-Naimark-Segal Map
# ============================================================================

def _get_operator_basis(algebra):
    """Get a basis of operators for the algebra.
    
    For a qubit (d=2): Returns [I, X, Y, Z]
    For a qutrit (d=3): Returns [I, X, Z, X^2, XZ, ZX, Z^2, X^2Z, XZ^2]
    
    Args:
        algebra: Algebra instance
        
    Returns:
        List of YawOperator instances forming a basis
    """
    if not hasattr(algebra, 'power_mod'):
        raise ValueError("Algebra must have power_mod attribute")
    
    d = algebra.power_mod
    dim = d * d  # Dimension of operator space
    
    # Get generators
    if not hasattr(algebra, 'X') or not hasattr(algebra, 'Z'):
        raise ValueError("Algebra must have X and Z generators")
    
    X = algebra.X
    Z = algebra.Z
    I = algebra.I
    
    # Build basis by enumerating X^a Z^b for a,b < d
    basis = []
    for b in range(d):  # Z power
        for a in range(d):  # X power
            if a == 0 and b == 0:
                op = I
            elif a == 0:
                op = Z ** b
            elif b == 0:
                op = X ** a
            else:
                op = (X ** a) * (Z ** b)
            
            basis.append(op.normalize())
    
    return basis

def _compute_gram_matrix(state, basis):
    """Compute Gram matrix G[i,j] = <B_i, B_j> = state.expect(B_i† B_j).
    
    Args:
        state: State instance (e.g., char(Z, 0))
        basis: List of operators
        
    Returns:
        numpy array of shape (n, n)
    """
    n = len(basis)
    G = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        for j in range(n):
            # Inner product: <B_i, B_j> = state.expect(B_i† B_j)
            # For Pauli operators, B_i† = B_i (they're Hermitian)
            # More generally, we need adjoint
            B_i_dag = basis[i].adjoint()
            product = (B_i_dag * basis[j]).normalize()
            
            # Compute expectation value
            expectation = state.expect(product)
            
            # Convert to complex number
            if hasattr(expectation, 'evalf'):
                expectation = complex(expectation.evalf())
            else:
                expectation = complex(expectation)
            
            G[i, j] = expectation
    
    return G

def _orthonormalize_basis(basis, gram_matrix, threshold=1e-10):
    """Orthonormalize basis using modified Gram-Schmidt on Gram matrix.
    
    Args:
        basis: List of operators
        gram_matrix: Gram matrix G[i,j] = <basis[i], basis[j]>
        threshold: Threshold for considering vectors as zero-norm
        
    Returns:
        Tuple (orthonormal_basis, coefficients)
        - orthonormal_basis: List of operators forming orthonormal basis
        - coefficients: Matrix C where orthonormal_basis[i] = sum_j C[i,j] basis[j]
    """
    n = len(basis)
    G = gram_matrix.copy()
    
    # We'll track which original basis vectors we keep
    # and what linear combinations they become
    kept_indices = []
    coefficients = []
    
    for i in range(n):
        # Compute norm squared: <v_i, v_i>
        norm_sq = G[i, i].real
        
        if norm_sq < threshold:
            # Skip zero-norm vectors
            continue
        
        # Normalize
        norm = np.sqrt(norm_sq)
        coeff = np.zeros(n, dtype=complex)
        coeff[i] = 1.0 / norm
        coefficients.append(coeff)
        kept_indices.append(i)
        
        # Gram-Schmidt: subtract projection from remaining vectors
        for j in range(i + 1, n):
            # <v_i, v_j> after normalization
            overlap = G[i, j] / norm
            
            # Update Gram matrix for v_j
            for k in range(j, n):
                G[j, k] -= overlap * G[i, k] / norm
                if k != j:
                    G[k, j] = G[j, k].conj()
    
    # Build orthonormal basis from coefficients
    orthonormal_basis = []
    for coeff in coefficients:
        # Construct linear combination: sum_j coeff[j] * basis[j]
        op = None
        for j in range(n):
            if abs(coeff[j]) > threshold:
                term = basis[j] * complex(coeff[j])
                if op is None:
                    op = term
                else:
                    op = op + term
        
        if op is not None:
            orthonormal_basis.append(op.normalize())
    
    # Convert coefficients to matrix
    if coefficients:
        C = np.array(coefficients)
    else:
        C = np.array([]).reshape(0, n)
    
    return orthonormal_basis, C

def gnsVec(state, op):
    """Convert operator to Hilbert space vector via GNS construction.
    
    The GNS construction maps operators to vectors in a Hilbert space:
        A ↦ |A⟩
    
    with inner product ⟨A|B⟩ = state.expect(A† B).
    
    Supports tensor products:
        gnsVec(ψ₁ ⊗ ψ₂, A ⊗ B) = gnsVec(ψ₁, A) ⊗ gnsVec(ψ₂, B)
    
    Args:
        state: State instance (e.g., char(Z, 0)) or TensorState
        op: YawOperator or TensorProduct to convert to vector
        
    Returns:
        numpy array representing the vector in the GNS Hilbert space
        
    Example:
        >>> pauli = Algebra(gens=['X', 'Z'], rels=['herm', 'unit', 'anti', 'pow(2)'])
        >>> psi0 = char(pauli.Z, 0)
        >>> vec_I = gnsVec(psi0, pauli.I)  # |0⟩ state
        >>> vec_X = gnsVec(psi0, pauli.X)  # |1⟩ state
        >>> 
        >>> # Tensor products
        >>> psi_00 = TensorState([psi0, psi0])
        >>> X1 = tensor(pauli.X, pauli.I)
        >>> vec_X1 = gnsVec(psi_00, X1)  # Kronecker product
    """
    # Handle tensor products
    if isinstance(state, TensorState) and isinstance(op, TensorProduct):
        if len(state.states) != len(op.factors):
            raise ValueError(
                f"State has {len(state.states)} factors but "
                f"operator has {len(op.factors)} factors"
            )
        
        # Compute GNS vector for each component
        component_vecs = []
        for s, o in zip(state.states, op.factors):
            vec = gnsVec(s, o)
            component_vecs.append(vec)
        
        # Kronecker product of all components
        result = component_vecs[0]
        for vec in component_vecs[1:]:
            result = np.kron(result, vec)
        
        return result
    
    # Single subsystem case
    # Get algebra from operator or state
    if hasattr(op, 'algebra') and op.algebra is not None:
        algebra = op.algebra
    elif hasattr(state, 'algebra') and state.algebra is not None:
        algebra = state.algebra
    else:
        raise ValueError("Cannot determine algebra from operator or state")
    
    # Get operator basis
    basis = _get_operator_basis(algebra)
    
    # Compute Gram matrix
    G = _compute_gram_matrix(state, basis)
    
    # Orthonormalize
    ortho_basis, C = _orthonormalize_basis(basis, G)
    
    # Express op in terms of orthonormal basis via direct inner products
    # alpha[i] = ⟨ortho_basis[i], op⟩_GNS = state.expect(ortho_basis[i]† * op)
    # This correctly handles operators that are linearly dependent in the GNS space
    
    if len(ortho_basis) == 0:
        return np.array([])
    
    alpha = np.zeros(len(ortho_basis), dtype=complex)
    op_normalized = op.normalize()
    
    for i, e_i in enumerate(ortho_basis):
        # Compute inner product ⟨e_i, op⟩ = state.expect(e_i† * op)
        e_i_dag = e_i.adjoint()
        product = (e_i_dag * op_normalized).normalize()
        
        # Compute expectation value
        expectation = state.expect(product)
        
        # Convert to complex number
        if hasattr(expectation, 'evalf'):
            expectation = complex(expectation.evalf())
        else:
            expectation = complex(expectation)
        
        alpha[i] = expectation
    
    return alpha

def gnsMat(state, op):
    """Convert operator to matrix acting on GNS Hilbert space.
    
    The GNS construction represents operators as matrices:
        M[i,j] = ⟨e_i| O |e_j⟩ = state.expect(e_i† O e_j)
    
    where {e_i} is the orthonormal operator basis.
    
    Supports tensor products:
        gnsMat(ψ₁ ⊗ ψ₂, A ⊗ B) = gnsMat(ψ₁, A) ⊗ gnsMat(ψ₂, B)
    
    where ⊗ on the right is the Kronecker product of matrices.
    
    Args:
        state: State instance (e.g., char(Z, 0)) or TensorState
        op: YawOperator or TensorProduct to convert to matrix
        
    Returns:
        numpy array representing the matrix in the GNS Hilbert space
        
    Example:
        >>> pauli = Algebra(gens=['X', 'Z'], rels=['herm', 'unit', 'anti', 'pow(2)'])
        >>> psi0 = char(pauli.Z, 0)
        >>> mat_X = gnsMat(psi0, pauli.X)  # Pauli X matrix
        >>> mat_Z = gnsMat(psi0, pauli.Z)  # Pauli Z matrix
        >>> 
        >>> # Tensor products
        >>> psi_00 = TensorState([psi0, psi0])
        >>> XX = tensor(pauli.X, pauli.X)
        >>> mat_XX = gnsMat(psi_00, XX)  # 4×4 matrix
    """
    # Handle tensor products
    if isinstance(state, TensorState) and isinstance(op, TensorProduct):
        if len(state.states) != len(op.factors):
            raise ValueError(
                f"State has {len(state.states)} factors but "
                f"operator has {len(op.factors)} factors"
            )
        
        # Compute GNS matrix for each component
        component_mats = []
        for s, o in zip(state.states, op.factors):
            mat = gnsMat(s, o)
            component_mats.append(mat)
        
        # Kronecker product of all components
        result = component_mats[0]
        for mat in component_mats[1:]:
            result = np.kron(result, mat)
        
        return result
    
    # Single subsystem case
    # Get algebra
    if hasattr(op, 'algebra') and op.algebra is not None:
        algebra = op.algebra
    elif hasattr(state, 'algebra') and state.algebra is not None:
        algebra = state.algebra
    else:
        raise ValueError("Cannot determine algebra from operator or state")
    
    # Get operator basis
    basis = _get_operator_basis(algebra)
    
    # Compute Gram matrix
    G = _compute_gram_matrix(state, basis)
    
    # Orthonormalize
    ortho_basis, C = _orthonormalize_basis(basis, G)
    
    n = len(ortho_basis)
    if n == 0:
        return np.array([]).reshape(0, 0)
    
    # Compute matrix elements: M[i,j] = <e_i| O |e_j>
    #                                  = state.expect(e_i† O e_j)
    M = np.zeros((n, n), dtype=complex)
    
    for i in range(n):
        e_i_dag = ortho_basis[i].adjoint()
        for j in range(n):
            # Compute e_i† O e_j
            product = (e_i_dag * op * ortho_basis[j]).normalize()
            
            # Compute expectation value
            expectation = state.expect(product)
            
            # Convert to complex number
            if hasattr(expectation, 'evalf'):
                expectation = complex(expectation.evalf())
            else:
                expectation = complex(expectation)
            
            M[i, j] = expectation
    
    return M

def _express_in_basis(op, basis, algebra):
    """Express operator as linear combination of basis elements.
    
    Args:
        op: YawOperator to express
        basis: List of basis operators
        algebra: Algebra instance
        
    Returns:
        numpy array of coefficients
    """
    from sympy import symbols, Symbol, expand, Add, Mul
    from sympy.core.numbers import Number as SympyNumber
    
    # Normalize operator
    op_norm = op.normalize()
    op_expr = op_norm._expr
    
    # Initialize coefficients
    coeffs = np.zeros(len(basis), dtype=complex)
    
    # Try to match each basis element
    # This is a simple pattern matching approach
    # For more complex expressions, would need symbolic manipulation
    
    # Expand the expression
    expanded = expand(op_expr)
    
    # If it's a sum, extract terms
    if isinstance(expanded, Add):
        terms = expanded.args
    else:
        terms = [expanded]
    
    # For each term, try to match against basis
    for term in terms:
        # Extract coefficient and operator part
        coeff = 1
        op_part = term
        
        if isinstance(term, Mul):
            # Separate numerical coefficient from operators
            coeff_factors = []
            op_factors = []
            
            for factor in term.args:
                if isinstance(factor, (SympyNumber, int, float, complex)):
                    coeff_factors.append(factor)
                else:
                    op_factors.append(factor)
            
            if coeff_factors:
                from sympy import Mul as SympyMul
                coeff = SympyMul(*coeff_factors)
            
            if op_factors:
                from sympy import Mul as SympyMul
                op_part = SympyMul(*op_factors)
            else:
                op_part = 1
        
        elif isinstance(term, (SympyNumber, int, float, complex)):
            coeff = term
            op_part = 1
        
        # Convert coefficient to complex
        if hasattr(coeff, 'evalf'):
            coeff = complex(coeff.evalf())
        else:
            coeff = complex(coeff)
        
        # Try to match op_part against each basis element
        for i, basis_elem in enumerate(basis):
            basis_expr = basis_elem._expr
            
            # Simple equality check
            if expand(op_part - basis_expr) == 0:
                coeffs[i] += coeff
                break
    
    return coeffs

def _create_bootstrap_eigenstate(observable, index, algebra):
    """Create an eigenstate without calling spec() (avoids circular dependency).
    
    This is used internally by spec() to create a default state for the
    GNS construction. It uses a simple heuristic for the eigenvalue
    (1.0 for index 0, -1.0 otherwise) which is sufficient for computing
    the spectrum via gnsMat.
    
    Args:
        observable: Operator to be eigenstate of
        index: Index (0 or 1 typically)
        algebra: Associated algebra
        
    Returns:
        EigenState with bootstrap eigenvalue
    """
    # Create an EigenState but bypass the __init__ that calls spec()
    state = object.__new__(EigenState)
    state.observable = observable
    state.index = index
    state.algebra = algebra
    # Use bootstrap eigenvalue (doesn't need to be exact for GNS construction)
    state.eigenvalue = 1.0 if index == 0 else -1.0
    return state

def spec(op, state=None, tolerance=1e-10):
    """Compute spectrum (eigenvalues) of an operator in descending order.
    
    Returns eigenvalues sorted in descending order by:
    1. Real part (primary)
    2. Imaginary part (secondary, if real parts are equal within tolerance)
    
    The function also computes the minimal polynomial by factoring:
        p(x) = (x - λ₁)(x - λ₂)...(x - λₖ)
    where λᵢ are the distinct eigenvalues.
    
    Args:
        op: YawOperator to analyze
        state: State for GNS representation (optional)
               If None, attempts to use char(first_gen, 0) or char(first_gen, 1)
        tolerance: Threshold for considering eigenvalues as distinct (default 1e-10)
        
    Returns:
        List of eigenvalues in descending order
        
    Example:
        >>> alg = qubit()
        >>> spec(alg.X)  # Returns [1.0, -1.0]
        >>> spec(alg.Z)  # Returns [1.0, -1.0]
        >>> spec(alg.I)  # Returns [1.0, 1.0]
        >>> 
        >>> # With explicit state
        >>> psi0 = char(alg.Z, 0)
        >>> spec(alg.X, psi0)  # Returns [1.0, -1.0]
    
    Notes:
        The minimal polynomial has degree equal to the number of distinct 
        eigenvalues. For hermitian operators (like Pauli matrices), all
        eigenvalues are real.
    """
    from sympy import Poly, symbols
    
    # Get algebra from operator
    if not hasattr(op, 'algebra') or op.algebra is None:
        raise ValueError("Operator must have an associated algebra")
    
    algebra = op.algebra
    
    # If no state provided, create a default one
    if state is None:
        # Try to get generators from the algebra (excluding identity)
        gens = _get_operator_basis(algebra)
        if not gens:
            raise ValueError("Cannot create default state: algebra has no generators")
        
        # Skip identity operator - look for a non-trivial generator
        # Prefer generators named X, Z, or similar
        first_gen = None
        for gen in gens:
            gen_str = str(gen)
            # Skip identity
            if gen_str == 'I':
                continue
            # Prefer single-letter generators (X, Z, etc)
            if len(gen_str) == 1:
                first_gen = gen
                break
        
        # If no single-letter generator, use any non-identity generator
        if first_gen is None:
            for gen in gens:
                if str(gen) != 'I':
                    first_gen = gen
                    break
        
        # If still None, fall back to first generator (even if identity)
        if first_gen is None:
            first_gen = gens[0]
        
        # Create a bootstrap eigenstate without calling spec()
        # to avoid circular dependency
        state = _create_bootstrap_eigenstate(first_gen, 0, algebra)
    
    # Convert operator to matrix using GNS construction
    M = gnsMat(state, op)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(M)
    
    # Sort eigenvalues in descending order
    # Primary sort: real part (descending)
    # Secondary sort: imaginary part (descending)
    eigenvalues_sorted = sorted(
        eigenvalues,
        key=lambda x: (-x.real, -x.imag)
    )
    
    # Convert to Python floats/complex and clean numerical noise
    result = []
    for ev in eigenvalues_sorted:
        # Convert to complex, then clean
        cleaned = _clean_number(complex(ev))
        result.append(cleaned)
    
    return result

def minimal_poly(op, state=None, tolerance=1e-10):
    """Compute minimal polynomial of an operator.
    
    The minimal polynomial is the monic polynomial of smallest degree that
    annihilates the operator: p(A) = 0.
    
    For diagonalizable operators, the minimal polynomial is:
        p(x) = (x - λ₁)(x - λ₂)...(x - λₖ)
    where λᵢ are the distinct eigenvalues.
    
    Args:
        op: YawOperator to analyze
        state: State for GNS representation (optional)
        tolerance: Threshold for considering eigenvalues as distinct
        
    Returns:
        Tuple (polynomial, roots) where:
        - polynomial: SymPy Poly object representing the minimal polynomial
        - roots: List of distinct eigenvalues (roots of minimal polynomial)
        
    Example:
        >>> alg = qubit()
        >>> poly, roots = minimal_poly(alg.X)
        >>> # poly is (x - 1)(x + 1) = x² - 1
        >>> # roots is [1.0, -1.0]
    """
    from sympy import Poly, symbols, expand
    
    # Get spectrum
    eigenvalues = spec(op, state, tolerance)
    
    # Find distinct eigenvalues (roots of minimal polynomial)
    distinct_eigenvalues = []
    for ev in eigenvalues:
        # Check if this eigenvalue is already in the list
        is_duplicate = False
        for existing in distinct_eigenvalues:
            if isinstance(ev, complex) or isinstance(existing, complex):
                diff = abs(ev - existing)
            else:
                diff = abs(ev - existing)
            
            if diff < tolerance:
                is_duplicate = True
                break
        
        if not is_duplicate:
            distinct_eigenvalues.append(ev)
    
    # Construct minimal polynomial: p(x) = (x - λ₁)(x - λ₂)...(x - λₖ)
    x = symbols('x')
    poly_expr = 1
    for ev in distinct_eigenvalues:
        poly_expr *= (x - ev)
    
    poly_expr = expand(poly_expr)
    poly = Poly(poly_expr, x)
    
    return poly, distinct_eigenvalues

# Convenience constructor for sum states
def sum_state(*states):
    """Create a sum of state functionals.
    
    Args:
        *states: State functionals to sum
        
    Returns:
        SumState instance
        
    Example:
        >>> psi_00 = char(Z, 0) @ char(Z, 0)
        >>> X_X = X @ X
        >>> phi = sum_state(psi_00, X_X.lmul(psi_00)) / sqrt(2)
    """
    return SumState(list(states))
