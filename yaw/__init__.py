"""
yaw: Quantum Programming as Algebra

Algebraic quantum computing based on C*-algebraic foundations.
The syntax is the algebra itself.
"""

__version__ = "0.1.0"
__author__ = "David Wakeham"
__license__ = "Apache-2.0"

# Import core functionality
from .yaw_prototype import *

# Explicitly list main exports for clarity
__all__ = [
    # Core classes
    'YawOperator', 'TensorProduct', 'Algebra', 'Context', 'Projector',
    
    # States
    'State', 'EigenState', 'TensorState', 'ConjugatedState',
    'TransformedState', 'CollapsedState', 'TensorSum',
    
    # Channels
    'OpChannel', 'opChannel', 'StChannel', 'stChannel',
    
    # Measurement
    'OpMeasurement', 'opMeasure', 'StMeasurement', 'stMeasure',
    'OpBranches', 'opBranches', 'StBranches', 'stBranches',
    'compose_st_branches', 'compose_op_branches',
    
    # Operators and utilities
    'tensor', 'tensor_power', 'char', 'conj_op', 'conj_state',
    'QFT', 'qft', 'proj', 'proj_algebraic', 'ctrl', 'ctrl_single',
    'comm', 'acomm',
    
    # Algebras
    'qudit', 'qubit',
    
    # QEC
    'Encoding', 'rep', 'StabilizerCode', 'five_qubit_code', 'bit_flip_code',
    
    # GNS construction (Hilbert space representations)
    'gnsVec', 'gnsMat'
]
