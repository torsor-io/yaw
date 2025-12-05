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

__all__ = [
    'Algebra', 'YawOperator', 'State', 'Context',
    'char', 'proj', 'qft', 'ctrl', 'tensor',
    'opChannel', 'stChannel', 'opMeasure', 'stMeasure',
    'opBranches', 'stBranches', 'Encoding'
]
