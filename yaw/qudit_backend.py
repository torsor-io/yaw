#!/usr/bin/env python3
"""
Numerical Qudit Backend using Clock and Shift Matrices

Based on: https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices
Section: "Construction: The clock and shift matrices"

This provides explicit matrix representations for qudits (d-level systems)
using Sylvester's generalized Pauli matrices.
"""

import numpy as np
from typing import Optional, List, Tuple
import sys
sys.path.insert(0, '/mnt/project')


class QuditBackend:
    """Numerical backend for d-level quantum systems using matrix representation.
    
    Uses the clock and shift matrices from Sylvester's construction:
    - X (shift): X|j⟩ = |j+1 mod d⟩  (cyclic permutation)
    - Z (clock): Z|j⟩ = ω^j|j⟩  where ω = exp(2πi/d)
    
    These satisfy the Weyl commutation relation: XZ = ωZX
    """
    
    def __init__(self, d: int):
        """Initialize qudit backend.
        
        Args:
            d: Dimension of the system (number of levels)
        """
        self.d = d
        self.omega = np.exp(2j * np.pi / d)
        
        # Generate shift matrix X
        self.X = self._shift_matrix()
        
        # Generate clock matrix Z
        self.Z = self._clock_matrix()
        
        # Identity
        self.I = np.eye(d, dtype=complex)
        
        # Fourier matrix (generalized Hadamard)
        self.H = self._fourier_matrix()
    
    def _shift_matrix(self) -> np.ndarray:
        """Shift operator: X|j⟩ = |j+1 mod d⟩
        
        Matrix form: permutation matrix that shifts basis states cyclically.
        """
        X = np.zeros((self.d, self.d), dtype=complex)
        for j in range(self.d):
            X[j, (j+1) % self.d] = 1
        return X
    
    def _clock_matrix(self) -> np.ndarray:
        """Clock operator: Z|j⟩ = ω^j|j⟩ where ω = exp(2πi/d)
        
        Matrix form: diagonal matrix with powers of ω.
        """
        return np.diag([self.omega**j for j in range(self.d)])
    
    def _fourier_matrix(self) -> np.ndarray:
        """Discrete Fourier transform matrix (generalized Hadamard).
        
        H_{jk} = (1/√d) ω^(jk)
        
        This transforms between X and Z eigenbases.
        """
        H = np.zeros((self.d, self.d), dtype=complex)
        for j in range(self.d):
            for k in range(self.d):
                H[j, k] = self.omega**(j*k) / np.sqrt(self.d)
        return H
    
    def projector(self, operator: str, index: int) -> np.ndarray:
        """Create projector onto eigenspace.
        
        Args:
            operator: 'X' or 'Z'
            index: Eigenspace index (0 to d-1)
            
        Returns:
            Projector matrix |λ_k⟩⟨λ_k|
        """
        if operator == 'X':
            # X eigenvectors are columns of H (Fourier matrix)
            eigvec = self.H[:, index]
        elif operator == 'Z':
            # Z eigenvectors are standard basis |j⟩
            eigvec = np.zeros(self.d, dtype=complex)
            eigvec[index] = 1
        else:
            raise ValueError(f"Unknown operator: {operator}")
        
        # Projector: |ψ⟩⟨ψ|
        return np.outer(eigvec, eigvec.conj())
    
    def reflect(self, projector: np.ndarray) -> np.ndarray:
        """Reflection operator: R = I - 2P"""
        return self.I - 2 * projector
    
    def apply_operator(self, operator: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Apply operator to state: |ψ'⟩ = U|ψ⟩"""
        return operator @ state
    
    def measure(self, state: np.ndarray, projectors: List[np.ndarray], 
                seed: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Measure state using projective measurement.
        
        Args:
            state: State vector
            projectors: List of projector matrices
            seed: Random seed
            
        Returns:
            (collapsed_state, outcome_index)
        """
        # Compute probabilities
        probs = []
        for P in projectors:
            # p_k = ⟨ψ|P_k|ψ⟩
            prob = np.real(state.conj() @ P @ state)
            probs.append(max(0.0, prob))  # Ensure non-negative
        
        # Normalize
        total = sum(probs)
        if total < 1e-10:
            raise ValueError("All measurement outcomes have zero probability")
        probs = [p / total for p in probs]
        
        # Sample outcome
        if seed is not None:
            np.random.seed(seed)
        outcome = np.random.choice(len(projectors), p=probs)
        
        # Collapse state
        P_outcome = projectors[outcome]
        collapsed = P_outcome @ state
        collapsed = collapsed / np.linalg.norm(collapsed)
        
        return collapsed, outcome
    
    def grover(self, mark: int, seed: Optional[int] = None, verbose: bool = False) -> int:
        """Grover's algorithm on d-level system.
        
        Args:
            mark: Index of marked element (0 to d-1)
            seed: Random seed
            verbose: Print debug info
            
        Returns:
            Measured outcome (guess)
        """
        # Oracle: flips phase of marked state
        P_mark = self.projector('Z', mark)
        Oracle = self.reflect(P_mark)
        
        # Diffusion: reflects about uniform superposition
        P_uniform = self.projector('X', 0)  # |+⟩ state
        Diffusion = -self.reflect(P_uniform)
        
        # Grover operator
        G = Diffusion @ Oracle
        
        # Optimal iterations
        r = max(1, int((np.pi/4) * np.sqrt(self.d)))
        
        if verbose:
            print(f"Grover search:")
            print(f"  Dimension: d = {self.d}")
            print(f"  Marked element: |{mark}⟩")
            print(f"  Iterations: r = {r}")
        
        # Initial state: uniform superposition (X eigenstate)
        psi_0 = self.H[:, 0]  # First column of Fourier matrix = |+⟩
        
        # Apply Grover operator r times
        psi_final = psi_0
        for _ in range(r):
            psi_final = G @ psi_final
        
        if verbose:
            # Compute probabilities
            probs = [np.abs(psi_final[j])**2 for j in range(self.d)]
            print(f"  Final probabilities: {[f'{p:.3f}' for p in probs]}")
        
        # Measure in computational (Z) basis
        Z_projectors = [self.projector('Z', j) for j in range(self.d)]
        _, guess = self.measure(psi_final, Z_projectors, seed=seed)
        
        return guess


def test_qudit_backend():
    """Test the numerical qudit backend."""
    print("="*70)
    print("NUMERICAL QUDIT BACKEND TEST")
    print("="*70)
    print()
    
    # Test for various dimensions
    for d in [2, 3, 4, 8]:
        print(f"Testing d = {d}")
        print("-" * 70)
        
        backend = QuditBackend(d)
        
        # Verify Weyl relation: XZ = ωZX
        XZ = backend.X @ backend.Z
        ZX = backend.Z @ backend.X
        expected = backend.omega * ZX
        
        error = np.max(np.abs(XZ - expected))
        print(f"  Weyl relation XZ = ωZX: error = {error:.2e}")
        
        if error < 1e-10:
            print(f"  ✓ Correct!")
        else:
            print(f"  ✗ Failed!")
        
        # Test Grover's algorithm
        mark = 1 if d > 1 else 0
        successes = 0
        trials = 30
        
        for seed in range(trials):
            guess = backend.grover(mark, seed=seed, verbose=False)
            if guess == mark:
                successes += 1
        
        success_rate = successes / trials * 100
        print(f"  Grover search for |{mark}⟩:")
        print(f"    Success rate: {successes}/{trials} = {success_rate:.0f}%")
        
        # Show one example with probabilities
        print(f"    Example probabilities:")
        _ = backend.grover(mark, seed=42, verbose=True)
        
        print()
    
    print("="*70)
    print("COMPARISON: d=3 Grover")
    print("="*70)
    print()
    
    backend = QuditBackend(3)
    mark = 1
    
    print(f"Searching for marked state: |{mark}⟩")
    print(f"Expected: High probability on |{mark}⟩ after Grover")
    print()
    
    # Run detailed example
    Oracle = backend.reflect(backend.projector('Z', mark))
    Diffusion = -backend.reflect(backend.projector('X', 0))
    G = Diffusion @ Oracle
    
    psi_0 = backend.H[:, 0]
    psi_1 = G @ psi_0
    
    probs = [np.abs(psi_1[j])**2 for j in range(3)]
    print(f"Probabilities after 1 iteration:")
    for j in range(3):
        marker = " ← marked" if j == mark else ""
        print(f"  P({j}) = {probs[j]:.4f}{marker}")
    
    print()
    
    # Expected for d=3: should have ~95% on marked state
    if probs[mark] > 0.8:
        print("✓ Grover is amplifying correctly!")
        print(f"  {probs[mark]*100:.1f}% probability on marked state")
    else:
        print("⚠ Unexpected probabilities")


if __name__ == '__main__':
    test_qudit_backend()
