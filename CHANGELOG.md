# Changelog

## [0.1.1] - 2025-12-13

### Added
- Complete tensor product algebra support
  - `TensorProduct.adjoint()`: Hermitian conjugate distributes over factors
  - `TensorProduct.__rshift__()`: Operator conjugation distributes over factors
  - Property shortcuts for adjoint: `.H`, `.dag`, `.d`, `.T`
  - `TensorProduct.conj_op()`: Explicit conjugation method
- Command history and line editing

### Fixed
- Tensor products now fully integrate with existing operator algebra
- Normalization works correctly for tensor products: `(X@X)*(X@X) = I@I`
- Conjugation preserves tensor structure: `(H@H) >> (X@X) = Z@Z`

## [0.1.0] - 2025-12-06

### Initial Release
- Basic operator algebra with symbolic normalization
- Qubit and qudit support
- State conjugation and expectation values
- Quantum error correction (stabilizer codes)
- GNS construction for spectral analysis
