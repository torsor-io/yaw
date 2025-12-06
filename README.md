# yaw: Algebraic Quantum Programming

![Status](https://img.shields.io/badge/status-research%20prototype-yellow)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.7+-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

**Stop fiddling with qubits and program observables directly.**

`yaw` is a quantum programming language where programs are elements of operator algebras, not circuit
diagrams.
Using the power of math, `yaw` enables high-level, hardware-agnostic, fault tolerance-friendly code.
As described below, it can be used a standalone language (with both
functional and `Pythonic` elements) or imported directly as a package in `Python`.

> *Syntax : semantics = algebra : Hilbert space.*

In digital circuits, syntax (proof) is governed by the Boolean algebra and
semantics (meaning) by truth tables. Similarly, our core insight
is that for quantum circuits, the syntax should be governed by
C*-algebras and the semantics by Hilbert space.

`yaw` directly works with the algebra, but also allows users to
"compile" to the usual Hilbert space vectors and matrices using
something called the [*Gelfand-Naimark-Segal (GNS) construction*](https://en.wikipedia.org/wiki/Gelfand%E2%80%93Naimark%E2%80%93Segal_construction).
So we don't abandon Hilbert space, just give it some help!

> yaw algebraic way

In terms of [Euler angles](https://en.wikipedia.org/wiki/Euler_angles), `yaw` is (aspirationally!) the
direction we want to be heading. In case this seems presumptuous, it also stands for `yaw algebraic
way`, a [backronym](https://en.wikipedia.org/wiki/Backronym) which is not only [recursive](https://en.wikipedia.org/wiki/Recursive_acronym), but
*self-conjugating*, since `YAW⁻¹` = `WAY`. Please note the lateral symmetry.
(Someone had too much time on their hands.)

## Table of Contents
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
- [Codebase](#codebase)
- [Details](#details)
- [Contact](#contact)

## Quick Start

### Installation

```bash
git clone https://github.com/torsor-io/yaw.git
cd yaw
pip install -r requirements.txt  # Just sympy and numpy for now
```

### Try the example

```bash
python -m yaw.y2py examples/hello_world.yaw -o hello.py
python hello.py
```

### Three ways to use `yaw`

**1. Interactive REPL**

To dive in immediately, you can use the REPL:

```bash
python -m yaw.yaw_repl
```

```python
yaw> $alg = <X, Z | herm, unit, anti>
yaw> H = (X + Z) / sqrt(2)
yaw> H >> Z
X
```

**2. Compile to `Python`**

You can use the same syntax in `.yaw` files and compile to `Python`
using `y2py.py`:

```bash
python -m yaw.y2py protocol.yaw -o protocol.py
python protocol.py
```

**3. Import as library**
Finally, if you prefer, you can work with `yaw` as a package in
`Python`. The syntax is a little clunkier, but essentially the same:

```python
from yaw import *

pauli = Algebra(gens=['X', 'Z'], rels=['herm', 'unit', 'anti', 'pow(2)'])
H = (pauli.X + pauli.Z) / sqrt(2)

(H >> pauli.Z).normalize()        # X (symbolic)
gnsMat(char(pauli.Z, 0), H)       # Hadamard matrix (concrete)
# [[ 0.707  0.707]
#  [ 0.707 -0.707]]
```
## Core concepts

### Algebras define systems

We define algebras from "scratch" using generators and relations, e.g.

```python
# Qubit
$alg = <X, Z | herm, unit, anti>

# Qutrit
$alg = <X, Z | herm, unit, pow(3), braid(exp(2*pi*I/3))>

```

Here, `$alg` is a global algebra variable, hence the `$` (taking inspiration from `Unix`).

### States are functionals

If we view operators as random variables, states are *expectation functionals*. Eigenstates become expectations for which an operator has zero variance:

```python
psi0 = char(Z, 0)      # |0⟩ eigenstate

   Z | psi0            # ⟨Z⟩ = 1.0
   X | psi 0           # ⟨X⟩ = 0.0
Z**2 | psi0            # ⟨Z²⟩ = 1.0 - same as mean squared

psi0.expect(Z)         # Python syntax
```

The vertical line `|` should recall conditional probability, which assigns (probablistic) *context*. Here, the state is itself a probabilistic context in which to evaluate the operator. This is also like the pipe operator from `Unix` and acts the same way. Snap!

### Fundamental combinators

We can combine operators, states, and channels in various ways:

```python
A @ B           # Tensor product of operators and states: A ⊗ B
U >> A          # Operator conjugation: A ↦ U† A U (Heisenberg)
U << psi        # State conjugation: ψ = ⟨ψ|⋅|ψ⟩ ↦ ⟨ψ|U† ⋅ U|ψ⟩ (Schrödinger)
A | psi         # Evaluate A in state ψ: ψ(A)
A | C_1 | C_2   # Evaluate A in channel C_1, then C_2, etc: C_2[C_1[A]]
```

### Measurement: four interfaces

"Measurement" is an ambiguous term; it can mean the average outcome, sampling from a state-induced distribution, or a branching process where we keep track of all outcomes. We implement all of the above:

```python
# 1. Expectation values (most common)
Z | psi             

# 2. Single-shot state measurement (Schrödinger)
measure = stMeasure([proj(Z, 0), proj(Z, 1)], psi)
collapsed_state, probability = measure()

# 3. Single-shot operator measurement (Heisenberg)
measure = opMeasure([K0, K1], psi)
transformed_op, probability = measure(X)

# 4. All branches (shows full ensemble - no sampling)
branches = stBranches([proj(Z, 0), proj(Z, 1)], psi)
for state, prob in branches():
    print(f"Probability: {prob}")
```

### Special syntax

There are a few special, additional pieces of syntax:

```python
[[X, Z]]                                 # Commutator
{{X, Z}}                                 # Anticommutator
[P_{k} = proj(Z, k) for k in range(2)]   # Create P_0, P_1
expr ! pow(3), herm                      # Local context (temporary relations)
gnsMat(A), gnsVec(psi)                   # Concrete GNS Hilbert space matrices/vectors
```

We plan to add more (to facilitate abstract manipulation) in future.

## Codebase

### Language components

`yaw` consists of three integrated tools:

`yaw/yaw_prototype.py`: *Core library*
- Operator algebra with symbolic manipulation
- States as functionals (+ GNS construction)
- Quantum channels (Heisenberg and Schrödinger pictures)
- Measurement primitives with Born rule statistics
- Quantum error correction (encodings as functors)
- Context management for complex programs

`yaw/yaw_repl.py`: *Interactive development*
- Algebra definition: `$alg = <gens | rels>`
- Subscripted variables: `P_{k}`
- List comprehensions with assignment
- Commutator/anticommutator syntax
- Multi-line statements (`for`, `if`, `def`)

`yaw/y2py.py`: *Python compiler*
- Transforms `.yaw` files to Python
- Preserves algebraic semantics
- Generates readable, documented code
- Handles special syntax (commutators, subscripts, comprehensions)

### Status

**Current: v0.1.0 (Initial Public Release)**

`yaw` is a research prototype demonstrating what algebraic quantum programming can look like. This release focuses on single-qubit operations, with partial support for multi-qubit operations.

**What works:**
- ✅ Operator algebra with symbolic normalization
- ✅ State functionals (eigenstate characterization)
- ✅ Unitary conjugation (`<<`, `>>` for single operators)
- ✅ GNS construction (operators → Hilbert space matrices/vectors)
- ✅ Projective measurement with branching
- ✅ REPL with algebra definition syntax
- ✅ Compiler (`.yaw` → Python)
- ✅ Single-qubit gates (Hadamard, Pauli, rotations)
- ✅ Basic tensor products

**Known limitations:**
- Tensor product conjugation (e.g., `(H @ I) >> A`) not yet implemented
- Multi-qubit controlled operations need automated spectral decomposition
- Basic QEC support needs to be finalized

**Examples:**
- `hello_world.yaw` demonstrates core concepts
- More examples coming soon!

### Roadmap

**v0.2.0 (Q1 2026)**: *Multi-qubit robustness*
- Tensor product conjugation (`TensorProduct.__rshift__`)
- Automated spectral decomposition for controlled operations
- Additional examples: Bell states, Grover, HSP, basic QEC
- Comprehensive test suite

**v0.3.0 (2027)**: *Oscillators and advanced codes*
- Oscillator algebra (continuous variables)
- Surface codes and topological QEC
- Compiler optimization passes
- Stabilizer code automation

**v0.4.0 (2028)**: *λix integration*
- Hardware-agnostic backend (λix)
- Type system for quantum programs
- Module/import system
- Production-ready tooling

### Contributing

We welcome contributions! Areas where you can help:

- **Core library**: New operators, states, channels, QEC codes
- **REPL**: Better error messages, tab completion, history
- **Compiler**: Optimization passes, type checking, new syntax
- **Examples**: Quantum algorithms, protocols, tutorials
- **Documentation**: Guides, API docs, pedagogical content
- **Testing**: Unit tests, integration tests, benchmarks

**Quick start for contributors:**
1. Fork the repository
2. Make your changes with clear commit messages
3. Submit a pull request
4. We'll review and provide feedback

## Details

### References

- [The Structure and Interpretation of Quantum Programs I: Foundations](https://arxiv.org/abs/2509.04527) (2025), David Wakeham. A detailed introduction to the mathematical foundations of `yaw`.
-
  [A Short History of Rocks: or, How to Invent Quantum Computing](https://arxiv.org/abs/2503.00005) (2025), David Wakeham. An essay + alternate history motivating the need for an algebraic approach to quantum computing.

### Citation

If you use `yaw` in research, please cite:

```bibtex
@software{yaw2025,
  author = {Wakeham, David},
  title = {yaw: Algebraic Quantum Programming},
  year = {2025},
  publisher = {Torsor Labs},
  url = {https://github.com/torsor-io/yaw},
  version = {0.1.0}
}
```

### License

Apache 2.0 - see [LICENSE](LICENSE).

## Contact

**David Wakeham**  
Torsor Labs  
[Email](mailto:david@torsor.io) | [Website](https://torsor.io)

---

*Made with love, coffee, and Claude.*
