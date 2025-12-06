# yaw: Algebraic Quantum Programming

`yaw` is a quantum programming language where programs are elements of operator algebras, not circuit
diagrams.
Using the power of math, `yaw` enables high-level, hardware
agnostic, and fault tolerance-friendly code.
Please note, it is currently a research prototype rather than a
production language.
As described below, it can be used a standalone language (with both
functional and `Pythonic` elements) or imported directly as a package in `Python`.

> *Syntax : semantics = algebra : Hilbert space.*

In digital circuits, the syntax (proofs) is governed by the Boolean algebra and
the semantics (meaning) by truth tables. Similarly, our core insight
is that for quantum circuits, the syntax should be governed by
C*-algebra and the semantics by Hilbert space.

`yaw` directly works with the algebra, but also allows users to
"compile" to the usual Hilbert space vectors and matrices using
something called the *Gelfand-Naimark-Segal* representation.
So we don't abandon Hilbert space, just give it some help!

> yaw algebraic way

In terms of Euler angles, `yaw` (aspirationally) is the
direction we're headed. However, it also stands for `yaw algebraic
way`, a backronym which is not merely recursive, but
*self-conjugating*, since `yaw⁻¹` = `way`.
Someone had too much time on their hands.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Language Components](#language-components)
- [Status and Roadmap](#status-and-roadmap)
- [Contributing](#contributing)
- [Philosophy](#philosophy)
- [References](#references)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Quick Start

### Installation

```bash
git clone https://github.com/torsor-io/yaw.git
cd yaw
pip install -r requirements.txt  # Just sympy for now
```

### Three Ways to Use yaw

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

**2. Compile to Python**

You can use the same syntax in `.yaw` files and compile to `Python`
using `yawc.py`:

```bash
python -m yaw.yawc protocol.yaw -o protocol.py
python protocol.py
```

**3. Import as Library**
Finally, if you prefer, you can work with `yaw` as a package in
`Python`. The syntax is a little clunkier, but essentially the same:

```python
from yaw import *

pauli = Algebra(gens=['X', 'Z'], rels=['herm', 'unit', 'anti'])
X, Z = pauli.X, pauli.Z
print((X * Z * X).normalize())  # -Z
```

## Core Concepts

### Algebras Define Systems

We define algebras from "scratch" using generators and relations, e.g.

```python
# Qubit (Pauli algebra)
$alg = <X, Z | herm, unit, anti>

# Qutrit
$alg = <X, Z | herm, unit, pow(3), braid(exp(2*pi*I/3))>

```

Here, `$alg` is a global algebra variable, hence the `$`.

### States are Functionals

If we view operators as random variables, states are *expectation functionals*. Eigenstates become expectations for which an operator has zero variance.

```python
psi0 = char(Z, 0)      # |0⟩ eigenstate

   Z | psi0               # ⟨Z⟩ = 1.0
   X | psi 0              # ⟨X⟩ = 0.0
Z**2 | psi0            # ⟨Z²⟩ = 1.0 - same as mean squared

psi0.expect(Z)         # Python syntax
```

The vertical line `|` should recall conditional probability, which put differently, assigned *context*. Here, the state is itself a context in which to evaluate the operator. This is also like the pipe operator from Unix. Snap!

### Fundamental Combinators

We can combine operators, states, and channels in various ways:

```python
A @ B           # Tensor product of operators and states: A ⊗ B
U >> A          # Operator conjugation: A ↦ U† A U (Heisenberg)
U << psi        # State conjugation: ψ(⋅) ↦ ψ(U† ⋅ U) (Schrödinger)
A | psi         # Evaluate A in state ψ: ψ(A)
A | C_1 | C_2   # Evaluate A in channel C_1, then C_2, etc: C_2[C_1[A]]
```

### Measurement: Four Interfaces

Measurement is a tricky notion, since it can mean the average outcome, a random choice over a state-induced distribution, or a branching process where we keep track of all outcomes.

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

### Special Syntax

There are a few special, additional pieces of syntax:

```python
[[X, Z]]                                 # Commutator
{{X, Z}}                                 # Anticommutator
[P_{k} = proj(Z, k) for k in range(2)]   # Create P_0, P_1
expr ! pow(3), herm                      # Local context (temporary relations)
```

## Language Components

`yaw` consists of three integrated tools:

`yaw/yaw_prototype.py` - Core library
- Operator algebra with symbolic manipulation
- States as functionals (+ GNS construction)
- Quantum channels (Heisenberg and Schrödinger pictures)
- Measurement primitives with Born rule statistics
- Quantum error correction (encodings as functors)
- Context management for complex programs

`yaw/yaw_repl.py` - Interactive development
- Algebra definition: `$alg = <gens | rels>`
- Subscripted variables: `P_{k}`
- List comprehensions with assignment
- Commutator/anticommutator syntax
- Multi-line statements (for, if, def)

`yaw/yawc.py` - Compiler
- Transforms `.yaw` files to Python
- Preserves algebraic semantics
- Generates readable, documented code
- Handles special syntax (commutators, subscripts, comprehensions)

## Status and roadmap

**Current: v0.1.0 (Initial Public Release)**

`yaw` is a research prototype transitioning to production. What works (modulo Heisenbugs):
- ✅ Core operator algebra with normalization
- ✅ State functionals with GNS foundations
- ✅ Full-featured REPL with special syntax
- ✅ Compiler (yaw → Python)
- ✅ Quantum channels and measurement
- ✅ Basic quantum error correction
- ✅ Multi-qudit tensor products
- ✅ Controlled operations and QFT

**Roadmap:**

*v0.2.0 (Q2-3 2026)*
- Oscillator algebra and bosonic codes
- Advanced QEC (surface, color codes)
- Compiler optimization passes
- Comprehensive test suite

*v0.3.0 (2027)*
- λix backend (hardware-agnostic compilation)
- Stabilizer code automation
- Type system for quantum programs
- Module/import system

## Contributing

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

## Philosophy

`yaw` embodies several principles:

- **Algebra first**: The mathematical structure *is* the program
- **Hardware agnostic**: Write for the abstraction, compile to any backend
- **Fault-tolerant native**: Designed for error-correction integration, not NISQ
- **Composability**: Small pieces combine naturally
- **Joy-coded**: If it's not beautiful, it's not done

## References

- [The Structure and Interpretation of Quantum Programs I: Foundations](https://arxiv.org/abs/2509.04527) (2025), David Wakeham. A detailed introduction to the mathematical foundations of `yaw`.
- [A Short History of Rocks: or, How to Invent Quantum Computing](https://arxiv.org/abs/2503.00005) (2025), David Wakeham. An essay + alternate history motivating the need for a new approach to quantum computing.

## Citation

If you use yaw in research, please cite:

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

## License

Apache 2.0 - see [LICENSE](LICENSE).

## Contact

**David Wakeham**  
Torsor Labs  
[Email](mailto:david@torsor.io) | [Website](https://torsor.io)

---

*Made with love, coffee, and Claude.*
