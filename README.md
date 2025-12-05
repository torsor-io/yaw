# yaw: Quantum Programming as Algebra

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

## Example: Quantum Teleportation

Here's quantum teleportation in `yaw`, demonstrating how we can build
everything we want from scratch:

```python
$alg = <X, Z | herm, unit, anti>       # Defines algebra of qubits

H = (X + Z) / sqrt(2)                  # Defines Hadamard gate
CNOT = ctrl(Z, [I, X])                 # Defines CNOT

psi00 = char(Z, 0) @ char(Z, 0)        # |00⟩

bell_pair = CNOT << (H @ I) << psi00   # Bell pair
to_send = # Alice's state here!
total_state = to_send @ bell_pair      # Total state

proj00 = proj(Z, 0) @ proj(Z, 0)       # Projector onto |00⟩

# Define Bell measurement projectors
def bell_proj(i, j):
	proj = (X**i @ X**j) >> proj00
	return CNOT >> (H @ I) >> proj
bell_projs = [[bell_proj(i, j) for i in [0,1]] 
               for j in [0,1]]

# To complete!
```

There are a few key differences from the normal circuit picture:

- **Functional**: We defined everything from scratch!
- **No circuit diagrams**: Operations are algebraic transformations.
- **States as functionals**: Not vectors, but functions on observables (GNS construction).
- **Branching measurement**: See all outcomes explicitly, not just one sample.

## Core Concepts

### Algebras Define Systems

```python
# Qubit (Pauli algebra)
$alg = <X, Z | herm, unit, anti>

# Qutrit (dimension 3)
$alg = <X, Z | herm, unit, pow(3), braid(exp(2*pi*I/3))>

# Relations encode physics: hermiticity, unitarity, dimension, braiding
```

### States are Functionals

```python
psi0 = char(Z, 0)      # |0⟩ eigenstate
psi0.expect(Z)         # ⟨Z⟩ = 1.0
psi0.expect(X)         # ⟨X⟩ = 0.0

# States eat operators, return expectation values
Z | psi0               # Alternate syntax: 1.0
```

### Three Fundamental Operations

```python
A @ B         # Tensor product: A ⊗ B
U >> A        # Conjugation: U† A U (transform operator)
U << psi      # State evolution: U|ψ⟩
A | psi       # Expectation: ⟨ψ|A|ψ⟩
```

### Measurement: Four Interfaces

```python
# 1. Expectation values (most common)
Z | psi

# 2. Single-shot state measurement (samples one outcome)
measure = stMeasure([proj(Z, 0), proj(Z, 1)], psi)
collapsed_state, probability = measure()

# 3. Single-shot operator measurement (Heisenberg picture)
measure = opMeasure([K0, K1], psi)
transformed_op, probability = measure(X)

# 4. All branches (shows full ensemble - no sampling)
branches = stBranches([proj(Z, 0), proj(Z, 1)], psi)
for state, prob in branches():
    print(f"Probability: {prob}")
```

### Special Syntax

```python
[[X, Z]]                               # Commutator
{{X, Z}}                               # Anticommutator
[P_{k} = proj(Z, k) for k in range(3)] # Create P_0, P_1, P_2
expr ! pow(3), herm                    # Local context (temporary relations)
```

## Language Components

yaw consists of three integrated tools:

**yaw/yaw_prototype.py** - Core library (~3700 lines)
- Operator algebra with symbolic manipulation
- States as functionals (GNS construction)
- Quantum channels (Heisenberg and Schrödinger pictures)
- Measurement primitives with Born rule statistics
- Quantum error correction (encodings as functors)
- Context management for complex programs

**yaw/yaw_repl.py** - Interactive development
- Algebra definition: `$alg = <gens | rels>`
- Subscripted variables: `P_{k}`
- List comprehensions with assignment
- Commutator/anticommutator syntax
- Multi-line statements (for, if, def)

**yaw/yawc.py** - Compiler
- Transforms `.yaw` files to Python
- Preserves algebraic semantics
- Generates readable, documented code
- Handles special syntax (commutators, subscripts, comprehensions)

## Contributing

We welcome contributions! Areas where you can help:

**Current: v0.1.0 (Initial Public Release)**

yaw is research software transitioning to production. What works:
- ✅ Core operator algebra with normalization
- ✅ State functionals with GNS foundations
- ✅ Full-featured REPL with special syntax
- ✅ Compiler (yaw → Python)
- ✅ Quantum channels and measurement
- ✅ Basic quantum error correction
- ✅ Multi-qudit tensor products
- ✅ Controlled operations and QFT

**Known limitations:**
- Oscillator (bosonic) support incomplete
- Limited compiler optimizations
- Some edge cases in state transformations
- Performance not yet optimized (correctness first)

**Roadmap:**

*v0.2.0 (Q1 2025)*
- Oscillator algebra and bosonic codes
- Advanced QEC (surface, color codes)
- Compiler optimization passes
- Comprehensive test suite

*v0.3.0 (Q2-Q3 2025)*
- λix backend (hardware-agnostic compilation)
- Stabilizer code automation
- Type system for quantum programs
- Module/import system

*v1.0.0 (2026)*
- Production-ready with λix integration
- Hardware backend connectors
- Formal verification tools
- Industry partnerships

## Status & Roadmap

**Current: v0.1.0 (Initial Public Release)**

yaw is research software transitioning to production. What works:
- ✅ Core operator algebra with normalization
- ✅ State functionals with GNS foundations
- ✅ Full-featured REPL with special syntax
- ✅ Compiler (yaw → Python)
- ✅ Quantum channels and measurement
- ✅ Basic quantum error correction
- ✅ Multi-qudit tensor products
- ✅ Controlled operations and QFT

**Known limitations:**
- Oscillator (bosonic) support incomplete
- Limited compiler optimizations
- Some edge cases in state transformations
- Performance not yet optimized (correctness first)

**Roadmap:**

*v0.2.0 (Q1 2025)*
- Oscillator algebra and bosonic codes
- Advanced QEC (surface, color codes)
- Compiler optimization passes
- Comprehensive test suite

*v0.3.0 (Q2-Q3 2025)*
- λix backend (hardware-agnostic compilation)
- Stabilizer code automation
- Type system for quantum programs
- Module/import system

*v1.0.0 (2026)*
- Production-ready with λix integration
- Hardware backend connectors
- Formal verification tools
- Industry partnerships

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

Questions or ideas? Open a [Discussion](https://github.com/torsorlabs/yaw/discussions) or [Issue](https://github.com/torsorlabs/yaw/issues).

## Philosophy

yaw embodies several principles:

- **Algebra first**: The mathematical structure *is* the program
- **Hardware agnostic**: Write for the abstraction, compile to any backend
- **Fault-tolerant native**: Designed for logical qubits, not NISQ workarounds
- **Composability**: Small pieces combine naturally
- **Joy-coded**: If it's not beautiful, it's not done

Traditional quantum computing: `Algorithm → Gates → Circuits → Hardware`

yaw: `Algorithm → Algebra → λix → Any Hardware`

The algebra layer is universal.

## Citation

If you use yaw in research, please cite:

```bibtex
@software{yaw2025,
  author = {Wakeham, David},
  title = {yaw: Quantum Programming as Algebra},
  year = {2025},
  publisher = {Torsor Labs},
  url = {https://github.com/torsorlabs/yaw},
  version = {0.1.0}
}
```

## Theoretical Foundation

yaw builds on rigorous mathematical physics:

**GNS Construction**: States are positive linear functionals ω: A → ℂ on operator algebras. The Gelfand-Naimark-Segal construction recovers Hilbert space representations, reversing the usual formulation—states are primary, Hilbert space emerges.

**C*-Algebras**: Operator algebras with addition, multiplication, adjoint, and norm. yaw operators satisfy algebraic relations (hermiticity, unitarity, braiding) without explicit matrix representations.

**Hardware Agnosticism**: By working at the algebraic level, yaw programs are independent of physical implementation—the same code compiles to superconducting qubits, trapped ions, photonics, or logical qubits.

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

yaw builds on foundations from C*-algebra theory (Gelfand, Naimark, Segal), categorical quantum mechanics (Abramsky, Coecke), algebraic QFT (Haag, Kastler), and modern quantum error correction (Gottesman, Kitaev, Preskill).

## Contact

**David Wakeham**  
Torsor Labs  
[Email](mailto:david@torsorlabs.com) | [Website](https://torsorlabs.com)

**Issues & Discussions**: [GitHub](https://github.com/torsorlabs/yaw)

---

*Made with love, coffee, and Claude.*

*"Boolean algebra for qubits."*
