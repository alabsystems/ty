<!-- dscan:allow(volatile_numbers) -->
# References

TY's design draws on decades of research in model checking, temporal logic,
theorem proving, SAT solving, and hardware verification. This document credits
the academic work and open-source projects that TY builds upon.

---

## TLA+ Language and Tools

### Foundational

- Leslie Lamport. *Specifying Systems: The TLA+ Language and Tools for Hardware
  and Software Engineers.* Addison-Wesley, 2002. The definitive reference for
  TLA+ syntax, semantics, and specification methodology.

- Leslie Lamport. "The Temporal Logic of Actions." *ACM TOPLAS*, 16(3):872-923,
  1994. The theoretical foundation: TLA+ is built on this temporal logic where
  every property is expressed as a formula of the form `Init /\ [][Next]_vars`.

- Leslie Lamport, Lawrence C. Paulson. "Should Your Specification Language Be
  Typed?" *ACM TOPLAS*, 21(3):502-526, 1999. The untyped design philosophy
  behind TLA+ that TY preserves.

- Leslie Lamport, Stephan Merz. "Auxiliary Variables in TLA+." 2017.
  arXiv:[1703.05121](https://arxiv.org/abs/1703.05121). Stuttering semantics
  and history/prophecy variables. TY's stuttering module follows this
  approach.

### TLC Model Checker

- Yuan Yu, Panagiotis Manolios, Leslie Lamport. "Model Checking TLA+
  Specifications." *CHARME*, 1999. The reference model checker. TY validates
  against TLC's behavior with targeted corpus checks for state counts, error
  detection, and fingerprint hashing (FP64 polynomial rolling hash). Full
  parity remains a release criterion, not a blanket claim.

### TLAPM Proof System

- Kaustuv Chaudhuri, Damien Doligez, Leslie Lamport, Stephan Merz. "Verifying
  Safety Properties with the TLA+ Proof System." *IJCAR*, 2010. The TLAPM
  architecture: hierarchical proof obligations dispatched to multiple backends
  (Zenon, Z3, LS4, Isabelle). TY currently carries `tla-zenon` and
  `tla-cert`; a TLAPM-compatible front door is not the active CLI surface.

### Apalache Symbolic Checker

- Igor Konnov, Jure Kukovec, Thanh-Hai Tran. "TLA+ Model Checking Made
  Symbolic." *OOPSLA*, 2019. Symbolic model checking for TLA+ via SMT encoding.
  TY implements Apalache's operator extensions (`:=` assignment, `:>` function
  constructor, `@@` function merge, `<:` type annotation) and maintains a
  compatibility test suite.

---

## Model Checking

### Explicit-State Model Checking

- Edmund M. Clarke, E. Allen Emerson, A. Prasad Sistla. "Automatic Verification
  of Finite-State Concurrent Systems Using Temporal Logic Specifications."
  *ACM TOPLAS*, 8(2):244-263, 1986. The foundational paper on temporal logic
  model checking.

- Gerard J. Holzmann. *The SPIN Model Checker: Primer and Reference Manual.*
  Addison-Wesley, 2003. Explicit-state model checking with partial order
  reduction. TY's POR implementation references SPIN's ample set construction.

### Bounded Model Checking

- Armin Biere, Alessandro Cimatti, Edmund M. Clarke, Yunshan Zhu. "Symbolic
  Model Checking without BDDs." *TACAS*, 1999. The original BMC paper:
  unrolling the transition relation K times and encoding as a SAT problem.
  TY's `ty check --bmc` and `tla-aiger` BMC engine implement this
  approach.

### IC3 / Property-Directed Reachability

- Aaron R. Bradley. "SAT-Based Model Checking without Unrolling." *VMCAI*,
  2011. doi:[10.1007/978-3-642-18275-4_7](https://doi.org/10.1007/978-3-642-18275-4_7).
  The IC3 algorithm: maintain a sequence of frames (overapproximations of
  reachable states), block counterexample cubes backward, propagate lemmas
  forward. TY's IC3 engines (`tla-aiger`, `tla-ay` PDR) implement this
  core algorithm.

- Aaron R. Bradley. "Understanding IC3." *SAT*, 2012.
  doi:[10.1007/978-3-642-31612-8_1](https://doi.org/10.1007/978-3-642-31612-8_1).
  Tutorial exposition of IC3 with correctness arguments.

- Niklas Een, Alan Mishchenko, Robert Brayton. "Efficient Implementation of
  Property Directed Reachability." *FMCAD*, 2011.
  doi:[10.1109/FMCAD.2011.6148886](https://doi.org/10.1109/FMCAD.2011.6148886).
  Practical IC3/PDR implementation: counterexample-to-generalization (CTG),
  priority queues for proof obligations, delta encoding of frame solvers.
  TY's IC3 engines implement CTG and the priority-based obligation queue.

- Zyad Hassan, Aaron R. Bradley, Fabio Somenzi. "Better Generalization in
  IC3." *FMCAD*, 2013. Improved MIC (minimum inductive clause) algorithms.
  TY implements core-based initial reduction before the literal-drop loop.

### k-Induction

- Mary Sheeran, Satnam Singh, Gunnar Stalmarck. "Checking Safety Properties
  Using Induction and a SAT-Solver." *FMCAD*, 2000. k-induction: prove that
  if all k-step paths satisfy the property, no (k+1)-step path can violate it.
  TY's `tla-aiger` k-induction engine implements this with simple-path
  constraints.

---

## Hardware Model Checking

### AIGER Format

- Armin Biere. "The AIGER And-Inverter Graph (AIG) Format Version 20071012."
  Johannes Kepler University, 2006-2007. The standard format for hardware
  model checking benchmarks. TY's AIGER parser handles both ASCII (.aag) and
  binary (.aig) formats with the extended HWMCC header (bad state properties,
  fairness constraints, justice properties).

### BTOR2 Format

- Aina Niemetz, Mathias Preiner, Clifford Wolf, Armin Biere. "BTOR2, BtorMC
  and Boolector 3.0." *CAV*, 2018. doi:[10.1007/978-3-319-96145-3_32](https://doi.org/10.1007/978-3-319-96145-3_32).
  Word-level hardware verification format. TY's BTOR2 parser and CHC
  translator implement the full BTOR2 instruction set.

### rIC3

- Yuheng Su. "rIC3." [github.com/gipsyh/rIC3](https://github.com/gipsyh/rIC3),
  2023-2025. A Rust IC3/PDR implementation (~28K LOC) that placed #2 at
  HWMCC'25 safety (274/330 benchmarks solved). TY's `tla-aiger` crate draws
  extensively from rIC3's architecture:
  - **Domain-restricted SAT** (`src/gipsat/domain.rs`): each IC3 query
    restricts the solver to cone-of-influence variables. TY implements this
    at the application level (`ic3/domain.rs`) by building mini-solvers with
    only domain-relevant clauses.
  - **Custom SAT integration** (`src/gipsat/`): bucket-queue VSIDS, solver
    cloning for frame reuse, `flip_to_none` for state lifting without extra
    SAT calls. TY implements flip-to-none in `ic3/lift.rs`.
  - **17-worker portfolio** (`src/portfolio/`): 11 IC3 variants + 4 BMC + 1
    k-induction + 1 multi-property. TY implements a 16-configuration thread-based portfolio.
  - **Preprocessing**: ternary simulation + functional reduction,
    SCORR-style equivalence, BVE. TY implements ternary simulation and COI
    reduction.
  - **CTG generalization** (`src/ic3/mic.rs`): recursive
    counterexample-to-generalization with activity-guided literal ordering.
  - **Parent lemma heuristic** (CAV'23): bias generalization toward
    previously discovered lemma shapes.

### IC3ref

- Aaron R. Bradley. "IC3ref." [github.com/arbrad/IC3ref](https://github.com/arbrad/IC3ref).
  The original IC3 reference implementation in C++ (~2K LOC). Clean,
  minimal, faithful to the VMCAI'11 paper.

### ABC

- Robert Brayton, Alan Mishchenko. "ABC: An Academic Industrial-Strength
  Verification Tool." *CAV*, 2010.
  doi:[10.1007/978-3-642-14295-6_5](https://doi.org/10.1007/978-3-642-14295-6_5).
  AIG synthesis and preprocessing: rewriting (`rw`), refactoring (`rf`),
  balancing (`b`). super-prove's success at HWMCC (#4 safety, 255/330) is
  largely attributable to ABC's preprocessing achieving 30-60% circuit
  shrinkage before verification.

### Other HWMCC Systems

- Aman Goel, Karem Sakallah. "AVR: Abstractly Verifying Reachability."
  *TACAS*, 2020. Abstraction-refinement based hardware verification.

- Makai Mann et al. "Pono: A Flexible and Extensible SMT-Based Model Checker."
  *CAV*, 2021. SMT-based hardware model checking from Stanford CENTAUR.

---

## Temporal Logic and Liveness

### Tableau Construction

- Zohar Manna, Amir Pnueli. *Temporal Verification of Reactive Systems:
  Safety.* Springer, 1995. Chapter 5 (pages 405-452): the tableau construction
  for liveness checking that TY's `liveness/tableau.rs` implements.

### Automata-Theoretic Model Checking

- Moshe Y. Vardi, Pierre Wolper. "An Automata-Theoretic Approach to Automatic
  Program Verification." *LICS*, 1986. The foundational approach: verify
  liveness by checking emptiness of the product of the system and the negation
  automaton.

- Moshe Y. Vardi. "Automata-Theoretic Model Checking Revisited." *VMCAI*,
  2007. doi:[10.1007/978-3-540-69738-1_10](https://doi.org/10.1007/978-3-540-69738-1_10).
  Survey of automata-based model checking techniques.

### LTL to Buchi Automata

- Paul Gastin, Denis Oddoux. "Fast LTL to Buchi Automata Translation."
  *CAV*, 2001. doi:[10.1007/3-540-44585-4_6](https://doi.org/10.1007/3-540-44585-4_6).
  Efficient on-the-fly LTL to NBA translation. TY's `liveness/on_the_fly/nba.rs`
  implements this approach.

### SCC Detection

- Robert Tarjan. "Depth-First Search and Linear Graph Algorithms." *SIAM
  Journal on Computing*, 1(2):146-160, 1972. The Tarjan SCC algorithm used
  throughout TY's liveness checking for detecting accepting cycles.

### Nested Depth-First Search

- Costas Courcoubetis, Moshe Y. Vardi, Pierre Wolper, Mihalis Yannakakis.
  "Memory-Efficient Algorithms for the Verification of Temporal Properties."
  *Formal Methods in System Design*, 1(2-3):275-288, 1992. The nested DFS
  algorithm for Buchi automata emptiness. TY's `liveness/on_the_fly/nested_dfs.rs`
  implements this.

- Gerard J. Holzmann, Doron Peled, Mihalis Yannakakis. "On Nested Depth First
  Search." *The Spin Verification System*, pages 23-32, 1996. Improved nested
  DFS formulation.

- Alfons Laarman, Jaco van de Pol, Michael Weber. "Multi-Core Nested Depth-First
  Search." *ATVA*, 2011. CNDFS parallel variant referenced in TY's liveness
  documentation.

---

## Partial Order Reduction

- Doron Peled. "All from One, One for All: On Model Checking Using
  Representatives." *CAV*, 1993. doi:[10.1007/3-540-56922-7_34](https://doi.org/10.1007/3-540-56922-7_34).
  Ample set partial order reduction. TY's `por/` module implements this.

- Patrice Godefroid. *Partial-Order Methods for the Verification of Concurrent
  Systems: An Approach to the State-Explosion Problem.* Springer LNCS 1032,
  1996. Sleep set and persistent set methods.

---

## Symmetry Reduction

- C. Norris Ip, David L. Dill. "Better Verification Through Symmetry."
  *Formal Methods in System Design*, 9(1-2):41-75, 1996. Symmetry reduction
  via permutation groups. TY supports the `SYMMETRY` keyword with automatic
  permutation group detection.

- E. Allen Emerson, A. Prasad Sistla. "Symmetry and Model Checking."
  *Formal Methods in System Design*, 9(1-2):105-131, 1996.

---

## Parallel Model Checking

- Robert D. Blumofe, Charles E. Leiserson. "Scheduling Multithreaded
  Computations by Work Stealing." *JACM*, 46(5):720-748, 1999. The
  work-stealing scheduling algorithm. TY's parallel BFS uses a
  work-stealing deque (via Rayon) for load balancing across worker threads.

- Gerard J. Holzmann, Dragan Bosnacki. "The Design of a Multicore Extension
  of the SPIN Model Checker." *IEEE TSE*, 33(10):659-674, 2007. Multicore
  explicit-state model checking with shared hash tables.

---

## SAT Solving

TY's hardware verification engines use ay-sat (a pure Rust CDCL solver) as
their SAT backend. The following references describe the core CDCL techniques
that ay-sat implements.

- Joao Marques-Silva, Karem Sakallah. "GRASP: A Search Algorithm for
  Propositional Satisfiability." *IEEE Transactions on Computers*, 1999.
  Conflict-driven clause learning (CDCL).

- Niklas Een, Niklas Sorensson. "An Extensible SAT-solver." *SAT*, 2003.
  MiniSat: two-watched literals and conflict clause minimization.

- Armin Biere et al. "CaDiCaL 2.0." *SAT*, 2024. Modern CDCL with
  inprocessing. CaDiCaL and Kissat are the state-of-the-art SAT competition
  solvers; ay-sat ports their core techniques.

- G. S. Tseitin. "On the Complexity of Derivation in Propositional Calculus."
  *Studies in Constructive Mathematics and Mathematical Logic*, 1968. The
  Tseitin transformation used in TY's AIG-to-CNF conversion.

---

## Theorem Proving

### Zenon

- Richard Bonichon, David Delahaye, Damien Doligez. "Zenon: An Extensible
  Automated Theorem Prover Producing Checkable Proofs." *LPAR*, 2007.
  doi:[10.1007/978-3-540-75560-9_13](https://doi.org/10.1007/978-3-540-75560-9_13).
  First-order analytic tableau prover with alpha/beta/gamma/delta rules.
  TY's `tla-zenon` is a Rust port.

### SMT-Based Verification

- Leonardo de Moura, Nikolaj Bjorner. "Z3: An Efficient SMT Solver." *TACAS*,
  2008. doi:[10.1007/978-3-540-78800-3_24](https://doi.org/10.1007/978-3-540-78800-3_24).
  Z3 remains a reference point for SMT-backed proof obligations. TY's active
  in-tree symbolic checking path is ay.

---

## Constrained Horn Clauses

TY uses ay's CHC engine for symbolic verification of TLA+ specifications
and BTOR2 hardware designs. The CHC engine implements multiple solving
strategies (PDR/IC3, CEGAR, TPA, TRL, PDKind, LAWI, IMC).

---

## Tools and Systems

| System | Authors | License | Role in TY |
|--------|---------|---------|-------------|
| [TLC](https://github.com/tlaplus/tlaplus) | Yu, Manolios, Lamport | MIT | Reference model checker; behavioral baseline |
| [TLAPM](https://github.com/tlaplus/tlapm) | Chaudhuri, Doligez, Lamport, Merz | BSD | Proof system architecture reference |
| [Zenon](https://github.com/zenon-prover/zenon) | Bonichon, Delahaye, Doligez | BSD | First-order prover; `tla-zenon` is a Rust port |
| [Apalache](https://github.com/informalsystems/apalache) | Konnov, Kukovec, Tran | Apache-2.0 | Symbolic TLA+ checking; operator extensions |
| [rIC3](https://github.com/gipsyh/rIC3) | Yuheng Su | GPL-3.0 | Primary AIGER engine reference |
| [IC3ref](https://github.com/arbrad/IC3ref) | Aaron Bradley | MIT | Original IC3 reference implementation |
| [ABC](https://github.com/berkeley-abc/abc) | Brayton, Mishchenko | custom | AIG synthesis techniques reference |
| [ay](https://github.com/alabsystems/ay) | Yates | Apache-2.0 | SAT/SMT/CHC backend |
| [Z3](https://github.com/Z3Prover/z3) | de Moura, Bjorner et al. | MIT | SMT architecture reference |
| [CaDiCaL](https://github.com/arminbiere/cadical) | Armin Biere | MIT | SAT techniques reference (via ay-sat) |
| [Kissat](https://github.com/arminbiere/kissat) | Armin Biere | MIT | SAT techniques reference (via ay-sat) |
| [Kani](https://github.com/model-checking/kani) | Amazon | Apache-2.0 | Rust verification; codegen harness target |
| [SPIN](https://spinroot.com/) | Holzmann | custom | Explicit-state model checking reference |
| [Rayon](https://github.com/rayon-rs/rayon) | Niko Matsakis et al. | Apache-2.0 | Work-stealing parallel BFS infrastructure |
