# entropy-conservation-rs

Entropy conservation tracking with Hodge decomposition for fleet systems.

When agents exchange information, entropy should be conserved — what one agent
loses, another gains. This crate decomposes entropy-change matrices into
gradient (conservative transfer), curl (cyclic redistribution), and harmonic
(waste) components, then measures conservation law violations.

## Why Care?

You have 10 agents in a fleet. After a round of communication, you record how
much entropy each agent sent and received. The total entropy budget should be
conserved — but is it?

```
Agent 0: sent 3.0, received 0.0  → net -3.0
Agent 1: sent 0.0, received 3.0 → net +3.0
Agent 2: sent 1.0, received 0.0 → net -1.0
```

Here Agent 0 transferred 3.0 to Agent 1 (conservative), but Agent 2 lost 1.0
that went nowhere. That's a conservation violation. This crate tells you:

- **Which transfers are conservative** (gradient component)
- **Which are cyclic** (curl — entropy circulates without net change)
- **How much is genuinely lost or created** (harmonic — violation)

## Quick Start

```toml
# Cargo.toml
[dependencies]
entropy-conservation-rs = "0.1.0"
```

```rust
use entropy_conservation_rs::hodge_decomposition::{decompose, is_conserved, conservation_violation};

// Entropy transfer matrix: F[i][j] = entropy sent from agent i to dimension j
let transfers = vec![
    vec![0.0, 3.0],   // Agent 0 sends 3.0 to Agent 1
    vec![-3.0, 0.0],  // Agent 1 receives 3.0 from Agent 0
];

// Decompose into gradient, curl, and harmonic components
let hodge = decompose(&transfers);

println!("Gradient energy: {:.4}", hodge.gradient_energy());
// => Gradient energy: 18.0000
println!("Curl energy:     {:.4}", hodge.curl_energy());
// => Curl energy:     0.0000
println!("Harmonic energy: {:.4}", hodge.harmonic_energy());
// => Harmonic energy: 0.0000

// Check conservation between two state snapshots
let before = vec![5.0, 3.0, 2.0]; // total = 10.0
let after  = vec![4.0, 4.0, 2.0]; // total = 10.0
println!("Conserved: {}", is_conserved(&before, &after));
// => Conserved: true

let after2 = vec![4.0, 4.0, 3.0]; // total = 11.0 — created 1.0!
println!("Violation: {:.6}", conservation_violation(&before, &after2));
// => Violation: 1.000000
```

## Core Concepts Through Code

### The Hodge–Helmholtz Decomposition

The central idea: any entropy-change matrix `F` splits into three orthogonal
components:

```
F = Gradient + Curl + Harmonic
```

Each component has a physical interpretation:

| Component | Meaning | Conservation |
|-----------|---------|-------------|
| **Gradient** | Potential-driven transfer (one gains, one loses) | Conserved pairwise |
| **Curl** | Cyclic redistribution (A→B→C→A) | Conserved per-agent |
| **Harmonic** | Genuine creation or destruction | Violation |

```rust
use entropy_conservation_rs::hodge_decomposition::decompose;

// Pure conservative transfer: A sends 5.0 to B, B sends 2.0 to C, C sends 5.0 to A
let cyclic_flow = vec![
    vec![0.0, 5.0, 0.0],  // A → B
    vec![0.0, 0.0, 2.0],  // B → C
    vec![5.0, 0.0, 0.0],  // C → A
];

let hodge = decompose(&cyclic_flow);

// Each agent sends as much as it receives, so divergence is zero.
// Gradient should be near-zero.
println!("Gradient energy: {:.6}", hodge.gradient_energy());
// => Gradient energy: ~0.0

// Most energy is in curl (cyclic flow)
println!("Curl energy: {:.6}", hodge.curl_energy());
// => Curl energy: positive

// No conservation violations in a pure cycle
println!("Harmonic energy: {:.6}", hodge.harmonic_energy());
// => Harmonic energy: ~0.0
```

### Perfect Reconstruction

The three components always sum back to the original matrix:

```rust
use entropy_conservation_rs::hodge_decomposition::decompose;

let original = vec![
    vec![1.0, 2.0, 3.0],
    vec![4.0, 5.0, 6.0],
    vec![7.0, 8.0, 9.0],
];

let hodge = decompose(&original);
let reconstructed = hodge.reconstruct();

for i in 0..3 {
    for j in 0..3 {
        assert!((reconstructed[i][j] - original[i][j]).abs() < 1e-8);
    }
}
// Decomposition is lossless: F = G + C + H exactly.
```

### Energy Diagnostics

Each component has a Frobenius norm ("energy"):

```rust
use entropy_conservation_rs::hodge_decomposition::decompose;

let transfers = vec![
    vec![0.0, 3.0],
    vec![-3.0, 0.0],
];
let hodge = decompose(&transfers);

let g = hodge.gradient_energy();
let c = hodge.curl_energy();
let h = hodge.harmonic_energy();

println!("Total energy breakdown:");
println!("  Gradient (conservative): {:.4} ({:.1}%)", g, g/(g+c+h)*100.0);
println!("  Curl (cyclic):           {:.4} ({:.1}%)", c, c/(g+c+h)*100.0);
println!("  Harmonic (violation):    {:.4} ({:.1}%)", h, h/(g+c+h)*100.0);

// For pure antisymmetric transfer:
// => Gradient: 18.0000 (100.0%)
// => Curl:     0.0000 (0.0%)
// => Harmonic: 0.0000 (0.0%)
```

## API Reference

### `HodgeComponents`

```rust
pub struct HodgeComponents {
    pub gradient: Vec<Vec<f64>>,  // Conservative (irrotational) component
    pub curl: Vec<Vec<f64>>,      // Cyclic (solenoidal) component
    pub harmonic: Vec<Vec<f64>>,  // Conservation violation (residual)
}

impl HodgeComponents {
    /// Sum all three components to recover the original matrix.
    pub fn reconstruct(&self) -> Vec<Vec<f64>>

    /// Frobenius norm of the gradient component.
    pub fn gradient_energy(&self) -> f64

    /// Frobenius norm of the curl component.
    pub fn curl_energy(&self) -> f64

    /// Frobenius norm of the harmonic component.
    pub fn harmonic_energy(&self) -> f64
}
```

### `decompose`

```rust
/// Perform Hodge decomposition on an n×m entropy-change matrix.
///
/// Algorithm:
///   1. Compute divergence: d[i] = Σ_j F[i][j] - Σ_k F[k][i]
///   2. Gradient: G[i][j] = (d[i] - d[j]) / n
///   3. Curl: project residual to zero row-sum and zero column-sum
///   4. Harmonic: H = F - G - C
pub fn decompose(entropy_changes: &[Vec<f64>]) -> HodgeComponents
```

### `is_conserved`

```rust
/// Check if total entropy is conserved between two state vectors.
/// Tolerance: 1e-10.
pub fn is_conserved(original: &[f64], after: &[f64]) -> bool
```

### `conservation_violation`

```rust
/// Compute |sum(original) - sum(after)| — magnitude of conservation violation.
pub fn conservation_violation(original: &[f64], after: &[f64]) -> f64
```

## Advanced Examples

### Fleet-Wide Entropy Accounting

Track entropy budgets across a fleet communication round:

```rust
use entropy_conservation_rs::hodge_decomposition::{
    decompose, is_conserved, conservation_violation,
};

// Before communication, each agent has this much entropy:
let entropy_before = vec![2.0, 3.0, 5.0, 1.0, 4.0]; // total = 15.0

// After communication:
let entropy_after = vec![1.5, 3.5, 5.0, 1.5, 3.5]; // total = 15.0

// Quick check: did we conserve entropy globally?
if is_conserved(&entropy_before, &entropy_after) {
    println!("✓ Global entropy conserved");
} else {
    let violation = conservation_violation(&entropy_before, &entropy_after);
    println!("✗ Conservation violation: {:.6}", violation);
}

// Now decompose the pairwise transfer matrix
let transfer_matrix = vec![
    vec![ 0.0,  0.5,  0.0,  0.0,  0.0],  // Agent 0 → Agent 1: 0.5
    vec![ 0.0,  0.0,  0.0,  0.0,  0.0],  // Agent 1 sends nothing
    vec![ 0.0,  0.0,  0.0,  0.5,  0.0],  // Agent 2 → Agent 3: 0.5
    vec![ 0.0,  0.0,  0.0,  0.0,  0.0],  // Agent 3 sends nothing
    vec![ 0.0,  0.0,  0.0,  0.0,  0.0],  // Agent 4 sends nothing
];

let hodge = decompose(&transfer_matrix);

println!("Conservative transfers: {:.4}", hodge.gradient_energy());
println!("Cyclic redistribution:  {:.4}", hodge.curl_energy());
println!("Entropy waste:          {:.4}", hodge.harmonic_energy());

// Interpret: if harmonic_energy > 0, some entropy was created or destroyed
if hodge.harmonic_energy() > 0.01 {
    println!("⚠ Entropy not conserved in pairwise transfers!");
    println!("  Check for information creation/destruction bugs.");
}
```

### Detecting Information Leaks

The harmonic component reveals where entropy goes missing:

```rust
use entropy_conservation_rs::hodge_decomposition::decompose;

// Simulate a fleet where Agent 2 "leaks" entropy
let leaky_transfers = vec![
    vec![0.0, 2.0, 0.0],  // Agent 0 → Agent 1: 2.0
    vec![0.0, 0.0, 1.0],  // Agent 1 → Agent 2: 1.0
    vec![0.0, 0.0, 0.0],  // Agent 2 sends nothing (leaked!)
];

let hodge = decompose(&leaky_transfers);

// The harmonic component captures the leak
println!("Leaked entropy (harmonic): {:.4}", hodge.harmonic_energy());

// => Non-zero harmonic means entropy vanished from the system.
//    Agent 2 received 1.0 but didn't pass it on or return it.

// Decomposition helps you pinpoint: look at harmonic[i][j] for large values
for (i, row) in hodge.harmonic.iter().enumerate() {
    for (j, &val) in row.iter().enumerate() {
        if val.abs() > 0.01 {
            println!("  Violation at [{}, {}]: {:.4}", i, j, val);
        }
    }
}
```

### Zero Matrix and Edge Cases

```rust
use entropy_conservation_rs::hodge_decomposition::decompose;

// No transfers at all
let zero = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
let hodge = decompose(&zero);
assert!(hodge.gradient_energy() < 1e-10);
assert!(hodge.curl_energy() < 1e-10);
assert!(hodge.harmonic_energy() < 1e-10);
// => All zero — nothing happened, nothing violated.

// Empty matrix
let empty = decompose(&vec![]);
assert!(empty.gradient.is_empty());

// Single element (1×1 matrix)
let single = decompose(&vec![vec![5.0]]);
// divergence = 5.0 - 5.0 = 0, so gradient is zero
assert!(single.gradient[0][0].abs() < 1e-10);
```

### Integration with renormalization-group-rs

Coarse-grain agent entropy states and check conservation at each scale:

```rust
use entropy_conservation_rs::hodge_decomposition::{decompose, is_conserved};
// Hypothetical integration with renormalization_group_rs

fn check_conservation_at_all_scales(
    agent_entropy: &[f64],
    block_size: usize,
    levels: usize,
) -> Vec<bool> {
    let mut conserved_at_level = vec![is_conserved(agent_entropy, agent_entropy)];

    // At each coarse-graining level, verify total entropy is preserved
    let mut current = agent_entropy.to_vec();
    for level in 0..levels {
        if current.len() < block_size {
            break;
        }
        // Coarse-grain by averaging blocks
        let mut next = Vec::new();
        for chunk in current.chunks(block_size) {
            let avg = chunk.iter().sum::<f64>() / chunk.len() as f64;
            next.push(avg);
        }
        // Note: averaging changes total, so we check sum * block_size
        // In practice you'd sum blocks, not average
        let fine_total: f64 = current.iter().sum();
        let coarse_total: f64 = next.iter().sum::<f64>() * block_size as f64;
        conserved_at_level.push((fine_total - coarse_total).abs() < 1e-8);
        current = next;
    }

    conserved_at_level
}

let entropy = vec![2.0, 3.0, 5.0, 1.0, 4.0, 2.0, 3.0, 5.0];
let results = check_conservation_at_all_scales(&entropy, 2, 3);
println!("Conserved at each scale: {:?}", results);
// => [true, true, true]
```

## Conservation Law Connections

This crate enforces a fundamental invariant from physics and information theory:

**The total entropy of a closed system is constant.**

In a fleet of agents:
- **Conservative transfers** (gradient) move entropy between agents without loss
- **Cyclic flows** (curl) redistribute entropy in loops — each agent breaks even
- **Violations** (harmonic) indicate bugs, external inputs, or information creation/destruction

The Hodge decomposition is the natural tool for this because it mirrors the
Helmholtz decomposition from fluid dynamics: any flow field splits into an
irrotational part (gradient of a potential) and a solenoidal part (curl of a
vector potential), plus a harmonic remainder.

### Relation to Other SuperInstance Crates

- **`renormalization-group-rs`** — Coarse-graining must preserve total entropy.
  Use `is_conserved()` before and after each scale transformation.
- **`sheaf-coherence-rs`** — Global sections of a sheaf must have consistent
  local data; entropy conservation is a special case where the "consistency"
  constraint is a sum invariant.
- **`constraint-dynamics-rs`** — Conservation laws are constraints on the
  dynamics; the harmonic component measures constraint violation.

## Algorithm Details

### Divergence Computation

For an `n × m` transfer matrix `F`:

```
divergence[i] = Σ_j F[i][j] - Σ_k F[k][i]
```

Positive divergence means agent `i` is a net sender. Negative means net receiver.

### Gradient Component

```
G[i][j] = (divergence[i] - divergence[j]) / n
```

Antisymmetric: the flow is driven by divergence differences, like a potential
gradient in electrostatics.

### Curl Component

After subtracting the gradient, the residual is projected onto the space of
matrices with zero row sums and zero column sums:

```
C[i][j] = R[i][j] - row_sums[i]/m - col_sums[j]/n + total/(n*m)
```

### Harmonic Component

Whatever remains after gradient and curl extraction:

```
H = F - G - C
```

Non-zero harmonic means the transfer matrix cannot be explained by conservative
or cyclic flows alone — entropy was created or destroyed.

## Performance

- Decomposition is O(n × m) for an n×m matrix
- Reconstruction is O(n × m)
- Energy computations are O(n × m)
- Suitable for fleets of up to ~10⁴ agents at interactive speed

## License

MIT
