# INTEGRATION.md — entropy-conservation-rs × conservation-law-rs × spectral-fleet-rs

**Entropy conservation tracking** uses Hodge–Helmholtz decomposition to split
entropy flows into gradient (conservative), curl (cyclic), and harmonic
(violation) components. It connects to Lagrangian conservation verification
and spectral fleet ranking.

## Synergy Map

```
conservation-law-rs          entropy-conservation-rs         spectral-fleet-rs
┌──────────────────┐        ┌──────────────────────┐       ┌─────────────────┐
│ AgentState        │        │ HodgeComponents      │       │ AdjacencyMatrix │
│ total_energy      │◄──────►│ gradient             │◄─────►│ PowerIteration  │
│ SymplecticIntegrat│        │ curl                 │       │ SpectralRank    │
│ verify_noether    │        │ harmonic             │       │ l2_norm         │
└──────────────────┘        │ decompose            │       └─────────────────┘
                            │ is_conserved         │
                            │ conservation_violation│
                            └──────────────────────┘
```

## Key Insight

Agent fleets generate entropy flows: tokens move, workloads shift, budgets
rebalance. Not all flows conserve total entropy. The Hodge decomposition
separates true conservation violations (harmonic component) from benign
cyclic flows (curl) and zero-sum transfers (gradient). Conservation-law
checks the physics invariant; spectral-fleet ranks agents by the magnitude
of their entropy impact.

## Example 1: Verify Entropy Conservation with Lagrangian Mechanics

Use `conservation-law` to verify that an agent fleet's total energy is
conserved, then use `entropy-conservation` to decompose the same fleet's
entropy flows.

```rust
use conservation_law::lagrangian::{AgentState, MechanicalLagrangian, total_energy};
use entropy_conservation::hodge_decomposition::{decompose, is_conserved};

fn audit_fleet(agent_states: &[AgentState<f64, 2>]) {
    // Step 1: Verify energy conservation via Lagrangian
    let potential = |q: &[f64; 2]| 0.5 * (q[0] * q[0] + q[1] * q[1]);
    let lagrangian = MechanicalLagrangian { mass: 1.0, potential_fn: potential };
    let e0 = total_energy(&lagrangian, &agent_states[0]);
    for state in agent_states {
        let e = total_energy(&lagrangian, state);
        assert!((e - e0).abs() < 1e-6, "energy not conserved");
    }

    // Step 2: Build entropy-change matrix from agent state transitions
    let n = agent_states.len() - 1;
    let mut entropy_changes: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let delta = vec![
            agent_states[i + 1].q[0] - agent_states[i].q[0],
            agent_states[i + 1].q[1] - agent_states[i].q[1],
        ];
        entropy_changes.push(delta);
    }

    // Step 3: Decompose into gradient / curl / harmonic
    let hodge = decompose(&entropy_changes);
    println!("gradient energy = {:.6}", hodge.gradient_energy());
    println!("curl energy     = {:.6}", hodge.curl_energy());
    println!("harmonic energy = {:.6}", hodge.harmonic_energy());

    // Step 4: If harmonic is near zero, entropy is conserved
    if hodge.harmonic_energy() < 1e-8 {
        println!("Fleet entropy is conserved.");
    } else {
        println!("Fleet has {}% genuine entropy creation.",
            100.0 * hodge.harmonic_energy() / hodge.reconstruct().len() as f64);
    }
}
```

## Example 2: Spectral Ranking by Entropy Impact

Build an adjacency matrix from entropy gradients and rank agents by
eigenvector centrality.

```rust
use entropy_conservation::hodge_decomposition::decompose;
use spectral_fleet::power_iteration::PowerIterError;
use spectral_fleet::{l2_norm, normalize};

fn rank_by_entropy_impact(entropy_matrix: &[Vec<f64>]) -> Result<Vec<(usize, f64)>, PowerIterError> {
    let hodge = decompose(entropy_matrix);

    // Use the gradient component (conservative transfers) as affinity
    let n = hodge.gradient.len();
    if n == 0 {
        return Ok(vec![]);
    }

    // Build symmetric affinity from gradient magnitudes
    let mut affinity = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let g = hodge.gradient[i][j];
            affinity[i][j] = (-g * g).exp(); // Gaussian kernel
        }
    }

    // Power iteration on the affinity matrix
    let mut vec = vec![1.0; n];
    normalize(&mut vec);
    for _ in 0..1000 {
        let mut next = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                next[i] += affinity[i][j] * vec[j];
            }
        }
        normalize(&mut next);
        if (l2_norm(&next) - l2_norm(&vec)).abs() < 1e-8 {
            break;
        }
        vec = next;
    }

    let mut ranked: Vec<(usize, f64)> = vec.iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(ranked)
}
```

## Example 3: Conservation Violation Alerting

Monitor a fleet in real time and raise an alert when entropy conservation
is violated beyond a threshold.

```rust
use entropy_conservation::hodge_decomposition::conservation_violation;

struct FleetMonitor {
    baseline: Vec<f64>,
    threshold: f64,
}

impl FleetMonitor {
    fn check(&self, current: &[f64]) -> Option<String> {
        let violation = conservation_violation(&self.baseline, current);
        if violation > self.threshold {
            Some(format!("Conservation violation: {:.4e}", violation))
        } else {
            None
        }
    }
}

fn main() {
    let monitor = FleetMonitor {
        baseline: vec![100.0, 200.0, 300.0],
        threshold: 1e-6,
    };

    let state_a = vec![100.0, 200.0, 300.0];
    let state_b = vec![99.5, 200.5, 300.0]; // small violation

    println!("{:?}", monitor.check(&state_a)); // None
    println!("{:?}", monitor.check(&state_b)); // Some(...)
}
```

## Cargo.toml Wiring

```toml
[dependencies]
entropy-conservation = { git = "https://github.com/SuperInstance/entropy-conservation-rs" }
conservation-law = { git = "https://github.com/SuperInstance/conservation-law-rs" }
spectral-fleet = { git = "https://github.com/SuperInstance/spectral-fleet-rs" }
```
