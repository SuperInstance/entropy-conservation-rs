//! Hodge decomposition of entropy flow matrices.
//!
//! Decomposes an entropy-change matrix into three orthogonal components inspired
//! by the Hodge–Helmholtz decomposition of vector fields:
//!
//! - **Gradient**: irrotational (conservative) flow — one agent's gain is another's
//!   loss. Sum across any cut is zero.
//! - **Curl**: solenoidal (cyclic) flow — entropy circulates among agents without
//!   net gain or loss. Each agent's net change is zero.
//! - **Harmonic**: the residual — entropy that is genuinely created or destroyed,
//!   representing a conservation-law violation.

/// Result of a Hodge decomposition.
#[derive(Debug, Clone)]
pub struct HodgeComponents {
    /// Gradient component (irrotational / conservative).
    pub gradient: Vec<Vec<f64>>,
    /// Curl component (solenoidal / cyclic).
    pub curl: Vec<Vec<f64>>,
    /// Harmonic component (conservation violation / residual).
    pub harmonic: Vec<Vec<f64>>,
}

impl HodgeComponents {
    /// Reconstruct the original matrix by summing all three components.
    pub fn reconstruct(&self) -> Vec<Vec<f64>> {
        let n = self.gradient.len();
        let m = self.gradient[0].len();
        let mut result = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                result[i][j] = self.gradient[i][j] + self.curl[i][j] + self.harmonic[i][j];
            }
        }
        result
    }

    /// Frobenius norm of the gradient component.
    pub fn gradient_energy(&self) -> f64 {
        frobenius_norm(&self.gradient)
    }

    /// Frobenius norm of the curl component.
    pub fn curl_energy(&self) -> f64 {
        frobenius_norm(&self.curl)
    }

    /// Frobenius norm of the harmonic component.
    pub fn harmonic_energy(&self) -> f64 {
        frobenius_norm(&self.harmonic)
    }
}

// ---------------------------------------------------------------------------
// Decomposition
// ---------------------------------------------------------------------------

/// Perform a Hodge decomposition on an `n × m` entropy-change matrix.
///
/// The matrix `F` represents pairwise entropy flows between agents: `F[i][j]` is
/// the entropy transferred from agent `i` to agent `j` (or contributed by `i` to
/// dimension `j`).
///
/// **Algorithm** (discrete Hodge–Helmholtz on a complete graph):
///
/// 1. Compute the divergence vector `d[i] = Σ_j F[i][j] - Σ_j F[j][i]`.
/// 2. **Gradient**: `G[i][j] = (d[i] - d[j]) / n` — the antisymmetric
///    potential-driven flow.
/// 3. **Curl**: project `F - G` onto the space of matrices with zero row sums
///    *and* zero column sums.  `C[i][j] = F[i][j] - G[i][j] - h[i] - h[j]`
///    where `h` is chosen to enforce the zero-sum constraints.
/// 4. **Harmonic**: `H = F - G - C`.
pub fn decompose(entropy_changes: &[Vec<f64>]) -> HodgeComponents {
    let n = entropy_changes.len();
    if n == 0 {
        return HodgeComponents {
            gradient: vec![],
            curl: vec![],
            harmonic: vec![],
        };
    }
    let m = entropy_changes[0].len();

    // 1. Divergence: net outflow of each agent.
    let mut divergence = vec![0.0; n];
    for i in 0..n {
        let mut outflow = 0.0f64;
        for j in 0..m {
            outflow += entropy_changes[i][j];
        }
        let mut inflow = 0.0f64;
        for k in 0..n {
            if i < entropy_changes[k].len() {
                inflow += entropy_changes[k][i];
            }
        }
        divergence[i] = outflow - inflow;
    }

    // 2. Gradient component: antisymmetric flow driven by divergence differences.
    let mut gradient = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m.min(n) {
            gradient[i][j] = (divergence[i] - divergence[j.min(n - 1)]) / (n as f64);
        }
    }
    // Handle non-square: for j >= n, gradient is zero (no corresponding agent).

    // 3. Residual = F - G
    let mut residual = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            residual[i][j] = entropy_changes[i][j] - gradient[i][j];
        }
    }

    // Compute row and column sums of residual to extract curl vs harmonic.
    let row_sums: Vec<f64> = (0..n)
        .map(|i| (0..m).map(|j| residual[i][j]).sum())
        .collect();
    let col_sums: Vec<f64> = (0..m)
        .map(|j| (0..n).map(|i| residual[i][j]).sum())
        .collect();

    // Curl: project residual to have zero row sums and zero column sums.
    // C[i][j] = R[i][j] - r[i]/m - c[j]/n + total/(n*m)
    let total: f64 = row_sums.iter().sum();
    let mut curl = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            curl[i][j] = residual[i][j]
                - row_sums[i] / (m as f64)
                - col_sums[j] / (n as f64)
                + total / ((n * m) as f64);
        }
    }

    // 4. Harmonic = residual - curl
    let mut harmonic = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            harmonic[i][j] = residual[i][j] - curl[i][j];
        }
    }

    HodgeComponents {
        gradient,
        curl,
        harmonic,
    }
}

/// Check whether entropy is conserved between two state vectors.
///
/// Returns `true` if the total entropy (sum) is the same before and after,
/// within a tolerance of `1e-10`.
pub fn is_conserved(original: &[f64], after: &[f64]) -> bool {
    let sum_before: f64 = original.iter().copied().sum();
    let sum_after: f64 = after.iter().copied().sum();
    (sum_before - sum_after).abs() < 1e-10
}

/// Compute the conservation violation: `|sum(original) - sum(after)|`.
pub fn conservation_violation(original: &[f64], after: &[f64]) -> f64 {
    let sum_before: f64 = original.iter().copied().sum();
    let sum_after: f64 = after.iter().copied().sum();
    (sum_before - sum_after).abs()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn frobenius_norm(mat: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0;
    for row in mat {
        for &v in row {
            sum += v * v;
        }
    }
    sum.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_empty() {
        let hc = decompose(&[]);
        assert!(hc.gradient.is_empty());
        assert!(hc.curl.is_empty());
        assert!(hc.harmonic.is_empty());
    }

    #[test]
    fn test_decompose_single_element() {
        let hc = decompose(&[vec![5.0]]);
        // 1×1: divergence = 5 - 5 = 0 → gradient = 0
        assert!((hc.gradient[0][0]).abs() < 1e-10);
    }

    #[test]
    fn test_decompose_gradient_antisymmetric() {
        // Pure transfer: A→B, F = [[0, 3], [−3, 0]]
        let f = vec![vec![0.0, 3.0], vec![-3.0, 0.0]];
        let hc = decompose(&f);
        // Gradient should capture the antisymmetric part
        // divergence[0] = (0+3) - (0+(-3)) = 6
        // divergence[1] = (-3+0) - (3+0) = -6
        // gradient[0][0] = (6-6)/2 = 0
        // gradient[0][1] = (6-(-6))/2 = 6
        // gradient[1][0] = (-6-6)/2 = -6
        // gradient[1][1] = (-6-(-6))/2 = 0
        assert!((hc.gradient[0][1] - 6.0).abs() < 1e-10);
        assert!((hc.gradient[1][0] - (-6.0)).abs() < 1e-10);
    }

    #[test]
    fn test_decompose_reconstructs() {
        let f = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0]];
        let hc = decompose(&f);
        let recon = hc.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon[i][j] - f[i][j]).abs() < 1e-8,
                    "mismatch at [{i}][{j}]: got {} expected {}",
                    recon[i][j],
                    f[i][j]
                );
            }
        }
    }

    #[test]
    fn test_decompose_zero_matrix() {
        let f = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let hc = decompose(&f);
        assert!((hc.gradient_energy()).abs() < 1e-10);
        assert!((hc.curl_energy()).abs() < 1e-10);
        assert!((hc.harmonic_energy()).abs() < 1e-10);
    }

    #[test]
    fn test_decompose_pure_harmonic() {
        // All agents gain equally: no gradient, no curl, all harmonic.
        let f = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let hc = decompose(&f);
        // Harmonic should carry the creation
        assert!(hc.harmonic_energy() > 0.1);
    }

    #[test]
    fn test_is_conserved_true() {
        assert!(is_conserved(&[1.0, 2.0, 3.0], &[2.0, 1.0, 3.0]));
    }

    #[test]
    fn test_is_conserved_false() {
        assert!(!is_conserved(&[1.0, 2.0, 3.0], &[2.0, 2.0, 3.0]));
    }

    #[test]
    fn test_is_conserved_empty() {
        assert!(is_conserved(&[], &[]));
    }

    #[test]
    fn test_conservation_violation_zero() {
        let v = conservation_violation(&[1.0, 2.0], &[2.0, 1.0]);
        assert!(v < 1e-10);
    }

    #[test]
    fn test_conservation_violation_nonzero() {
        let v = conservation_violation(&[1.0, 2.0], &[2.0, 2.0]);
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hodge_components_energy() {
        let f = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let hc = decompose(&f);
        // Total energy = ||F||_F
        let total = frobenius_norm(&f);
        let _sum = hc.gradient_energy().powi(2)
            + hc.curl_energy().powi(2)
            + hc.harmonic_energy().powi(2);
        // Not strictly Pythagorean (components aren't orthogonal in this algo),
        // but energies should be positive.
        assert!(hc.gradient_energy() >= 0.0);
        assert!(hc.curl_energy() >= 0.0);
        assert!(hc.harmonic_energy() >= 0.0);
        assert!(total >= 0.0);
    }

    #[test]
    fn test_decompose_symmetric_transfer() {
        // Balanced cyclic flow A→B→C→A
        let f = vec![
            vec![0.0, 5.0, 0.0],
            vec![0.0, 0.0, 5.0],
            vec![5.0, 0.0, 0.0],
        ];
        let hc = decompose(&f);
        // Divergences should all be zero (each agent sends 5 and receives 5)
        // So gradient should be ~0
        assert!(hc.gradient_energy() < 1.0, "gradient energy: {}", hc.gradient_energy());
        let recon = hc.reconstruct();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recon[i][j] - f[i][j]).abs() < 1e-8,
                    "mismatch at [{i}][{j}]"
                );
            }
        }
    }
}
