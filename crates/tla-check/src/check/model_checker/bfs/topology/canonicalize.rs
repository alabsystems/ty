use tla_jit_abi::ActionHomotopy;

/// The Homotopic Canonicalizer.
/// Part of the Geometric Supremacy program (Step 3).
///
/// Instead of hashing the raw memory layout of a state, this component
/// sorts symmetric topological nodes into a canonical orbit representative.
/// This allows TY to achieve zero-cost state deduplication.
#[derive(Clone)]
pub struct HomotopicCanonicalizer {
    /// The indices of state variables that form symmetric groups.
    symmetric_var_groups: Vec<Vec<u16>>,
}

impl HomotopicCanonicalizer {
    pub fn new(evidence: &ActionHomotopy) -> Self {
        let mut symmetric_var_groups: Vec<Vec<u16>> = evidence
            .symmetric_var_groups
            .iter()
            .filter_map(|group| {
                let mut group = group.clone();
                group.sort_unstable();
                group.dedup();
                (group.len() > 1).then_some(group)
            })
            .collect();
        symmetric_var_groups.sort_unstable();
        symmetric_var_groups.dedup();

        Self {
            symmetric_var_groups,
        }
    }

    /// True when this canonicalizer can never rewrite a buffer (no symmetric
    /// var groups survived construction — `canonicalize_in_place` is a no-op).
    ///
    /// WP-11 slice-1 fence: `TopologyAnalyzer::analyze_stability` deliberately
    /// emits NO `symmetric_var_groups` (its sort-based collapse is not
    /// lexmin-over-the-declared-group and must not be re-armed outside the
    /// verified `state::flat_symmetry` machinery), so every
    /// production-constructed canonicalizer reports `true` here. The compiled
    /// BFS fingerprint hook `debug_assert!`s this — see
    /// `FlatBufferCanonicalizationAuthority` in `model_checker/fingerprint.rs`.
    #[must_use]
    pub fn is_inert(&self) -> bool {
        self.symmetric_var_groups.is_empty()
    }

    /// Canonicalizes a flat JIT state buffer in-place.
    ///
    /// Complexity: O(V log V) where V is the number of symmetric variables.
    /// This replaces the O(N!) factorial brute-force symmetry searches
    /// used by legacy model checkers.
    ///
    /// Part of Geometric Supremacy: uses a scratch buffer to avoid per-state allocations.
    pub fn canonicalize_in_place(&self, flat_state: &mut [i64], scratch: &mut Vec<i64>) {
        if self.symmetric_var_groups.is_empty() {
            return; // Fast path: no symmetry to collapse
        }

        for group in &self.symmetric_var_groups {
            if !group.iter().all(|&idx| (idx as usize) < flat_state.len()) {
                continue;
            }

            // Fast path for common small groups (Reviewer A: Sorting Tax).
            if group.len() == 2 {
                let v0 = flat_state[group[0] as usize];
                let v1 = flat_state[group[1] as usize];
                if v0 > v1 {
                    flat_state[group[0] as usize] = v1;
                    flat_state[group[1] as usize] = v0;
                }
                continue;
            }

            // Reuse scratch buffer to avoid per-state allocation.
            scratch.clear();
            for &idx in group {
                scratch.push(flat_state[idx as usize]);
            }

            scratch.sort_unstable();

            for (i, &idx) in group.iter().enumerate() {
                flat_state[idx as usize] = scratch[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(groups: Vec<Vec<u16>>) -> ActionHomotopy {
        ActionHomotopy::new("Next", true, vec!["Node".to_string()], groups)
    }

    #[test]
    fn action_homotopy_serde_defaults_symmetric_var_groups_for_legacy_evidence() {
        let json = r#"{
            "action_name": "Next",
            "is_stable": true,
            "symmetry_groups": ["Node"]
        }"#;

        let evidence: ActionHomotopy = serde_json::from_str(json).unwrap();

        assert_eq!(evidence.action_name, "Next");
        assert!(evidence.is_stable);
        assert_eq!(evidence.symmetry_groups, vec!["Node"]);
        assert!(evidence.symmetric_var_groups.is_empty());
    }

    #[test]
    fn action_homotopy_serde_roundtrips_symmetric_var_groups() {
        let evidence = ActionHomotopy::new(
            "Next",
            true,
            vec!["Node".to_string()],
            vec![vec![0, 2], vec![3, 4]],
        );

        let encoded = serde_json::to_string(&evidence).unwrap();
        let decoded: ActionHomotopy = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, evidence);
    }

    #[test]
    fn inert_fence_reports_correctly() {
        // WP-11 slice-1 fence: production evidence (TopologyAnalyzer) always
        // carries empty symmetric_var_groups, so the constructed canonicalizer
        // must be inert; a non-empty multi-slot group would arm the (unsound
        // for orbit dedup) sort-collapse and MUST report non-inert so the
        // fingerprint hook's debug_assert trips.
        assert!(HomotopicCanonicalizer::new(&evidence(vec![])).is_inert());
        assert!(
            HomotopicCanonicalizer::new(&evidence(vec![vec![], vec![3], vec![2, 2]])).is_inert(),
            "degenerate groups are dropped at construction and stay inert"
        );
        assert!(
            !HomotopicCanonicalizer::new(&evidence(vec![vec![0, 1]])).is_inert(),
            "a live multi-slot group must be visible to the fence"
        );
    }

    #[test]
    fn canonicalize_sorts_symmetric_groups_by_state_slot() {
        let canonicalizer = HomotopicCanonicalizer::new(&evidence(vec![vec![2, 0, 1]]));
        let mut state = vec![30, 10, 20, 99];
        let mut scratch = Vec::new();

        canonicalizer.canonicalize_in_place(&mut state, &mut scratch);

        assert_eq!(state, vec![10, 20, 30, 99]);
    }

    #[test]
    fn canonicalize_skips_out_of_range_groups() {
        let canonicalizer = HomotopicCanonicalizer::new(&evidence(vec![vec![0, 99]]));
        let mut state = vec![2, 1];
        let mut scratch = Vec::new();

        canonicalizer.canonicalize_in_place(&mut state, &mut scratch);

        assert_eq!(state, vec![2, 1]);
    }

    #[test]
    fn canonicalize_ignores_empty_singleton_and_duplicate_groups() {
        let canonicalizer =
            HomotopicCanonicalizer::new(&evidence(vec![vec![], vec![1], vec![0, 0, 1]]));
        let mut state = vec![9, 3, 7];
        let mut scratch = Vec::new();

        canonicalizer.canonicalize_in_place(&mut state, &mut scratch);

        assert_eq!(state, vec![3, 9, 7]);
    }
}
