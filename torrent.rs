//! BitTorrent piece selection using Kona primitives and data structures.

use crate::k::{K, KData};

/// Select pieces to download based on the specified policy, availability, and weights.
///
/// # Arguments
/// - `peer_bitfields`: A list of bitfields (type 0 general list of int arrays) for all peers,
///   or a single integer array (type 1) representing the combined peer availability or a single peer's bitfield.
/// - `own_bitfield`: An integer array (type 1) of 1s and 0s representing our owned pieces.
/// - `policy`: An integer atom (type -1) representing the selection policy:
///   - `0` = Rarest-First: prioritizes pieces available on the fewest peers (rarest).
///   - `1` = Sequential: downloads pieces in strict index order.
///   - `2` = Priority-Weighted: prioritizes pieces with the highest custom weights.
/// - `custom_weights`: Optional float/integer array (type 2 or 1) containing weights for each piece,
///   used only by policy `2`.
///
/// # Returns
/// An integer array `K` (type 1) containing the indices of pieces to request, sorted by preference.
pub fn select_pieces(
    peer_bitfields: &K,
    own_bitfield: &K,
    policy: &K,
    custom_weights: Option<&K>,
) -> K {
    // 1. Parse policy
    let pol = match &policy.data {
        KData::Ints(v) if !v.is_empty() => v[0],
        _ => panic!("policy must be an integer"),
    };

    // 2. Parse own_bitfield
    let own = own_bitfield.ki_data();
    let num_pieces = own.len();

    // 3. Compute availability from peer_bitfields
    let availability = match peer_bitfields.t {
        0 => {
            // General list of peer bitfields
            let peers = match &peer_bitfields.data {
                KData::List(v) => v,
                _ => unreachable!(),
            };
            if peers.is_empty() {
                vec![0i64; num_pieces]
            } else {
                let mut counts = vec![0i64; num_pieces];
                for peer in peers {
                    let peer_data = peer.ki_data();
                    for i in 0..num_pieces.min(peer_data.len()) {
                        if peer_data[i] > 0 {
                            counts[i] += 1;
                        }
                    }
                }
                counts
            }
        }
        1 | -1 => {
            // A single peer's bitfield or a pre-summed availability array
            let peer_data = peer_bitfields.ki_data();
            let mut counts = vec![0i64; num_pieces];
            for i in 0..num_pieces.min(peer_data.len()) {
                counts[i] = peer_data[i];
            }
            counts
        }
        _ => panic!("peer_bitfields must be a general list or integer array/atom"),
    };

    // 4. Find candidates: pieces we do NOT own (own[i] == 0) and are available on at least one peer (availability[i] > 0)
    let mut candidates = Vec::new();
    for i in 0..num_pieces {
        if own[i] == 0 && availability[i] > 0 {
            candidates.push(i);
        }
    }

    // 5. Sort candidates according to policy
    match pol {
        0 => {
            // Rarest-First: sort by availability ascending, tie-break by index ascending
            candidates.sort_by(|&a, &b| {
                availability[a].cmp(&availability[b])
                    .then_with(|| a.cmp(&b))
            });
        }
        1 => {
            // Sequential: sort by index ascending
            candidates.sort_by(|&a, &b| a.cmp(&b));
        }
        2 => {
            // Priority-Weighted: sort by custom weights descending, tie-break by index ascending
            let mut weights = vec![0.0; num_pieces];
            if let Some(w) = custom_weights {
                match &w.data {
                    KData::Floats(fv) => {
                        for i in 0..num_pieces.min(fv.len()) {
                            weights[i] = fv[i];
                        }
                    }
                    KData::Ints(iv) => {
                        for i in 0..num_pieces.min(iv.len()) {
                            weights[i] = iv[i] as f64;
                        }
                    }
                    _ => panic!("custom_weights must be a float or integer array"),
                }
            }
            candidates.sort_by(|&a, &b| {
                let wa = weights[a];
                let wb = weights[b];
                wb.partial_cmp(&wa)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| availability[a].cmp(&availability[b]))
                    .then_with(|| a.cmp(&b))
            });
        }
        _ => panic!("Unknown policy: {}", pol),
    }

    // 6. Return selected pieces as a K integer array
    let result_ints = candidates.into_iter().map(|idx| idx as i64).collect();
    K::from_ints(result_ints)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rarest_first() {
        // own: 0 1 0 1 (we need piece 0 and 2)
        // peer availability:
        // peer 0: 1 1 1 0
        // peer 1: 1 1 0 0
        // peer 2: 0 1 1 0
        // Total availability:
        // piece 0: 2 (peer 0, peer 1)
        // piece 1: 3
        // piece 2: 2 (peer 0, peer 2)
        // piece 3: 0
        // Rarest-first should request piece 0 and 2 (availability of both is 2).
        // Since both have availability 2, they will sort by index: 0, then 2.
        
        let p0 = K::from_ints(vec![1, 1, 1, 0]);
        let p1 = K::from_ints(vec![1, 1, 0, 0]);
        let p2 = K::from_ints(vec![0, 1, 1, 0]);
        let peers = K::from_list(vec![p0, p1, p2]);
        
        let own = K::from_ints(vec![0, 1, 0, 1]);
        let policy = K::ki(0); // Rarest-First
        
        let selected = select_pieces(&peers, &own, &policy, None);
        assert_eq!(selected.t, 1); // IntArray
        assert_eq!(selected.ki_data(), &[0, 2]);
    }

    #[test]
    fn test_rarest_first_differing_availability() {
        // own: 0 0 0 1 (we need 0, 1, 2)
        // peer availability:
        // piece 0: 3 peers
        // piece 1: 1 peer
        // piece 2: 2 peers
        // piece 3: 0 peers
        // Availability: 3, 1, 2, 0.
        // Rarest-first: piece 1 (avail 1), piece 2 (avail 2), piece 0 (avail 3).
        // Output order: 1, 2, 0.
        let peers_avail = K::from_ints(vec![3, 1, 2, 0]);
        let own = K::from_ints(vec![0, 0, 0, 1]);
        let policy = K::ki(0);
        
        let selected = select_pieces(&peers_avail, &own, &policy, None);
        assert_eq!(selected.ki_data(), &[1, 2, 0]);
    }

    #[test]
    fn test_sequential() {
        // own: 0 0 0 1 (we need 0, 1, 2)
        // Availability: 3, 1, 2, 0.
        // Sequential: 0, 1, 2.
        let peers_avail = K::from_ints(vec![3, 1, 2, 0]);
        let own = K::from_ints(vec![0, 0, 0, 1]);
        let policy = K::ki(1); // Sequential
        
        let selected = select_pieces(&peers_avail, &own, &policy, None);
        assert_eq!(selected.ki_data(), &[0, 1, 2]);
    }

    #[test]
    fn test_priority_weighted() {
        // own: 0 0 0 1 (we need 0, 1, 2)
        // Availability: 3, 1, 2, 0.
        // Weights: 0.1, 0.9, 0.5, 9.9
        // Sorted weights for needed candidates (0, 1, 2):
        // piece 1: weight 0.9
        // piece 2: weight 0.5
        // piece 0: weight 0.1
        // Output order: 1, 2, 0.
        let peers_avail = K::from_ints(vec![3, 1, 2, 0]);
        let own = K::from_ints(vec![0, 0, 0, 1]);
        let policy = K::ki(2); // Priority-Weighted
        let weights = K::from_floats(vec![0.1, 0.9, 0.5, 9.9]);
        
        let selected = select_pieces(&peers_avail, &own, &policy, Some(&weights));
        assert_eq!(selected.ki_data(), &[1, 2, 0]);
    }
}
