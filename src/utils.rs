use ndarray::Array2;
use petgraph::graph::NodeIndex;

pub fn commodities_from_traffic_matrix(
    node_count: usize,
    traffic_mat: &Array2<f64>,
    fullness_factor: Option<f64>,
) -> Vec<(NodeIndex, NodeIndex)> {
    let mut commodities = Vec::new();
    let factor = fullness_factor.unwrap_or(1.0);

    for r_idx in 0..node_count {
        for c_idx in 0..node_count {
            if traffic_mat[[r_idx, c_idx]] > 1e-9 {
                let should_add = if factor >= 1.0 {
                    true
                } else {
                    rand::random::<f64>() < factor
                };

                if should_add {
                    commodities.push((NodeIndex::new(r_idx), NodeIndex::new(c_idx)));
                }
            }
        }
    }
    commodities
}