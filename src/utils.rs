use ndarray::Array2;
use petgraph::graph::NodeIndex;


// TODO: Random commodities instead of full commodities
pub fn commodities_from_traffic_matrix(
    node_count: usize,
    traffic_mat: &Array2<f64>,
) -> Vec<(NodeIndex, NodeIndex)> {
    let mut commodities = Vec::new();
    for r_idx in 0..node_count {
        for c_idx in 0..node_count {
            if traffic_mat[[r_idx, c_idx]] > 1e-9 {
                commodities.push((NodeIndex::new(r_idx), NodeIndex::new(c_idx)));
            }
        }
    }
    commodities
}
