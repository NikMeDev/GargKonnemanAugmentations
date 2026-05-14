use ndarray::Array2;

pub fn normalize_traffic_matrix(traffic_mat: &mut Array2<f64>) {
    let max_val = traffic_mat.iter().copied().fold(0.0, f64::max);

    if max_val > 0.0 {
        *traffic_mat /= max_val;
    }
}
