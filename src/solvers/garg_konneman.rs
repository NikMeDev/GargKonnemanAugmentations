use anyhow::Result;
use anyhow::anyhow;
use gxhash::HashMap;
use gxhash::HashMapExt;
use petgraph::algo::astar;
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::EdgeRef;
use rayon::prelude::*;
use serde::Serialize;

pub type GraphType = DiGraph<String, f64>;

fn get_normalization_factor(graph: &GraphType, f_edge_flows: &HashMap<EdgeIndex, f64>) -> f64 {
    let mut c_normalization_factor = 0.0;
    for edge_ref in graph.edge_references() {
        let capacity_u_ij = *edge_ref.weight();
        if capacity_u_ij <= 1e-9 {
            continue;
        }

        let flow_f_ij = f_edge_flows.get(&edge_ref.id()).unwrap_or(&0.0);
        let ratio = flow_f_ij / capacity_u_ij;
        if ratio > c_normalization_factor {
            c_normalization_factor = ratio;
        }
    }
    c_normalization_factor
}

fn get_flow_sum(x_path_flows: &HashMap<Vec<NodeIndex>, f64>, c_normalization_factor: f64) -> f64 {
    x_path_flows
        .values()
        .map(|&flow_val| flow_val / c_normalization_factor)
        .sum()
}

fn normalize_flows(x_path_flows: &mut HashMap<Vec<NodeIndex>, f64>, c_normalization_factor: f64) {
    if c_normalization_factor <= 1e-9 {
        for flow_val in x_path_flows.values_mut() {
            *flow_val = 0.0;
        }
    } else {
        for flow_val in x_path_flows.values_mut() {
            *flow_val /= c_normalization_factor;
        }
    }
}

fn is_close_to_true(current_flow: f64, target_flow: f64, epsilon: f64) {
    target_flow / current_flow <= 1.0 + epsilon
}

fn should_log(iteration: usize) -> bool {
    // log if iteration is square of whole number
    let sqrt = (iteration as f64).sqrt();
    let sqrt_int = sqrt as usize;
    let sqrt_int_squared = (sqrt_int * sqrt_int) as usize;
    iteration == sqrt_int_squared
}

#[derive(Debug, Clone, Serialize)]
pub struct IterationInfo {
    pub iteration: usize,
    pub current_flow_sum: f64,
    pub elapsed_time: std::time::Duration,
}

pub fn garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    let mut x_path_flows: HashMap<Vec<NodeIndex>, f64> = HashMap::new();
    let mut f_edge_flows: HashMap<EdgeIndex, f64> = HashMap::new();
    let mut w_edge_costs: HashMap<EdgeIndex, f64> = HashMap::new();

    for edge_ref in graph.edge_references() {
        f_edge_flows.insert(edge_ref.id(), 0.0);
        w_edge_costs.insert(edge_ref.id(), 1.0);
    }

    let m = graph.edge_count();

    let threshold = (m as f64).ln() / (epsilon * epsilon);

    let mut history: Vec<IterationInfo> = Vec::new();

    let mut iteration = 0;
    let start_time = std::time::SystemTime::now();

    'outer: loop {
        for edge_ref in graph.edge_references() {
            let capacity_u_ij = *edge_ref.weight();
            if capacity_u_ij <= 1e-9 {
                continue;
            }
            let flow_f_ij = f_edge_flows.get(&edge_ref.id()).unwrap_or(&0.0);
            if flow_f_ij / capacity_u_ij >= threshold {
                break 'outer;
            }
        }

        let mut current_best_path_nodes: Option<Vec<NodeIndex>> = None;
        let mut min_path_overall_cost = f64::INFINITY;

        for &(source_node, target_node) in commodities {
            if source_node == target_node {
                continue;
            }

            let path_search_result = astar(
                graph,
                source_node,
                |finish_node| finish_node == target_node,
                |edge_ref| {
                    let capacity_u_e = *edge_ref.weight();
                    if capacity_u_e <= 1e-9 {
                        return f64::INFINITY;
                    }
                    w_edge_costs.get(&edge_ref.id()).unwrap_or(&1.0) / capacity_u_e
                },
                |_| 0.0,
            );

            if let Some((path_cost, node_list_path)) = path_search_result {
                if path_cost < min_path_overall_cost && path_cost.is_finite() {
                    min_path_overall_cost = path_cost;
                    current_best_path_nodes = Some(node_list_path);
                }
            }
        }

        let p_star_nodes = current_best_path_nodes.unwrap();

        let mut p_star_edges: Vec<EdgeIndex> = Vec::with_capacity(p_star_nodes.len() - 1);
        let mut u_bottleneck_capacity = f64::INFINITY;

        for i in 0..(p_star_nodes.len() - 1) {
            let node_u_idx = p_star_nodes[i];
            let node_v_idx = p_star_nodes[i + 1];

            let edge_id = graph.find_edge(node_u_idx, node_v_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge from path {:?} -> {:?} not found in graph",
                    node_u_idx,
                    node_v_idx
                )
            })?;
            p_star_edges.push(edge_id);

            let edge_capacity_u_ij = *graph.edge_weight(edge_id).unwrap();
            if edge_capacity_u_ij < u_bottleneck_capacity {
                u_bottleneck_capacity = edge_capacity_u_ij;
            }
        }

        *x_path_flows.entry(p_star_nodes.clone()).or_insert(0.0) += u_bottleneck_capacity;

        for &edge_id in &p_star_edges {
            *f_edge_flows.get_mut(&edge_id).unwrap() += u_bottleneck_capacity;

            let capacity_u_e = *graph.edge_weight(edge_id).unwrap();
            let cost_w_e = w_edge_costs.get_mut(&edge_id).unwrap();
            *cost_w_e *= 1.0 + (epsilon * u_bottleneck_capacity / capacity_u_e);
        }

        let c_normalization_factor = get_normalization_factor(graph, &f_edge_flows);
        let current_flow_sum: f64 = get_flow_sum(&x_path_flows, c_normalization_factor);
        let elapsed_time = start_time.elapsed().unwrap();

        if should_log(iteration) {
            history.push(IterationInfo {
                iteration,
                current_flow_sum,
                elapsed_time,
            });
        }

        if let Some(target_flow_val) = target_flow {
            if is_close_to_true(current_flow_sum, target_flow_val, epsilon) {
                break;
            }
        }

        if iteration % 10000 == 0 {
            println!(
                "Iteration {}: Current flow scaled sum: {} \t Current elapsed: {:?}",
                iteration, current_flow_sum, elapsed_time,
            );
        }
        iteration += 1;
    }

    normalize_flows(
        &mut x_path_flows,
        get_normalization_factor(graph, &f_edge_flows),
    );
    println!(
        "Total time: {:?}\nTotal Iterations: {}",
        start_time.elapsed().unwrap(),
        iteration
    );
    Ok((x_path_flows, history))
}

pub fn par_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    let mut x_path_flows: HashMap<Vec<NodeIndex>, f64> = HashMap::new();
    let mut f_edge_flows: HashMap<EdgeIndex, f64> = HashMap::new();
    let mut w_edge_costs: HashMap<EdgeIndex, f64> = HashMap::new();

    for edge_ref in graph.edge_references() {
        f_edge_flows.insert(edge_ref.id(), 0.0);
        w_edge_costs.insert(edge_ref.id(), 1.0);
    }

    let m = graph.edge_count();

    let threshold = (m as f64).ln() / (epsilon * epsilon);

    let mut history: Vec<IterationInfo> = Vec::new();

    let mut iteration = 0;
    let start_time = std::time::SystemTime::now();

    'outer: loop {
        for edge_ref in graph.edge_references() {
            let capacity_u_ij = *edge_ref.weight();
            if capacity_u_ij <= 1e-9 {
                continue;
            }
            let flow_f_ij = f_edge_flows.get(&edge_ref.id()).unwrap_or(&0.0);
            if flow_f_ij / capacity_u_ij >= threshold {
                break 'outer;
            }
        }

        let best_result = commodities
            .par_iter()
            .map(|&(source_node, target_node)| {
                if source_node == target_node {
                    return None;
                }

                let path_search_result = astar(
                    graph,
                    source_node,
                    |finish_node| finish_node == target_node,
                    |edge_ref| {
                        let capacity_u_e = *edge_ref.weight();
                        if capacity_u_e <= 1e-9 {
                            return f64::INFINITY;
                        }
                        w_edge_costs.get(&edge_ref.id()).unwrap_or(&1.0) / capacity_u_e
                    },
                    |_| 0.0,
                );
                path_search_result
            })
            .reduce(
                || None,
                |a, b| {
                    if a.is_none() {
                        return b;
                    }
                    if b.is_none() {
                        return a;
                    }
                    let (cost_a, path_a) = a.unwrap();
                    let (cost_b, path_b) = b.unwrap();
                    if cost_a < cost_b {
                        Some((cost_a, path_a))
                    } else {
                        Some((cost_b, path_b))
                    }
                },
            );
        let (_, p_star_nodes) = best_result.unwrap();

        let mut p_star_edges: Vec<EdgeIndex> = Vec::with_capacity(p_star_nodes.len() - 1);
        let mut u_bottleneck_capacity = f64::INFINITY;

        for i in 0..(p_star_nodes.len() - 1) {
            let node_u_idx = p_star_nodes[i];
            let node_v_idx = p_star_nodes[i + 1];

            let edge_id = graph.find_edge(node_u_idx, node_v_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge from path {:?} -> {:?} not found in graph",
                    node_u_idx,
                    node_v_idx
                )
            })?;
            p_star_edges.push(edge_id);

            let edge_capacity_u_ij = *graph.edge_weight(edge_id).unwrap();
            if edge_capacity_u_ij < u_bottleneck_capacity {
                u_bottleneck_capacity = edge_capacity_u_ij;
            }
        }

        *x_path_flows.entry(p_star_nodes.clone()).or_insert(0.0) += u_bottleneck_capacity;

        for &edge_id in &p_star_edges {
            *f_edge_flows.get_mut(&edge_id).unwrap() += u_bottleneck_capacity;

            let capacity_u_e = *graph.edge_weight(edge_id).unwrap();
            let cost_w_e = w_edge_costs.get_mut(&edge_id).unwrap();
            *cost_w_e *= 1.0 + (epsilon * u_bottleneck_capacity / capacity_u_e);
        }

        let c_normalization_factor = get_normalization_factor(graph, &f_edge_flows);
        let current_flow_sum: f64 = get_flow_sum(&x_path_flows, c_normalization_factor);
        let elapsed_time = start_time.elapsed().unwrap();

        if should_log(iteration) {
            history.push(IterationInfo {
                iteration,
                current_flow_sum,
                elapsed_time,
            });
        }

        if let Some(target_flow_val) = target_flow {
            if is_close_to_true(current_flow_sum, target_flow_val, epsilon) {
                break;
            }
        }

        if iteration % 10000 == 0 {
            println!(
                "Iteration {}: Current flow scaled sum: {} \t Current elapsed: {:?}",
                iteration, current_flow_sum, elapsed_time,
            );
        }
        iteration += 1;
    }

    normalize_flows(
        &mut x_path_flows,
        get_normalization_factor(graph, &f_edge_flows),
    );
    println!(
        "Total time: {:?}\nTotal Iterations: {}",
        start_time.elapsed().unwrap(),
        iteration
    );
    Ok((x_path_flows, history))
}

pub fn fleischer_fptas_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    let delta = (1.0f64 + epsilon)
        * ((1.0f64 + epsilon) * (graph.node_count() as f64)).powf(-1.0f64 / epsilon);

    let mut l_edge_lengths: HashMap<EdgeIndex, f64> = graph
        .edge_references()
        .map(|edge_ref| (edge_ref.id(), delta))
        .collect();

    let mut f_edge_flows: HashMap<EdgeIndex, f64> = graph
        .edge_references()
        .map(|edge_ref| (edge_ref.id(), 0.0))
        .collect();

    let mut x_path_flows: HashMap<Vec<NodeIndex>, f64> = HashMap::new();
    let mut history: Vec<IterationInfo> = Vec::new();
    let start_time = std::time::SystemTime::now();

    let log_val = ((1.0f64 + epsilon) / delta).ln();
    let log_base = (1.0f64 + epsilon).ln();

    let r_max = (log_val / log_base).ceil() as usize;

    let mut iteration = 0;

    for r_iter_val in 1..=r_max {
        for &(source_node, target_node) in commodities {
            if source_node == target_node {
                continue;
            }

            let path_search_initial = astar(
                graph,
                source_node,
                |finish_node| finish_node == target_node,
                |edge_ref| {
                    *l_edge_lengths
                        .get(&edge_ref.id())
                        .expect("Edge length missing for A* (initial path search)")
                },
                |_| 0.0,
            );

            iteration += 1;

            if let Some((mut current_path_cost, mut current_path_nodes)) = path_search_initial {
                if current_path_nodes.len() < 2 {
                    continue;
                }

                let threshold_r = (1.0f64).min(delta * (1.0 + epsilon).powi(r_iter_val as i32));

                while current_path_cost < threshold_r {
                    let mut u_bottleneck = f64::INFINITY;
                    let mut path_edges_details: Vec<(EdgeIndex, f64)> =
                        Vec::with_capacity(current_path_nodes.len().saturating_sub(1));

                    for i in 0..(current_path_nodes.len() - 1) {
                        let u_idx = current_path_nodes[i];
                        let v_idx = current_path_nodes[i + 1];

                        let edge_id = graph.find_edge(u_idx, v_idx).ok_or_else(|| {
                            anyhow!(
                                "Edge from path {:?} -> {:?} not found in graph",
                                u_idx,
                                v_idx
                            )
                        })?;
                        let capacity_u_e = *graph.edge_weight(edge_id).ok_or_else(|| {
                            anyhow!(
                                "Capacity for edge {:?} (path component {:?}->{:?}) not found",
                                edge_id,
                                u_idx,
                                v_idx
                            )
                        })?;

                        if capacity_u_e < u_bottleneck {
                            u_bottleneck = capacity_u_e;
                        }
                        path_edges_details.push((edge_id, capacity_u_e));
                    }

                    if u_bottleneck <= 1e-9 || path_edges_details.is_empty() {
                        break;
                    }

                    *x_path_flows
                        .entry(current_path_nodes.clone())
                        .or_insert(0.0) += u_bottleneck;

                    for (edge_id, capacity_u_e) in &path_edges_details {
                        *f_edge_flows.get_mut(edge_id).unwrap() += u_bottleneck;
                        let l_e = l_edge_lengths.get_mut(edge_id).unwrap();
                        *l_e *= 1.0 + (epsilon * u_bottleneck / *capacity_u_e);
                    }

                    let path_search_next = astar(
                        graph,
                        source_node,
                        |finish_node| finish_node == target_node,
                        |edge_ref| {
                            *l_edge_lengths
                                .get(&edge_ref.id())
                                .expect("Edge length missing for A* (while loop path search)")
                        },
                        |_| 0.0,
                    );

                    iteration += 1;

                    if let Some((next_cost, next_nodes)) = path_search_next {
                        if next_nodes.len() < 2 {
                            current_path_cost = f64::INFINITY;
                        } else {
                            current_path_cost = next_cost;
                            current_path_nodes = next_nodes;
                        }
                    } else {
                        current_path_cost = f64::INFINITY;
                    }
                }
            }
        }

        let c_normalization_factor = get_normalization_factor(graph, &f_edge_flows);
        let current_flow_sum: f64 = get_flow_sum(&x_path_flows, c_normalization_factor);
        let elapsed_time = start_time.elapsed().unwrap();

        if should_log(iteration) {
            history.push(IterationInfo {
                iteration,
                current_flow_sum,
                elapsed_time,
            });
        }

        if let Some(target_flow_val) = target_flow {
            if is_close_to_true(current_flow_sum, target_flow_val, epsilon) {
                break;
            }
        }

        if iteration % 10000 == 0 {
            println!(
                "FPTAS r-iteration {}: Current total flow sum: {:.6} \t Elapsed: {:?}",
                r_iter_val, current_flow_sum, elapsed_time
            );
        }
    }

    normalize_flows(
        &mut x_path_flows,
        get_normalization_factor(graph, &f_edge_flows),
    );

    Ok((x_path_flows, history))
}

const ADAGRAD_FUDGE_FACTOR: f64 = 1e-8;

pub fn adaptive_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    let mut x_path_flows: HashMap<Vec<NodeIndex>, f64> = HashMap::new();
    let mut f_edge_flows: HashMap<EdgeIndex, f64> = HashMap::new();
    let mut w_edge_costs: HashMap<EdgeIndex, f64> = HashMap::new();
    let mut h_edge_accumulators: HashMap<EdgeIndex, f64> = HashMap::new();

    for edge_ref in graph.edge_references() {
        f_edge_flows.insert(edge_ref.id(), 0.0);
        w_edge_costs.insert(edge_ref.id(), 1.0);
        h_edge_accumulators.insert(edge_ref.id(), 0.0);
    }

    let m = graph.edge_count();
    let threshold = (m as f64).ln() / (epsilon * epsilon);

    let mut history: Vec<IterationInfo> = Vec::new();
    let mut iteration = 0;
    let start_time = std::time::SystemTime::now();

    'outer: loop {
        for edge_ref in graph.edge_references() {
            let capacity_u_ij = *edge_ref.weight();
            if capacity_u_ij <= 1e-9 {
                continue;
            }
            let flow_f_ij = f_edge_flows.get(&edge_ref.id()).unwrap_or(&0.0);
            if flow_f_ij / capacity_u_ij >= threshold {
                break 'outer;
            }
        }

        let mut current_best_path_nodes: Option<Vec<NodeIndex>> = None;
        let mut min_path_overall_cost = f64::INFINITY;

        for &(source_node, target_node) in commodities {
            if source_node == target_node {
                continue;
            }

            let path_search_result = astar(
                graph,
                source_node,
                |finish_node| finish_node == target_node,
                |edge_ref| {
                    let capacity_u_e = *edge_ref.weight();
                    if capacity_u_e <= 1e-9 {
                        return f64::INFINITY;
                    }
                    w_edge_costs.get(&edge_ref.id()).unwrap_or(&1.0) / capacity_u_e
                },
                |_| 0.0,
            );

            if let Some((path_cost, node_list_path)) = path_search_result {
                if path_cost < min_path_overall_cost && path_cost.is_finite() {
                    min_path_overall_cost = path_cost;
                    current_best_path_nodes = Some(node_list_path);
                }
            }
        }

        let p_star_nodes = current_best_path_nodes.unwrap();

        let mut p_star_edges_details: Vec<(EdgeIndex, f64)> =
            Vec::with_capacity(p_star_nodes.len().saturating_sub(1));
        let mut u_bottleneck_capacity = f64::INFINITY;

        for i in 0..(p_star_nodes.len() - 1) {
            let node_u_idx = p_star_nodes[i];
            let node_v_idx = p_star_nodes[i + 1];

            let edge_id = graph.find_edge(node_u_idx, node_v_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge from path {:?} -> {:?} not found in graph",
                    node_u_idx,
                    node_v_idx
                )
            })?;

            let edge_capacity_u_ij = *graph.edge_weight(edge_id).unwrap();
            p_star_edges_details.push((edge_id, edge_capacity_u_ij));

            if edge_capacity_u_ij < u_bottleneck_capacity {
                u_bottleneck_capacity = edge_capacity_u_ij;
            }
        }

        *x_path_flows.entry(p_star_nodes.clone()).or_insert(0.0) += u_bottleneck_capacity;

        for &(edge_id, capacity_u_e) in &p_star_edges_details {
            *f_edge_flows.get_mut(&edge_id).unwrap() += u_bottleneck_capacity;

            if capacity_u_e <= 1e-9 {
                continue;
            }

            let h_e_current = u_bottleneck_capacity / capacity_u_e;

            let h_accumulator_past = h_edge_accumulators.get(&edge_id).unwrap_or(&0.0);
            let eta_scalar = epsilon / (h_accumulator_past + ADAGRAD_FUDGE_FACTOR).sqrt();

            let cost_w_e = w_edge_costs.get_mut(&edge_id).unwrap();
            *cost_w_e *= 1.0 + (eta_scalar * h_e_current);

            *h_edge_accumulators.get_mut(&edge_id).unwrap() += h_e_current * h_e_current;
        }

        let c_normalization_factor = get_normalization_factor(graph, &f_edge_flows);
        let current_flow_sum: f64 = get_flow_sum(&x_path_flows, c_normalization_factor);
        let elapsed_time = start_time.elapsed().unwrap();

        if should_log(iteration) {
            history.push(IterationInfo {
                iteration,
                current_flow_sum,
                elapsed_time,
            });
        }

        if let Some(target_flow_val) = target_flow {
            if is_close_to_true(current_flow_sum, target_flow_val, epsilon) {
                break;
            }
        }

        if iteration % 10000 == 0 {
            println!(
                "Adaptive Iteration {}: Current flow scaled sum: {} \t Current elapsed: {:?}",
                iteration, current_flow_sum, elapsed_time,
            );
        }
        iteration += 1;
    }

    normalize_flows(
        &mut x_path_flows,
        get_normalization_factor(graph, &f_edge_flows),
    );
    println!(
        "Adaptive Total time: {:?}\nAdaptive Total Iterations: {}",
        start_time.elapsed().unwrap(),
        iteration
    );
    Ok((x_path_flows, history))
}

pub fn par_adaptive_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    let mut x_path_flows: HashMap<Vec<NodeIndex>, f64> = HashMap::new();
    let mut f_edge_flows: HashMap<EdgeIndex, f64> = HashMap::new();
    let mut w_edge_costs: HashMap<EdgeIndex, f64> = HashMap::new();
    let mut h_edge_accumulators: HashMap<EdgeIndex, f64> = HashMap::new();

    for edge_ref in graph.edge_references() {
        f_edge_flows.insert(edge_ref.id(), 0.0);
        w_edge_costs.insert(edge_ref.id(), 1.0);
        h_edge_accumulators.insert(edge_ref.id(), 0.0);
    }

    let m = graph.edge_count();
    let threshold = (m as f64).ln() / (epsilon * epsilon);

    let mut history: Vec<IterationInfo> = Vec::new();
    let mut iteration = 0;
    let start_time = std::time::SystemTime::now();

    'outer: loop {
        for edge_ref in graph.edge_references() {
            let capacity_u_ij = *edge_ref.weight();
            if capacity_u_ij <= 1e-9 {
                continue;
            }
            let flow_f_ij = f_edge_flows.get(&edge_ref.id()).unwrap_or(&0.0);
            if flow_f_ij / capacity_u_ij >= threshold {
                break 'outer;
            }
        }

        let best_result = commodities
            .par_iter()
            .map(|&(source_node, target_node)| {
                if source_node == target_node {
                    return None;
                }

                let path_search_result = astar(
                    graph,
                    source_node,
                    |finish_node| finish_node == target_node,
                    |edge_ref| {
                        let capacity_u_e = *edge_ref.weight();
                        if capacity_u_e <= 1e-9 {
                            return f64::INFINITY;
                        }
                        w_edge_costs.get(&edge_ref.id()).unwrap_or(&1.0) / capacity_u_e
                    },
                    |_| 0.0,
                );
                path_search_result
            })
            .reduce(
                || None,
                |a, b| {
                    if a.is_none() {
                        return b;
                    }
                    if b.is_none() {
                        return a;
                    }
                    let (cost_a, path_a) = a.unwrap();
                    let (cost_b, path_b) = b.unwrap();
                    if cost_a < cost_b {
                        Some((cost_a, path_a))
                    } else {
                        Some((cost_b, path_b))
                    }
                },
            );
        let (_, p_star_nodes) = best_result.unwrap();

        let mut p_star_edges_details: Vec<(EdgeIndex, f64)> =
            Vec::with_capacity(p_star_nodes.len().saturating_sub(1));
        let mut u_bottleneck_capacity = f64::INFINITY;

        for i in 0..(p_star_nodes.len() - 1) {
            let node_u_idx = p_star_nodes[i];
            let node_v_idx = p_star_nodes[i + 1];

            let edge_id = graph.find_edge(node_u_idx, node_v_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge from path {:?} -> {:?} not found in graph",
                    node_u_idx,
                    node_v_idx
                )
            })?;

            let edge_capacity_u_ij = *graph.edge_weight(edge_id).unwrap();
            p_star_edges_details.push((edge_id, edge_capacity_u_ij));

            if edge_capacity_u_ij < u_bottleneck_capacity {
                u_bottleneck_capacity = edge_capacity_u_ij;
            }
        }

        *x_path_flows.entry(p_star_nodes.clone()).or_insert(0.0) += u_bottleneck_capacity;

        for &(edge_id, capacity_u_e) in &p_star_edges_details {
            *f_edge_flows.get_mut(&edge_id).unwrap() += u_bottleneck_capacity;

            if capacity_u_e <= 1e-9 {
                continue;
            }

            let h_e_current = u_bottleneck_capacity / capacity_u_e;

            let h_accumulator_past = h_edge_accumulators.get(&edge_id).unwrap_or(&0.0);
            let eta_scalar = epsilon / (h_accumulator_past + ADAGRAD_FUDGE_FACTOR).sqrt();

            let cost_w_e = w_edge_costs.get_mut(&edge_id).unwrap();
            *cost_w_e *= 1.0 + (eta_scalar * h_e_current);

            *h_edge_accumulators.get_mut(&edge_id).unwrap() += h_e_current * h_e_current;
        }

        let c_normalization_factor = get_normalization_factor(graph, &f_edge_flows);
        let current_flow_sum: f64 = get_flow_sum(&x_path_flows, c_normalization_factor);
        let elapsed_time = start_time.elapsed().unwrap();

        if should_log(iteration) {
            history.push(IterationInfo {
                iteration,
                current_flow_sum,
                elapsed_time,
            });
        }

        if let Some(target_flow_val) = target_flow {
            if is_close_to_true(current_flow_sum, target_flow_val, epsilon) {
                break;
            }
        }

        if iteration % 10000 == 0 {
            println!(
                "Adaptive Iteration {}: Current flow scaled sum: {} \t Current elapsed: {:?}",
                iteration, current_flow_sum, elapsed_time,
            );
        }
        iteration += 1;
    }

    normalize_flows(
        &mut x_path_flows,
        get_normalization_factor(graph, &f_edge_flows),
    );
    println!(
        "Adaptive Total time: {:?}\nAdaptive Total Iterations: {}",
        start_time.elapsed().unwrap(),
        iteration
    );
    Ok((x_path_flows, history))
}
