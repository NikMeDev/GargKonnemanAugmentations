use anyhow::{anyhow, Result};
use gxhash::{HashMap, HashMapExt};
use petgraph::graph::{DiGraph, EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, EdgeIndexable, NodeIndexable};
use rayon::prelude::*;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub type GraphType = DiGraph<String, f64>;

use std::cell::RefCell;

thread_local! {
    static DIJKSTRA_ENV: RefCell<Option<DijkstraEnv>> = RefCell::new(None);
}

#[derive(Clone)]
struct State {
    cost: f64,
    position: NodeIndex,
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost) == Ordering::Equal && self.position == other.position
    }
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(other.cost.total_cmp(&self.cost))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.total_cmp(&self.cost)
    }
}

struct DijkstraEnv {
    distances: Vec<f64>,
    came_from: Vec<Option<(NodeIndex, EdgeIndex)>>,
    heap: BinaryHeap<State>,
    generation: u32,
    visited_gen: Vec<u32>,
}

impl DijkstraEnv {
    fn new(node_count: usize) -> Self {
        Self {
            distances: vec![f64::INFINITY; node_count],
            came_from: vec![None; node_count],
            heap: BinaryHeap::new(),
            generation: 0,
            visited_gen: vec![0; node_count],
        }
    }

    fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.visited_gen.fill(0);
            self.generation = 1;
        }
        self.heap.clear();
    }

    #[inline(always)]
    fn get_dist(&self, node: NodeIndex) -> f64 {
        let idx = node.index();
        if self.visited_gen[idx] == self.generation {
            self.distances[idx]
        } else {
            f64::INFINITY
        }
    }

    #[inline(always)]
    fn set_dist(&mut self, node: NodeIndex, dist: f64, prev: Option<(NodeIndex, EdgeIndex)>) {
        let idx = node.index();
        self.visited_gen[idx] = self.generation;
        self.distances[idx] = dist;
        self.came_from[idx] = prev;
    }

    fn find_path<F>(
        &mut self,
        graph: &GraphType,
        source: NodeIndex,
        target: NodeIndex,
        cost_fn: F,
    ) -> Option<(f64, Vec<NodeIndex>, Vec<EdgeIndex>)>
    where
        F: Fn(EdgeIndex, f64) -> f64,
    {
        self.clear();
        self.set_dist(source, 0.0, None);
        self.heap.push(State { cost: 0.0, position: source });

        let mut found = false;

        while let Some(State { cost, position }) = self.heap.pop() {
            if position == target {
                found = true;
                break;
            }

            if cost > self.get_dist(position) {
                continue;
            }

            for edge_ref in graph.edges(position) {
                let capacity = *edge_ref.weight();
                if capacity <= 1e-9 {
                    continue;
                }

                let edge_cost = cost_fn(edge_ref.id(), capacity);
                let next_cost = cost + edge_cost;
                let next_node = edge_ref.target();

                if next_cost < self.get_dist(next_node) {
                    self.set_dist(next_node, next_cost, Some((position, edge_ref.id())));
                    self.heap.push(State { cost: next_cost, position: next_node });
                }
            }
        }

        if found {
            let mut path_nodes = Vec::new();
            let mut path_edges = Vec::new();
            let mut current = target;
            while let Some((prev_node, edge_id)) = self.came_from[current.index()] {
                if self.visited_gen[current.index()] != self.generation {
                    break;
                }
                path_nodes.push(current);
                path_edges.push(edge_id);
                current = prev_node;
                if current == source {
                    break;
                }
            }
            path_nodes.push(source);
            path_nodes.reverse();
            path_edges.reverse();
            Some((self.get_dist(target), path_nodes, path_edges))
        } else {
            None
        }
    }
}

fn get_normalization_factor(graph: &GraphType, f_edge_flows: &[f64]) -> f64 {
    let mut c_normalization_factor = 0.0f64;
    for (idx, edge) in graph.raw_edges().iter().enumerate() {
        let capacity_u_ij = edge.weight;
        if capacity_u_ij <= 1e-9 {
            continue;
        }

        let flow_f_ij = f_edge_flows[idx];
        let ratio = flow_f_ij / capacity_u_ij;
        if ratio > c_normalization_factor {
            c_normalization_factor = ratio;
        }
    }
    c_normalization_factor
}

fn get_flow_sum(x_path_flows: &HashMap<Vec<NodeIndex>, f64>, c_normalization_factor: f64) -> f64 {
    let sum: f64 = x_path_flows.values().sum();
    sum / c_normalization_factor
}

fn normalize_flows(x_path_flows: &mut HashMap<Vec<NodeIndex>, f64>, c_normalization_factor: f64) {
    if c_normalization_factor <= 1e-9 {
        for flow_val in x_path_flows.values_mut() {
            *flow_val = 0.0;
        }
    } else {
        let inv = 1.0 / c_normalization_factor;
        for flow_val in x_path_flows.values_mut() {
            *flow_val *= inv;
        }
    }
}

fn is_close_to_true(_current_flow: f64, _target_flow: f64, _epsilon: f64) -> bool {
    false // target_flow / current_flow <= 1.0 + epsilon
}

fn should_log(iteration: usize) -> bool {
    let s = (iteration as f64).sqrt() as usize;
    s * s == iteration
}

#[derive(Debug, Clone, Serialize)]
pub struct IterationInfo {
    pub iteration: usize,
    pub current_flow_sum: f64,
    pub elapsed_time: std::time::Duration,
}

const ADAPTIVE_OPTIMIZER_EPSILON: f64 = 1e-10;
const RMSPROP_BETA: f64 = 0.99;
const ADAM_BETA1: f64 = 0.9;
const ADAM_BETA2: f64 = 0.999;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OptimizerMethod {
    Standard,
    AdaGrad,
    RmsProp,
    Adam,
}

fn get_cost_multiplier(
    method: OptimizerMethod,
    edge_id: EdgeIndex,
    grad: f64,
    gk_epsilon: f64,
    step_index: usize,
    squared_grad_accumulators: &mut [f64],
    first_moments: &mut [f64],
    last_updated_step: &mut [usize],
) -> f64 {
    if method == OptimizerMethod::Standard {
        return 1.0 + gk_epsilon * grad;
    }

    let idx = edge_id.index();
    let k = step_index.saturating_sub(last_updated_step[idx]).saturating_sub(1) as i32;

    match method {
        OptimizerMethod::Standard => unreachable!(),
        OptimizerMethod::AdaGrad => {
            squared_grad_accumulators[idx] += grad * grad;
            last_updated_step[idx] = step_index;
            
            let denom = squared_grad_accumulators[idx].sqrt() + ADAPTIVE_OPTIMIZER_EPSILON;
            1.0 + (gk_epsilon / denom) * grad
        }
        OptimizerMethod::RmsProp => {
            if k > 0 {
                squared_grad_accumulators[idx] *= RMSPROP_BETA.powi(k);
            }
            squared_grad_accumulators[idx] = RMSPROP_BETA * squared_grad_accumulators[idx] + (1.0 - RMSPROP_BETA) * (grad * grad);
            last_updated_step[idx] = step_index;

            let denom = squared_grad_accumulators[idx].sqrt() + ADAPTIVE_OPTIMIZER_EPSILON;
            1.0 + (gk_epsilon / denom) * grad
        }
        OptimizerMethod::Adam => {
            if k > 0 {
                first_moments[idx] *= ADAM_BETA1.powi(k);
                squared_grad_accumulators[idx] *= ADAM_BETA2.powi(k);
            }
            first_moments[idx] = ADAM_BETA1 * first_moments[idx] + (1.0 - ADAM_BETA1) * grad;
            squared_grad_accumulators[idx] = ADAM_BETA2 * squared_grad_accumulators[idx] + (1.0 - ADAM_BETA2) * (grad * grad);
            last_updated_step[idx] = step_index;

            let t = step_index as i32;
            let m_hat = first_moments[idx] / (1.0 - ADAM_BETA1.powi(t));
            let v_hat = squared_grad_accumulators[idx] / (1.0 - ADAM_BETA2.powi(t));
            let denom = v_hat.sqrt() + ADAPTIVE_OPTIMIZER_EPSILON;
            1.0 + (gk_epsilon / denom) * m_hat
        }
    }
}

fn select_best_path(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    w_edge_costs: &[f64],
    parallel: bool,
    env: &mut DijkstraEnv,
) -> Option<(Vec<NodeIndex>, Vec<EdgeIndex>)> {
    if parallel {
        let node_bound = graph.node_bound();
        commodities
            .par_iter()
            .cloned()
            .filter_map(|(source_node, target_node)| {
                if source_node == target_node {
                    return None;
                }
                
                DIJKSTRA_ENV.with(|cell| {
                    let mut env_opt = cell.borrow_mut();
                    
                    if env_opt.is_none() || env_opt.as_ref().unwrap().distances.len() < node_bound {
                        *env_opt = Some(DijkstraEnv::new(node_bound));
                    }
                    
                    let local_env = env_opt.as_mut().unwrap();
                    local_env.find_path(graph, source_node, target_node, |edge_id, capacity| {
                        w_edge_costs[edge_id.index()] / capacity
                    })
                })
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, nodes, edges)| (nodes, edges))
    } else {
        let mut current_best: Option<(Vec<NodeIndex>, Vec<EdgeIndex>)> = None;
        let mut min_path_overall_cost = f64::INFINITY;

        for &(source_node, target_node) in commodities {
            if source_node == target_node {
                continue;
            }

            if let Some((path_cost, nodes, edges)) = env.find_path(graph, source_node, target_node, |edge_id, capacity| {
                w_edge_costs[edge_id.index()] / capacity
            }) {
                if path_cost < min_path_overall_cost && path_cost.is_finite() {
                    min_path_overall_cost = path_cost;
                    current_best = Some((nodes, edges));
                }
            }
        }

        current_best
    }
}

fn garg_konemann_impl(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
    method: OptimizerMethod,
    parallel: bool,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    let mut x_path_flows: HashMap<Vec<NodeIndex>, f64> = HashMap::new();
    
    let edge_bound = graph.edge_bound();
    let mut f_edge_flows = vec![0.0; edge_bound];
    let mut w_edge_costs = vec![1.0; edge_bound];

    let is_adaptive = method != OptimizerMethod::Standard;
    let mut squared_grad_accumulators = if is_adaptive { vec![0.0; edge_bound] } else { vec![] };
    let mut first_moments = if method == OptimizerMethod::Adam { vec![0.0; edge_bound] } else { vec![] };
    let mut last_updated_step = if is_adaptive { vec![0_usize; edge_bound] } else { vec![] };

    let mut history: Vec<IterationInfo> = Vec::new();
    let mut iteration = 0;
    let start_time = std::time::SystemTime::now();
    let mut max_congestion = 0.0;

    let node_bound = graph.node_bound();
    let mut env = DijkstraEnv::new(node_bound);

    'outer: loop {
        let (p_star_nodes, p_star_edges) =
            select_best_path(graph, commodities, &w_edge_costs, parallel, &mut env).unwrap_or_default();
        
        if p_star_nodes.len() < 2 {
            break 'outer;
        }

        let mut p_star_edges_details: Vec<(EdgeIndex, f64)> =
            Vec::with_capacity(p_star_edges.len());
        let mut u_bottleneck_capacity = f64::INFINITY;

        for &edge_id in &p_star_edges {
            let edge_capacity_u_ij = *graph.edge_weight(edge_id).unwrap();
            p_star_edges_details.push((edge_id, edge_capacity_u_ij));

            if edge_capacity_u_ij < u_bottleneck_capacity {
                u_bottleneck_capacity = edge_capacity_u_ij;
            }
        }

        *x_path_flows.entry(p_star_nodes).or_insert(0.0) += u_bottleneck_capacity;

        let mut needs_scaling = false;

        for &(edge_id, capacity_u_e) in &p_star_edges_details {
            let idx = edge_id.index();
            f_edge_flows[idx] += u_bottleneck_capacity;

            let congestion = f_edge_flows[idx] / capacity_u_e;
            if congestion > max_congestion {
                max_congestion = congestion;
            }

            if capacity_u_e <= 1e-9 {
                continue;
            }

            let grad = u_bottleneck_capacity / capacity_u_e;
            let multiplier = get_cost_multiplier(
                method,
                edge_id,
                grad,
                epsilon,
                iteration + 1,
                &mut squared_grad_accumulators,
                &mut first_moments,
                &mut last_updated_step,
            );

            let cost_w_e = &mut w_edge_costs[idx];
            *cost_w_e *= multiplier;
            
            if *cost_w_e > 1e250 {
                needs_scaling = true;
            }
        }

        if needs_scaling {
            for w in w_edge_costs.iter_mut() {
                *w *= 1e-200;
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

        if iteration % 100000 == 0 {
            println!(
                "Iteration {}: Current flow scaled sum: {} \t Current elapsed: {:?}",
                iteration, current_flow_sum, elapsed_time,
            );
        }
        iteration += 1;
        
        if elapsed_time.as_secs() > 10 {
            println!("Timeout reached, stopping the algorithm.");
            break;
        }
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

pub fn garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::Standard, false)
}

pub fn par_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::Standard, true)
}

pub fn adaptive_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::AdaGrad, false)
}

pub fn par_adaptive_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::AdaGrad, true)
}

pub fn adaptive_rmsprop_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::RmsProp, false)
}

pub fn par_adaptive_rmsprop_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::RmsProp, true)
}

pub fn adaptive_adam_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::Adam, false)
}

pub fn par_adaptive_adam_garg_konemann_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    garg_konemann_impl(graph, commodities, epsilon, target_flow, OptimizerMethod::Adam, true)
}

pub fn fleischer_fptas_mcf(
    graph: &GraphType,
    commodities: &[(NodeIndex, NodeIndex)],
    epsilon: f64,
    target_flow: Option<f64>,
) -> Result<(HashMap<Vec<NodeIndex>, f64>, Vec<IterationInfo>)> {
    let delta = (1.0f64 + epsilon)
        * ((1.0f64 + epsilon) * (graph.node_count() as f64)).powf(-1.0f64 / epsilon);

    let edge_bound = graph.edge_bound();
    let mut l_edge_lengths = vec![delta; edge_bound];
    let mut f_edge_flows = vec![0.0; edge_bound];

    let mut x_path_flows: HashMap<Vec<NodeIndex>, f64> = HashMap::new();
    let mut history: Vec<IterationInfo> = Vec::new();
    let start_time = std::time::SystemTime::now();

    let log_val = ((1.0f64 + epsilon) / delta).ln();
    let log_base = (1.0f64 + epsilon).ln();

    let r_max = (log_val / log_base).ceil() as usize;

    let mut iteration = 0;
    
    let node_bound = graph.node_bound();
    let mut env = DijkstraEnv::new(node_bound);

    for r_iter_val in 1..=r_max {
        for &(source_node, target_node) in commodities {
            if source_node == target_node {
                continue;
            }

            let path_search_initial = env.find_path(
                graph,
                source_node,
                target_node,
                |edge_id, _| l_edge_lengths[edge_id.index()]
            );

            iteration += 1;

            if let Some((mut current_path_cost, mut current_path_nodes, mut current_path_edges)) = path_search_initial {
                if current_path_nodes.len() < 2 {
                    continue;
                }

                let threshold_r = (1.0f64).min(delta * (1.0 + epsilon).powi(r_iter_val as i32));

                while current_path_cost < threshold_r {
                    let mut u_bottleneck = f64::INFINITY;
                    let mut path_edges_details: Vec<(EdgeIndex, f64)> =
                        Vec::with_capacity(current_path_edges.len());

                    for &edge_id in &current_path_edges {
                        let capacity_u_e = *graph.edge_weight(edge_id).ok_or_else(|| {
                            anyhow!("Capacity for edge {:?} not found", edge_id)
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

                    for &(edge_id, capacity_u_e) in &path_edges_details {
                        let idx = edge_id.index();
                        f_edge_flows[idx] += u_bottleneck;
                        let l_e = &mut l_edge_lengths[idx];
                        *l_e *= 1.0 + (epsilon * u_bottleneck / capacity_u_e);
                    }

                    let path_search_next = env.find_path(
                        graph,
                        source_node,
                        target_node,
                        |edge_id, _| l_edge_lengths[edge_id.index()]
                    );

                    iteration += 1;
                    let elapsed_time = start_time.elapsed().unwrap();
                    if elapsed_time.as_secs() > 60 * 30 {
                        println!("Timeout reached, stopping the algorithm.");
                        break;
                    }

                    if let Some((next_cost, next_nodes, next_edges)) = path_search_next {
                        if next_nodes.len() < 2 {
                            current_path_cost = f64::INFINITY;
                        } else {
                            current_path_cost = next_cost;
                            current_path_nodes = next_nodes;
                            current_path_edges = next_edges;
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