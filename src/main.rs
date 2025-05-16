use anyhow::Result;
use benchmark_rust::parsers::sndlib_parser::{load_graph_and_traffic, resolve_sndlib_base_path};
use benchmark_rust::solvers::garg_konneman::{
    adaptive_garg_konemann_mcf, fleischer_fptas_mcf, garg_konemann_mcf,
    par_adaptive_garg_konemann_mcf, par_garg_konemann_mcf,
};
use benchmark_rust::utils::commodities_from_traffic_matrix;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use polars::prelude::*;
use serde_json;

#[derive(Debug, Clone, Copy)]
enum Algorithm {
    GargKonemann,
    ParGargKonemann,
    FleischerFPTAS,
    AdaptiveGargKonemann,
    ParAdaptiveGargKonnemann,
}

impl Algorithm {
    fn as_str(&self) -> &'static str {
        match self {
            Algorithm::GargKonemann => "GargKonemann",
            Algorithm::ParGargKonemann => "ParGargKonemann",
            Algorithm::FleischerFPTAS => "FleischerFPTAS",
            Algorithm::AdaptiveGargKonemann => "AdaptiveGargKonemann",
            Algorithm::ParAdaptiveGargKonnemann => "ParAdaptiveGargKonneman",
        }
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    dataset: String,
    algorithm: String,
    epsilon: f64,
    time_sec: f64,
    iterations: usize,
    max_congestion: f64,
    flow_sum: f64,
    iteration_history_json: String,
    error: String,
}

fn run_single_benchmark(
    dataset_name: &str,
    algorithm_enum: Algorithm,
    epsilon: f64,
    base_path: &PathBuf,
) -> BenchmarkResult {
    let start_time = Instant::now();

    let mut path_candidate = base_path
        .join(dataset_name)
        .join(format!("{}.xml", dataset_name));
    if !path_candidate.exists() {
        path_candidate = base_path.join(format!("{}.xml", dataset_name));
    }

    let default_error_history = "[]".to_string();

    if !path_candidate.exists() {
        return BenchmarkResult {
            dataset: dataset_name.to_string(),
            algorithm: algorithm_enum.as_str().to_string(),
            epsilon,
            time_sec: start_time.elapsed().as_secs_f64(),
            iterations: 0,
            max_congestion: f64::NAN,
            flow_sum: f64::NAN,
            iteration_history_json: default_error_history.clone(),
            error: format!("Dataset file not found for {}", dataset_name),
        };
    }
    let file_path = path_candidate;

    let graph_load_result = load_graph_and_traffic(file_path);
    let (graph, traffic_mat) = match graph_load_result {
        Ok(data) => data,
        Err(e) => {
            return BenchmarkResult {
                dataset: dataset_name.to_string(),
                algorithm: algorithm_enum.as_str().to_string(),
                epsilon,
                time_sec: start_time.elapsed().as_secs_f64(),
                iterations: 0,
                max_congestion: f64::NAN,
                flow_sum: f64::NAN,
                iteration_history_json: default_error_history.clone(),
                error: format!("Failed to load graph/traffic: {}", e),
            };
        }
    };

    let commodities = commodities_from_traffic_matrix(graph.node_count(), &traffic_mat);

    if commodities.is_empty() && graph.node_count() > 0 {
        return BenchmarkResult {
            dataset: dataset_name.to_string(),
            algorithm: algorithm_enum.as_str().to_string(),
            epsilon,
            time_sec: start_time.elapsed().as_secs_f64(),
            iterations: 0,
            max_congestion: 0.0,
            flow_sum: 0.0,
            iteration_history_json: default_error_history.clone(),
            error: "No commodities found".to_string(),
        };
    }

    let solver_call_time = Instant::now();
    let solve_result = match algorithm_enum {
        Algorithm::GargKonemann => garg_konemann_mcf(&graph, &commodities, epsilon),
        Algorithm::ParGargKonemann => par_garg_konemann_mcf(&graph, &commodities, epsilon),
        Algorithm::FleischerFPTAS => fleischer_fptas_mcf(&graph, &commodities, epsilon),
        Algorithm::AdaptiveGargKonemann => {
            adaptive_garg_konemann_mcf(&graph, &commodities, epsilon)
        }
        Algorithm::ParAdaptiveGargKonnemann => {
            par_adaptive_garg_konemann_mcf(&graph, &commodities, epsilon)
        }
    };
    let exec_time_solver_only = solver_call_time.elapsed().as_secs_f64();

    match solve_result {
        Ok((solution, history)) => {
            let flow_sum_val: f64 = solution.values().sum();
            let total_iterations = history.len();
            let history_json = serde_json::to_string(&history).unwrap_or_else(|e| {
                eprintln!("Failed to serialize iteration history: {}", e);
                default_error_history.clone()
            });
            let mut max_congestion = 0.0f64;
            if !solution.is_empty() {
                for (edge_nodes, flow_value) in &solution {
                    if let Some(edge_idx) = graph.find_edge(edge_nodes[0], edge_nodes[1]) {
                        if let Some(capacity) = graph.edge_weight(edge_idx) {
                            if *capacity > 1e-9 {
                                let congestion = flow_value / capacity;
                                if congestion > max_congestion {
                                    max_congestion = congestion;
                                }
                            } else if *flow_value > 1e-9 {
                                max_congestion = f64::INFINITY;
                                break;
                            }
                        }
                    }
                }
            }

            BenchmarkResult {
                dataset: dataset_name.to_string(),
                algorithm: algorithm_enum.as_str().to_string(),
                epsilon,
                time_sec: exec_time_solver_only,
                iterations: total_iterations,
                max_congestion,
                flow_sum: flow_sum_val,
                iteration_history_json: history_json,
                error: String::new(),
            }
        }
        Err(e) => BenchmarkResult {
            dataset: dataset_name.to_string(),
            algorithm: algorithm_enum.as_str().to_string(),
            epsilon,
            time_sec: exec_time_solver_only,
            iterations: 0,
            max_congestion: f64::NAN,
            flow_sum: f64::NAN,
            iteration_history_json: default_error_history.clone(),
            error: format!("Solver error: {}", e),
        },
    }
}

fn main() -> Result<()> {
    let datasets_to_run = vec!["brain"];
    let algorithms_to_run = vec![
        Algorithm::ParAdaptiveGargKonnemann,
        Algorithm::AdaptiveGargKonemann,
        Algorithm::FleischerFPTAS,
        Algorithm::GargKonemann,
        Algorithm::ParGargKonemann,
    ];
    let epsilons_to_test = vec![0.1, 0.01];

    let mut all_results: Vec<BenchmarkResult> = Vec::new();
    let sndlib_base_path = resolve_sndlib_base_path();

    println!("SNDlib base path: {:?}", sndlib_base_path);
    println!("Starting benchmarks...\n");

    for dataset_name in &datasets_to_run {
        for &algorithm_enum in &algorithms_to_run {
            for &epsilon_val in &epsilons_to_test {
                println!(
                    "Running: Dataset={}, Algorithm={}, Epsilon={}",
                    dataset_name,
                    algorithm_enum.as_str(),
                    epsilon_val
                );
                let result = run_single_benchmark(
                    dataset_name,
                    algorithm_enum,
                    epsilon_val,
                    &sndlib_base_path,
                );
                println!(
                    "  Finished: Time={:.3}s, Iters={}, MaxCong={:.4}, FlowSum={:.4}, HistLen={}, Error='{}'",
                    result.time_sec,
                    result.iterations,
                    result.max_congestion,
                    result.flow_sum,
                    result.iteration_history_json.chars().count(),
                    result.error
                );
                all_results.push(result);
            }
        }
        println!("---");
    }

    println!("\nAll benchmarks completed. Writing results to CSV...");

    if all_results.is_empty() {
        println!("No results to write.");
        return Ok(());
    }

    let datasets_col: Vec<String> = all_results.iter().map(|r| r.dataset.clone()).collect();
    let algorithms_col: Vec<String> = all_results.iter().map(|r| r.algorithm.clone()).collect();
    let epsilons_col: Vec<f64> = all_results.iter().map(|r| r.epsilon).collect();
    let times_col: Vec<f64> = all_results.iter().map(|r| r.time_sec).collect();
    let iterations_col: Vec<u64> = all_results.iter().map(|r| r.iterations as u64).collect();
    let max_congestions_col: Vec<f64> = all_results.iter().map(|r| r.max_congestion).collect();
    let flow_sums_col: Vec<f64> = all_results.iter().map(|r| r.flow_sum).collect();
    let iteration_history_json_col: Vec<String> = all_results
        .iter()
        .map(|r| r.iteration_history_json.clone())
        .collect();
    let errors_col: Vec<String> = all_results.iter().map(|r| r.error.clone()).collect();

    let mut df_results = df!(
        "dataset" => datasets_col,
        "algorithm" => algorithms_col,
        "epsilon" => epsilons_col,
        "time_sec" => times_col,
        "iterations" => iterations_col,
        "max_congestion" => max_congestions_col,
        "flow_sum" => flow_sums_col,
        "iteration_history_json" => iteration_history_json_col,
        "error" => errors_col,
    )?;

    let mut output_file = File::create("benchmark_results.csv")?;
    CsvWriter::new(&mut output_file)
        .include_header(true)
        .with_separator(b',')
        .finish(&mut df_results)?;

    println!("Results successfully written to benchmark_results.csv");

    Ok(())
}
