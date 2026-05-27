use anyhow::Result;
use benchmark_rust::parsers::gml_parser::{
    load_graph_and_traffic as load_graph_and_traffic_gml, resolve_gml_base_path,
};
use benchmark_rust::parsers::sndlib_parser::{
    load_graph_and_traffic as load_graph_and_traffic_xml, resolve_sndlib_base_path,
};
use benchmark_rust::solvers::garg_konneman::{
    adaptive_adam_garg_konemann_mcf, adaptive_garg_konemann_mcf,
    adaptive_rmsprop_garg_konemann_mcf, fleischer_fptas_mcf, garg_konemann_mcf,
    par_adaptive_adam_garg_konemann_mcf, par_adaptive_garg_konemann_mcf,
    par_adaptive_rmsprop_garg_konemann_mcf, par_garg_konemann_mcf,
    adaptive_hedge_garg_konemann_mcf, par_adaptive_hedge_garg_konemann_mcf,
};
use benchmark_rust::utils::commodities_from_traffic_matrix;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
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
    AdaptiveRmsPropGargKonemann,
    ParAdaptiveRmsPropGargKonemann,
    AdaptiveAdamGargKonemann,
    ParAdaptiveAdamGargKonemann,
    AdaptiveHedgeGargKonemann,
    ParAdaptiveHedgeGargKonemann,
}

impl Algorithm {
    fn as_str(&self) -> &'static str {
        match self {
            Algorithm::GargKonemann => "GargKonemann",
            Algorithm::ParGargKonemann => "ParGargKonemann",
            Algorithm::FleischerFPTAS => "FleischerFPTAS",
            Algorithm::AdaptiveGargKonemann => "AdaptiveGargKonemann",
            Algorithm::ParAdaptiveGargKonnemann => "ParAdaptiveGargKonneman",
            Algorithm::AdaptiveRmsPropGargKonemann => "AdaptiveRmsPropGargKonemann",
            Algorithm::ParAdaptiveRmsPropGargKonemann => "ParAdaptiveRmsPropGargKonemann",
            Algorithm::AdaptiveAdamGargKonemann => "AdaptiveAdamGargKonemann",
            Algorithm::ParAdaptiveAdamGargKonemann => "ParAdaptiveAdamGargKonemann",
            Algorithm::AdaptiveHedgeGargKonemann => "AdaptiveHedgeGargKonemann",
            Algorithm::ParAdaptiveHedgeGargKonemann => "ParAdaptiveHedgeGargKonemann",
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

fn find_datasets(dir: &Path, datasets: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let folder_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if folder_name.starts_with('.') {
                    continue;
                }

                let has_gml = path.join("graph.gml").exists() && path.join("traffic_mat.json").exists();
                let has_xml = path.join(format!("{}.xml", folder_name)).exists();

                if has_gml || has_xml {
                    datasets.push(path);
                } else {
                    find_datasets(&path, datasets);
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let mut true_flow_map: HashMap<String, f64> = HashMap::new();
    let true_flows_csv_path = "cvxpy_flows.csv";
    match File::open(true_flows_csv_path) {
        Ok(file) => {
            let reader = CsvReader::new(file);
            match reader.finish() {
                Ok(df_true_flows) => {
                    let folder_series = df_true_flows.column("folder")?;
                    let flow_sum_series = df_true_flows.column("flow_sum")?;

                    let folder_col = folder_series.str()?;
                    let flow_sum_col = flow_sum_series.f64()?;

                    for (opt_folder, opt_flow_sum) in
                        folder_col.into_iter().zip(flow_sum_col.into_iter())
                    {
                        if let (Some(folder), Some(flow_sum)) = (opt_folder, opt_flow_sum) {
                            true_flow_map.insert(folder.to_string(), flow_sum);
                        }
                    }
                    println!(
                        "Successfully loaded true flow sums from {}",
                        true_flows_csv_path
                    );
                }
                Err(_) => {
                    eprintln!(
                        "Could not read or parse DataFrame from {}",
                        true_flows_csv_path
                    );
                }
            }
        }
        Err(e) => {
            eprintln!(
                "Warning: Could not open {}: {}. Proceeding without true flow data.",
                true_flows_csv_path, e
            );
        }
    }

    let gml_base_path = resolve_gml_base_path();
    let sndlib_base_path = resolve_sndlib_base_path();

    let mut datasets_to_run: Vec<PathBuf> = Vec::new();
    find_datasets(&gml_base_path, &mut datasets_to_run);
    datasets_to_run.sort();

    if datasets_to_run.is_empty() {
        eprintln!(
            "Warning: Could not find any datasets in GML base path directory: {:?}",
            gml_base_path
        );
    }

    let algorithms_to_run = vec![
        Algorithm::GargKonemann,
        Algorithm::ParGargKonemann,
        //Algorithm::FleischerFPTAS,
        Algorithm::AdaptiveGargKonemann,
        Algorithm::ParAdaptiveGargKonnemann,
        Algorithm::AdaptiveRmsPropGargKonemann,
        Algorithm::ParAdaptiveRmsPropGargKonemann,
        Algorithm::AdaptiveAdamGargKonemann,
        Algorithm::ParAdaptiveAdamGargKonemann,
        Algorithm::AdaptiveHedgeGargKonemann,
        Algorithm::ParAdaptiveHedgeGargKonemann,
    ];
    let epsilons_to_test = vec![0.01];

    let mut all_results: Vec<BenchmarkResult> = Vec::new();
    let default_error_history = "[]".to_string();

    println!("GML base path: {:?}", gml_base_path);
    println!("SNDlib base path: {:?}", sndlib_base_path);
    println!("Found {} datasets to run.", datasets_to_run.len());
    println!("Starting benchmarks...\n");

    for dataset_path in &datasets_to_run {
        let relative_path = dataset_path.strip_prefix(&gml_base_path).unwrap_or(dataset_path);
        
        let dataset_name = relative_path
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join("/");

        let folder_name = dataset_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("Unknown");

        println!("Processing Dataset: {} (Path: {:?})", dataset_name, dataset_path);

        let load_start_time = Instant::now();

        let graph_load_result = if dataset_path.join("graph.gml").exists()
            && dataset_path.join("traffic_mat.json").exists()
        {
            load_graph_and_traffic_gml(dataset_path)
        } else {
            let mut xml_path = dataset_path.join(format!("{}.xml", dataset_name));
            if !xml_path.exists() {
                xml_path = sndlib_base_path
                    .join(&relative_path)
                    .join(format!("{}.xml", folder_name));
            }

            if !xml_path.exists() {
                xml_path = sndlib_base_path.join(format!("{}.xml", folder_name));
            }

            if !xml_path.exists() {
                let err_msg = format!("Dataset files not found for {}", folder_name);
                eprintln!("  {}", err_msg);

                for &algorithm_enum in &algorithms_to_run {
                    for &epsilon_val in &epsilons_to_test {
                        all_results.push(BenchmarkResult {
                            dataset: dataset_name.clone(),
                            algorithm: algorithm_enum.as_str().to_string(),
                            epsilon: epsilon_val,
                            time_sec: load_start_time.elapsed().as_secs_f64(),
                            iterations: 0,
                            max_congestion: f64::NAN,
                            flow_sum: f64::NAN,
                            iteration_history_json: default_error_history.clone(),
                            error: err_msg.clone(),
                        });
                    }
                }
                continue;
            }
            load_graph_and_traffic_xml(xml_path)
        };

        let (graph, traffic_mat) = match graph_load_result {
            Ok(data) => data,
            Err(e) => {
                let err_msg = format!("Failed to load graph/traffic: {}", e);
                eprintln!("  {}", err_msg);

                for &algorithm_enum in &algorithms_to_run {
                    for &epsilon_val in &epsilons_to_test {
                        all_results.push(BenchmarkResult {
                            dataset: dataset_name.clone(),
                            algorithm: algorithm_enum.as_str().to_string(),
                            epsilon: epsilon_val,
                            time_sec: load_start_time.elapsed().as_secs_f64(),
                            iterations: 0,
                            max_congestion: f64::NAN,
                            flow_sum: f64::NAN,
                            iteration_history_json: default_error_history.clone(),
                            error: err_msg.clone(),
                        });
                    }
                }
                continue;
            }
        };

        let commodities =
            commodities_from_traffic_matrix(graph.node_count(), &traffic_mat, Some(0.1));

        if commodities.is_empty() && graph.node_count() > 0 {
            let err_msg = "No commodities found".to_string();
            eprintln!("  {}", err_msg);

            for &algorithm_enum in &algorithms_to_run {
                for &epsilon_val in &epsilons_to_test {
                    all_results.push(BenchmarkResult {
                        dataset: dataset_name.clone(),
                        algorithm: algorithm_enum.as_str().to_string(),
                        epsilon: epsilon_val,
                        time_sec: load_start_time.elapsed().as_secs_f64(),
                        iterations: 0,
                        max_congestion: 0.0,
                        flow_sum: 0.0,
                        iteration_history_json: default_error_history.clone(),
                        error: err_msg.clone(),
                    });
                }
            }
            continue;
        }

        let true_flow_for_run = true_flow_map.get(&dataset_name).copied();
        if true_flow_for_run.is_some() {
            println!("  Using true_flow: {:?}", true_flow_for_run.unwrap());
        } else {
            println!("  No true_flow found for dataset: {}", dataset_name);
        }

        for &algorithm_enum in &algorithms_to_run {
            for &epsilon_val in &epsilons_to_test {
                println!(
                    "  Running: Algorithm={}, Epsilon={}",
                    algorithm_enum.as_str(),
                    epsilon_val
                );

                let solver_call_time = Instant::now();
                let solve_result = match algorithm_enum {
                    Algorithm::GargKonemann => {
                        garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::ParGargKonemann => {
                        par_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::FleischerFPTAS => {
                        fleischer_fptas_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::AdaptiveGargKonemann => {
                        adaptive_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::ParAdaptiveGargKonnemann => {
                        par_adaptive_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::AdaptiveRmsPropGargKonemann => {
                        adaptive_rmsprop_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::ParAdaptiveRmsPropGargKonemann => {
                        par_adaptive_rmsprop_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::AdaptiveAdamGargKonemann => {
                        adaptive_adam_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::ParAdaptiveAdamGargKonemann => {
                        par_adaptive_adam_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::AdaptiveHedgeGargKonemann => {
                        adaptive_hedge_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                    Algorithm::ParAdaptiveHedgeGargKonemann => {
                        par_adaptive_hedge_garg_konemann_mcf(&graph, &commodities, epsilon_val, true_flow_for_run)
                    }
                };
                let exec_time_solver_only = solver_call_time.elapsed().as_secs_f64();

                let result = match solve_result {
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
                            dataset: dataset_name.clone(),
                            algorithm: algorithm_enum.as_str().to_string(),
                            epsilon: epsilon_val,
                            time_sec: exec_time_solver_only,
                            iterations: total_iterations,
                            max_congestion,
                            flow_sum: flow_sum_val,
                            iteration_history_json: history_json,
                            error: String::new(),
                        }
                    }
                    Err(e) => BenchmarkResult {
                        dataset: dataset_name.clone(),
                        algorithm: algorithm_enum.as_str().to_string(),
                        epsilon: epsilon_val,
                        time_sec: exec_time_solver_only,
                        iterations: 0,
                        max_congestion: f64::NAN,
                        flow_sum: f64::NAN,
                        iteration_history_json: default_error_history.clone(),
                        error: format!("Solver error: {}", e),
                    },
                };

                println!(
                    "    Finished: Time={:.3}s, Iters={}, MaxCong={:.4}, FlowSum={:.4}, HistLen={}, Error='{}'",
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