use anyhow::{Context, Result, anyhow, bail};
use ndarray::Array2;
use petgraph::Graph;
use petgraph::graph::DiGraph;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::utils::normalize_traffic_matrix;

type GraphType = DiGraph<String, f64>;

#[derive(Debug, Default)]
struct GmlNode {
    id: Option<i64>,
    label: Option<String>,
}

#[derive(Debug, Default)]
struct GmlEdge {
    source: Option<i64>,
    target: Option<i64>,
    bandwidth: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TrafficMatrixFile {
    num_nodes: usize,
    #[serde(default)]
    mat: Vec<TrafficMatrixRow>,
}

#[derive(Debug, Deserialize)]
struct TrafficMatrixRow {
    source: usize,
    #[serde(default)]
    targets: Vec<usize>,
    #[serde(default)]
    demands: Vec<f64>,
}

pub fn resolve_gml_base_path() -> PathBuf {
    let path = PathBuf::from("./networks_gml");
    if !path.exists() {
        eprintln!("Warning: GML base path not found, falling back to current directory.");
        PathBuf::from(".")
    } else {
        path
    }
}

pub fn load_graph_and_traffic<P: AsRef<Path>>(dataset_dir: P) -> Result<(GraphType, Array2<f64>)> {
    let dataset_dir = dataset_dir.as_ref();
    let graph_path = dataset_dir.join("graph.gml");
    let traffic_path = dataset_dir.join("traffic_mat.json");

    let graph = load_graph(&graph_path)?;
    let mut traffic = load_traffic_matrix(&traffic_path, graph.node_count())?;
    normalize_traffic_matrix(&mut traffic);

    Ok((graph, traffic))
}

fn load_graph<P: AsRef<Path>>(gml_path: P) -> Result<GraphType> {
    let file = File::open(gml_path.as_ref())
        .with_context(|| format!("Failed to open GML file {}", gml_path.as_ref().display()))?;
    let reader = BufReader::new(file);

    let mut graph: GraphType = Graph::new();
    let mut id_to_index = HashMap::new();

    let mut current_node: Option<GmlNode> = None;
    let mut current_edge: Option<GmlEdge> = None;

    for (line_no, raw_line) in reader.lines().enumerate() {
        let line_no = line_no + 1;
        let line = raw_line.with_context(|| format!("Failed to read line {}", line_no))?;
        let trimmed = line.trim();

        if trimmed == "node [" {
            current_node = Some(GmlNode::default());
            continue;
        }
        if trimmed == "edge [" {
            current_edge = Some(GmlEdge::default());
            continue;
        }

        if trimmed == "]" {
            if let Some(node) = current_node.take() {
                finalize_node(node, &mut graph, &mut id_to_index)?;
                continue;
            }
            if let Some(edge) = current_edge.take() {
                finalize_edge(edge, &mut graph, &id_to_index)?;
                continue;
            }
            continue;
        }

        if let Some(node) = current_node.as_mut() {
            parse_node_line(node, trimmed, line_no)?;
        } else if let Some(edge) = current_edge.as_mut() {
            parse_edge_line(edge, trimmed, line_no)?;
        }
    }

    if current_node.is_some() || current_edge.is_some() {
        bail!(
            "Malformed GML file {}: unclosed node/edge block",
            gml_path.as_ref().display()
        );
    }

    Ok(graph)
}

fn finalize_node(
    node: GmlNode,
    graph: &mut GraphType,
    id_to_index: &mut HashMap<i64, petgraph::prelude::NodeIndex>,
) -> Result<()> {
    let id = node.id.context("Node is missing required `id` field")?;
    let label = node.label.unwrap_or_else(|| id.to_string());

    if id_to_index.contains_key(&id) {
        bail!("Duplicate node id `{}` in GML file", id);
    }

    let idx = graph.add_node(label);
    id_to_index.insert(id, idx);
    Ok(())
}

fn finalize_edge(
    edge: GmlEdge,
    graph: &mut GraphType,
    id_to_index: &HashMap<i64, petgraph::prelude::NodeIndex>,
) -> Result<()> {
    let source_id = edge
        .source
        .context("Edge is missing required `source` field")?;
    let target_id = edge
        .target
        .context("Edge is missing required `target` field")?;
    let bandwidth = edge.bandwidth.unwrap_or(1.0);

    let source_idx = *id_to_index
        .get(&source_id)
        .ok_or_else(|| anyhow!("Edge source node id `{}` not found", source_id))?;
    let target_idx = *id_to_index
        .get(&target_id)
        .ok_or_else(|| anyhow!("Edge target node id `{}` not found", target_id))?;

    graph.add_edge(source_idx, target_idx, bandwidth);
    Ok(())
}

fn parse_node_line(node: &mut GmlNode, line: &str, line_no: usize) -> Result<()> {
    if let Some(rest) = line.strip_prefix("id ") {
        node.id = Some(
            rest.parse::<i64>()
                .with_context(|| format!("Invalid node id at line {}", line_no))?,
        );
    } else if let Some(rest) = line.strip_prefix("label ") {
        node.label = Some(strip_optional_quotes(rest));
    }
    Ok(())
}

fn parse_edge_line(edge: &mut GmlEdge, line: &str, line_no: usize) -> Result<()> {
    if let Some(rest) = line.strip_prefix("source ") {
        edge.source = Some(
            rest.parse::<i64>()
                .with_context(|| format!("Invalid edge source at line {}", line_no))?,
        );
    } else if let Some(rest) = line.strip_prefix("target ") {
        edge.target = Some(
            rest.parse::<i64>()
                .with_context(|| format!("Invalid edge target at line {}", line_no))?,
        );
    } else if let Some(rest) = line.strip_prefix("bandwidth ") {
        edge.bandwidth = Some(
            rest.parse::<f64>()
                .with_context(|| format!("Invalid edge bandwidth at line {}", line_no))?,
        );
    }
    Ok(())
}

fn strip_optional_quotes(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

fn load_traffic_matrix<P: AsRef<Path>>(
    traffic_path: P,
    expected_nodes: usize,
) -> Result<Array2<f64>> {
    let file = File::open(traffic_path.as_ref()).with_context(|| {
        format!(
            "Failed to open traffic matrix file {}",
            traffic_path.as_ref().display()
        )
    })?;

    let parsed: TrafficMatrixFile =
        serde_json::from_reader(BufReader::new(file)).with_context(|| {
            format!(
                "Failed to parse traffic matrix JSON {}",
                traffic_path.as_ref().display()
            )
        })?;

    if parsed.num_nodes != expected_nodes {
        bail!(
            "Traffic matrix node count mismatch: graph has {}, traffic has {}",
            expected_nodes,
            parsed.num_nodes
        );
    }

    let mut traffic = Array2::zeros((parsed.num_nodes, parsed.num_nodes));

    for row in parsed.mat {
        if row.targets.len() != row.demands.len() {
            bail!(
                "Traffic row source {} has mismatched targets ({}) and demands ({}) lengths",
                row.source,
                row.targets.len(),
                row.demands.len()
            );
        }
        if row.source >= parsed.num_nodes {
            bail!(
                "Traffic row source {} is out of bounds for {} nodes",
                row.source,
                parsed.num_nodes
            );
        }

        for (target, demand) in row.targets.into_iter().zip(row.demands.into_iter()) {
            if target >= parsed.num_nodes {
                bail!(
                    "Traffic target {} is out of bounds for {} nodes",
                    target,
                    parsed.num_nodes
                );
            }
            traffic[[row.source, target]] = demand;
        }
    }

    Ok(traffic)
}
