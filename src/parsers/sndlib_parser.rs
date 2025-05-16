use anyhow::Result;
use ndarray::Array2;
use petgraph::graph::DiGraph;
use petgraph::Graph;
use quick_xml::de::from_reader;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;

use super::xml_models::Network;

type GraphType = DiGraph<String, f64>;

pub fn resolve_sndlib_base_path() -> PathBuf {
    let mut base_path = PathBuf::from("./sndlib_xml/");
    if !base_path.exists() {
        base_path = PathBuf::from(".");
        eprintln!("Warning: Default SNDlib path not found. Using current directory.");
    }
    base_path
}

fn normalize_traffic_matrix(traffic_mat: &mut Array2<f64>) {
    let max_val = traffic_mat.iter().copied().fold(0.0, f64::max);

    if max_val > 0.0 {
        *traffic_mat /= max_val;
    }
}

pub fn load_graph_and_traffic<P: AsRef<Path>>(
    file_path: P,
) -> Result<(GraphType, Array2<f64>), anyhow::Error> {
    let file = File::open(file_path)?;
    let xml: Network = from_reader(BufReader::new(file))?;
    let network = xml.network_structure;
    let mut graph: GraphType = Graph::new();
    let mut id_to_index = HashMap::new();

    // Nodes

    for node in network.nodes.node_list {
        let node_index = graph.add_node(node.id.clone());
        id_to_index.insert(node.id, node_index);
    }

    let node_count = graph.node_count();

    // Edges

    for link in network.links.link_list {
        let source = id_to_index.get(&link.source).expect("Source not found");
        let target = id_to_index
            .get(&link.target)
            .expect("Target not found in nodes");
        let weight = if let Some(module) = link.pre_installed_module {
            module.capacity
        } else if !link.additional_modules.add_module_list.is_empty() {
            link.additional_modules.add_module_list[0].capacity
        } else {
            1.0
        };
        graph.add_edge(*source, *target, weight);
        graph.add_edge(*target, *source, weight);
    }

    // Demands

    let demands = xml.demands.demand_list;

    let mut traffic_mat = Array2::zeros((node_count, node_count));

    for demand in demands {
        let source = id_to_index.get(&demand.source).expect("Source not found");
        let target = id_to_index
            .get(&demand.target)
            .expect("Target not found in nodes");
        traffic_mat[[source.index(), target.index()]] = demand.demand_value;
    }

    normalize_traffic_matrix(&mut traffic_mat);

    Ok((graph, traffic_mat))
}
