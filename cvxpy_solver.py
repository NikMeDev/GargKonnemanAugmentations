from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xmltodict

import cvxpy as cp
import time

import os

FLOAT = np.float64

def read_graph_sndlib_xml(filename: Path) -> nx.Graph:
    with open(filename, "r") as file:
        graph_dct = xmltodict.parse(file.read())["network"]["networkStructure"]

    graph = nx.DiGraph()

    for node in graph_dct["nodes"]["node"]:
        graph.add_node(
            node["@id"],
            x=FLOAT(node["coordinates"]["x"]),
            y=FLOAT(node["coordinates"]["y"]),
        )

    for edge in graph_dct["links"]["link"]:
        cost = FLOAT(edge.get("routingCost", 1.0))
        if "preInstalledModule" in edge:
            bandwidth = FLOAT(edge["preInstalledModule"]["capacity"])
        elif "additionalModules" in edge and "addModule" in edge["additionalModules"]:
            module = edge["additionalModules"]["addModule"]
            if isinstance(module, list):
                module = module[0]
            bandwidth = FLOAT(module["capacity"])
        else:
            bandwidth = FLOAT(1.0)
        graph.add_edge(edge["source"], edge["target"], cost=cost, bandwidth=bandwidth)
        graph.add_edge(edge["target"], edge["source"], cost=cost, bandwidth=bandwidth)

    return graph


def read_traffic_mat_sndlib_xml(filename) -> np.ndarray:
    with open(filename, "r") as file:
        xml_dct = xmltodict.parse(file.read())["network"]

    node_label_to_num = {node["@id"]: i for i, node in enumerate(xml_dct["networkStructure"]["nodes"]["node"])}
    traffic_mat = np.zeros((len(node_label_to_num), len(node_label_to_num)), dtype=FLOAT)
    for demand in xml_dct["demands"]["demand"]:
        source = node_label_to_num[demand["source"]]
        target = node_label_to_num[demand["target"]]
        traffic_mat[source, target] = demand["demandValue"]
    return traffic_mat

def scale_graph_bandwidth_and_cost(graph: nx.Graph) -> nx.Graph:
    scaled_graph = graph.copy()
    max_bandwidth = max(nx.get_edge_attributes(graph, "bandwidth").values())
    max_cost = max(nx.get_edge_attributes(graph, "cost").values())
    for edge in graph.edges:
        scaled_graph.edges[edge]["bandwidth"] /= max_bandwidth
        scaled_graph.edges[edge]["cost"] /= max_cost
    return scaled_graph

sndlib_xml = Path("./sndlib_xml").resolve()

folders = os.listdir(sndlib_xml)

def solve_cvxpy_max_multicommodity_flow(graph, commodities, tolerance):
    bandwidth = np.array(list(nx.get_edge_attributes(graph, "bandwidth").values()), dtype=np.float64)
    flow = cp.Variable((len(graph.edges), len(commodities)))

    final_flows = []
    nodes_list = list(graph.nodes)
    for i, (source, target) in enumerate(commodities):
        target_node = nodes_list[target]
        for edge in list(graph.edges()):
            if edge[1] == target_node:
                edge_idx = list(graph.edges()).index(edge)
                final_flows.append(flow[edge_idx, i])

    flow_sum = cp.sum(final_flows)

    prob = cp.Problem(cp.Maximize(flow_sum),
                         [cp.sum(flow, axis=1) <= bandwidth, flow >= 0])
    
    start = time.time()
    prob.solve(tol_gap_rel=tolerance)
    end = time.time()
    duration = end - start
    print(f"Solved in {duration:.2f} seconds")
    return flow.value, duration

def traffic_mat_to_commodities(traffic_mat):
    commodities = []
    for i in range(traffic_mat.shape[0]):
        for j in range(traffic_mat.shape[1]):
            if traffic_mat[i, j] > 0:
                commodities.append((i, j))
    return commodities

results = []

for folder in folders[:100]:
    print(f"Processing graph: {folder}")
    folder_path = sndlib_xml / folder

    file = folder_path / f"{folder}.xml"

    graph = read_graph_sndlib_xml(file)
    traffic_mat = read_traffic_mat_sndlib_xml(file)
    commodities = traffic_mat_to_commodities(traffic_mat)

    flow, duration = solve_cvxpy_max_multicommodity_flow(graph, commodities, 1e-5)

    final_flows = []
    nodes_list = list(graph.nodes)
    for i, (source, target) in enumerate(commodities):
        target_node = nodes_list[target]
        for edge in list(graph.edges()):
            if edge[1] == target_node:
                edge_idx = list(graph.edges()).index(edge)
                final_flows.append(flow[edge_idx, i])

    flow_sum = sum(final_flows)

    results.append({
        "folder": folder,
        "flow_sum": flow_sum,
        "duration": duration,
    })

df_results = pd.DataFrame(results)
df_results.to_csv("cvxpy_flows.csv", index=False)
print(df_results)