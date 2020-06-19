import osmnx as ox
import networkx as nx

from math import *
import geopy.distance
import numpy as np
import collections
import pandas as pd
import dgl
from mxnet import nd, gpu
from custom.city import RFNCity, DGLCity
import torch



def latlng2dist(v1, v2):
    R = 6373.0
    lat1, lon1 = v1
    lat2, lon2 = v2
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def add_edge_position_info(G: nx.MultiDiGraph):
    for edge in list(G.edges(keys=True, data=True)):
        u, v, k, info = edge
        u_pos_x = G.nodes[u]['x']
        u_pos_y = G.nodes[u]['y']
        v_pos_x = G.nodes[v]['x']
        v_pos_y = G.nodes[v]['y']
        info['u_coord'] = (u_pos_x, u_pos_y)
        info['v_coord'] = (v_pos_x, v_pos_y)
        if 'geometry' in info:
            geom_coords = list(info['geometry'].coords)
            info['u_next_coord'] = geom_coords[1]
            info['v_prev_coord'] = geom_coords[-2]
        else:
            info['u_next_coord'] = (v_pos_x, v_pos_y)
            info['v_prev_coord'] = (u_pos_x, u_pos_y)


def make_dual_graph(G: nx.MultiDiGraph):
    D = nx.DiGraph()
    D.add_nodes_from(
        edge
        for edge in G.edges
    )

    for e in G.edges(keys=True):
        u, v, k = e
        for o in G.out_edges(v, keys=True):
            if e != o:
                D.add_edge(e, o)

    # copy edge attrib
    for e in D.nodes(data=True):
        edge, info = e
        u, v, k = edge
        for a, b in G.edges[u, v, k].items():
            info[a] = b

    return D


def add_between_edge_attrib_custom(D, N=4):
    X_B = np.zeros(shape=(D.number_of_edges(), N))
    theta_histogram = [0 for _ in range(N)]
    index = 0
    for between_edge in D.edges(data=True):
        edge1, edge2, info = between_edge
        # print(G.edges[edge1]['v_coord'])
        assert D.nodes[edge1]['v_coord'][0] == D.nodes[edge2]['u_coord'][0]
        assert D.nodes[edge1]['v_coord'][1] == D.nodes[edge2]['u_coord'][1]

        p1 = D.nodes[edge1]['v_prev_coord']
        p2 = D.nodes[edge1]['v_coord']
        p3 = D.nodes[edge2]['u_next_coord']

        c = latlng2dist(p1, p2)
        a = latlng2dist(p2, p3)
        b = latlng2dist(p3, p1)

        AB = (p2[0] - p1[0], p2[1] - p1[1])
        BC = (p3[0] - p2[0], p3[1] - p2[1])
        cr = AB[0] * BC[1] - AB[1] * BC[0]

        if b != 0 and a != 0 and c != 0:
            cosB = (a * a + c * c - b * b) / (2 * a * c)
            cosB = min(max(cosB, -1), 1)
            theta = degrees(acos(cosB))
            theta = 180.0 - theta
            theta *= -1 if cr < 0 else 1
        else:
            theta = 180

        U = 360.0 / N

        theta = (theta + 0.5 * U + 360) % 360
        theta_i = int((theta / 360.0) * N)
        info["turning_angle"] = theta_i
        X_B[index, theta_i] = 1.0
        theta_histogram[theta_i] += 1
        index += 1

    print(theta_histogram)
    return X_B


def add_between_edge_attrib(D, N=4):
    X_B = np.zeros(shape=(D.number_of_edges(), N))
    theta_histogram = [0 for _ in range(N)]
    index = 0
    for between_edge in D.edges(data=True):
        edge1, edge2, info = between_edge
        b1 = D.nodes[edge1]["bearing"]
        b2 = D.nodes[edge2]["bearing"]
        theta = b2 - b1
        if np.isnan(b1):
            theta = 0
        elif np.isnan(b2):
            theta = 0

        U = 360.0 / N
        theta = (theta + 0.5 * U + 360) % 360
        theta_i = int((theta / 360.0) * N)
        info["turning_angle"] = theta_i
        X_B[index, theta_i] = 1.0
        theta_histogram[theta_i] += 1
        index += 1

    print(theta_histogram)
    return X_B

def get_edge_attrib(G: nx.MultiDiGraph):
    check_lists = ['residential', 'unclassified', 'tertiary',
                   'tertiary_link', 'trunk_link', 'primary',
                   'secondary', 'living_street', 'primary_link',
                   'secondary_link', 'motorway_link', 'trunk',
                   'motorway', 'road', 'bus_guideway'
                   ]

    X_E = np.zeros(shape=(G.number_of_edges(), len(check_lists) + 1), dtype=np.float32)
    for edge_index, edge in enumerate(G.edges(keys=True, data=True)):
        u, v, k, info = edge
        highway_type = info["highway"]
        i = 0
        for valid_highway_type in check_lists:
            if type(highway_type) is list:
                if valid_highway_type in highway_type:
                    X_E[edge_index, i] = 1
            else:
                if valid_highway_type == highway_type:
                    X_E[edge_index, i] = 1
            i += 1
        X_E[edge_index, i] = info['length'] / 100.0

    return X_E


def get_edge_attrib_old(G: nx.MultiDiGraph):
    nodes, edges = ox.graph_to_gdfs(G, nodes=True)
    temp_data = edges['highway'].apply(lambda x: x.split(':')).apply(collections.Counter)
    edge_attrib = pd.DataFrame.from_records(temp_data).fillna(value=0.0)
    edge_attrib['length'] = edges['length'] / 100.0
    print(list(edge_attrib.columns))
    return edge_attrib.values


def get_edge_attrib_for_GCN(G: nx.MultiDiGraph, edge_attrib, vertex_attrib):
    N_E = edge_attrib.shape[1]
    N_V = vertex_attrib.shape[1]
    X_E = np.zeros(shape=(G.number_of_edges(), N_E + 2*N_V))
    for i, e in enumerate(G.edges(keys=True, data=True)):
        u, v, k, info = e
        X_E[i, 0:N_E] = edge_attrib[i]
        X_E[i, N_E:N_E + N_V] = vertex_attrib[u]
        X_E[i, N_E+N_V:N_E + 2*N_V] = vertex_attrib[v]
    return X_E

def get_vertex_attrib(G: nx.MultiDiGraph):
    X_V = np.zeros(shape=(G.number_of_nodes(), 2), dtype=np.float32)
    index = 0
    for node in G.nodes(data=True):
        u, info = node
        X_V[index][0] = G.in_degree(u)
        X_V[index][1] = G.out_degree(u)
        index += 1
    return X_V / 6

def get_edge_target_y(G: nx.MultiDiGraph, target_value="speed_kph"):
    nodes, edges = ox.graph_to_gdfs(G, nodes=True)
    return edges[target_value].values / 60

def custom_dgl(G: nx.MultiDiGraph):
    g = dgl.DGLGraph()
    g.add_nodes(G.number_of_nodes())
    u_to_new = {}
    i = 0
    for n in G.nodes(data=True):
        u, info = n
        u_to_new[u] = i
        i += 1
        #g.nodes[i].data = info

    for e in G.edges(data=True):
        u, v, info = e
        g.add_edge(u_to_new[u], u_to_new[v], data=info)

    return g


def load_city_graph(city_name):
    G = ox.load_graphml('data/%s_drive_network_original.graphml' % city_name)
    G = nx.convert_node_labels_to_integers(G, ordering='default')
    G = ox.add_edge_bearings(G)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    # for e in G.edges(data=True):
    #     u, v, info = e
    #     if type(info['highway']) == list:
    #         info['highway'] = ":".join(info['highway'])
    return G


def generate_required_city_graph(city_name, G):
    add_edge_position_info(G)
    D = make_dual_graph(G)

    X_B_np = add_between_edge_attrib(D)
    X_E_np = get_edge_attrib(G)
    X_V_np = get_vertex_attrib(G)
    y_np = get_edge_target_y(G, target_value="speed_kph")

    print("Primal V,E: (%d, %d), Dual V,E: (%d, %d)"
          % (G.number_of_nodes(), G.number_of_edges(), D.number_of_nodes(), D.number_of_edges()))

    for i, n in enumerate(D.nodes(data=True)):
        u, info = n
        info["edge_info"] = u

    primal_graph = G
    dual_graph = D

    X_V = nd.array(X_V_np, ctx=gpu())
    X_E = nd.array(X_E_np, ctx=gpu())
    X_B = nd.array(X_B_np, ctx=gpu())
    y = nd.array(y_np, ctx=gpu())

    rfncity = RFNCity(city_name, primal_graph, dual_graph)
    rfncity.set_features(X_V, X_E, X_B, y)

    dual_graph_int_indexed = nx.convert_node_labels_to_integers(dual_graph, ordering='default')
    dual_graph_dgl = dgl.DGLGraph()
    dual_graph_dgl.from_networkx(dual_graph_int_indexed, node_attrs=['length', 'edge_info'])

    X_E_gcn_np = get_edge_attrib_for_GCN(G, X_E_np, X_V_np)
    X_E_gcn_torch = torch.from_numpy(X_E_gcn_np).float()
    y_torch = torch.from_numpy(y_np).float()

    dglcity = DGLCity(city_name, dual_graph_dgl)
    dglcity.set_features(X_E_gcn_torch, y_torch)

    return rfncity, dglcity