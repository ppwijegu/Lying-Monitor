__author__ = 'pivithuru'
import networkx as nx
import csv


def read_network_data_gexf(path):
    return nx.read_gexf(path)



def noordin_assign_colors_com(G_1):
    for node in G_1:
        if G_1.node[node]["color"]=="blue":
            G_1.node[node]["color"]="Blue"
        else:
            G_1.node[node]["color"]="Red"
    return G_1

def noordin_assign_hierarchy(G_1):
    node_hierarchy={}
    with open("data//Role_score.csv") as f:
        roles = csv.DictReader(f)

        for element in roles:
            if element["Role score"]!='':
                node_hierarchy[element['Name']]=int(element["Role score"])

    print len(node_hierarchy.keys())
    nx.set_node_attributes(G_1,'centrality',node_hierarchy)

    return G_1

def get_noordin_network(data_file_gexf):

    G_1=nx.Graph(read_network_data_gexf(data_file_gexf))
    G_2=noordin_assign_colors_com(G_1)
    G_3=noordin_assign_hierarchy(G_2)


    return G_3

