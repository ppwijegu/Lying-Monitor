__author__ = 'pivithuru'

import networkx as nx
import csv
import pickle
import operator

class NetworkData():

    def __init__(self):

        self.G = None


    def read_network_data(self, path):

        pass
    def assign_colors(self):

        pass

    # assign hierarchy to nodes here
    def assign_hierarchy(self):

        pass

    def print_network_data(self):

        print "No.nodes: "+ str(len(self.G.nodes()))
        print "No.edges: "+ str(len(self.G.edges()))
        print "No.red node: "+ str(len([n for n in self.G.nodes() if self.G.node[n]["color"] == "Red" ]))


    def get_network_data(self, path):

        self.read_network_data(path)
        self.assign_colors()
        self.assign_hierarchy()

        self.print_network_data()
        return self.G

class NoordinData(NetworkData):

    def __init__(self):

        NetworkData.__init__(self)

    def read_network_data(self,path):

        self.G = nx.read_gexf(path)



    def assign_colors(self):
        for node in self.G:
            if self.G.node[node]["color"]=="blue":
                self.G.node[node]["color"]="Blue"
            else:
                self.G.node[node]["color"]="Red"


    def assign_hierarchy(self):
        node_hierarchy={}
        with open("data//Noordin//Role_score.csv") as f:
            roles = csv.DictReader(f)

            for element in roles:
                if element["Role score"]!='':
                    node_hierarchy[element['Name']]=int(element["Role score"])

        nx.set_node_attributes(self.G,'centrality',node_hierarchy)


class PokeCData(NetworkData):

    def __init__(self):

        NetworkData.__init__(self)


    def assign_hierarchy(self):

        for u in self.G.nodes():

                centrality=self.G.degree(u)

                self.G.node[u]['centrality']=centrality

    def read_network_data(self, path):

        G_all = pickle.load(open(path))
        self.G = sorted(nx.connected_component_subgraphs(G_all), key = len, reverse=True)[0]



class Facebook100Data(NetworkData):

    def __init__(self, attribute, order):

        NetworkData.__init__(self)

        self.attribute = attribute

        self.order = order


    def assign_hierarchy(self):

        for node in self.G.nodes():
            self.G.node[node]["centrality"] = nx.degree(self.G,node)

    def read_network_data(self, path):

        G_current = pickle.load(open(path))
        self.G = sorted(nx.connected_component_subgraphs(G_current), key = len, reverse=True)[0]

    def assign_colors(self):

        node_attribute={}

        for n in self.G.nodes():

            att = self.G.node[n][self.attribute]

            if node_attribute.has_key(att):
                node_attribute[att]+=1
            else:
                node_attribute[att]=1

        att_sorted=sorted(node_attribute.iteritems(),key=operator.itemgetter(1),reverse=True)

        if att_sorted[self.order][0]!=0:
            att_val=att_sorted[self.order][0]

        else:

            att_val=att_sorted[self.order][0]


        for n in self.G.nodes():

            if self.G.node[n][self.attribute]==att_val:

                self.G.node[n]["color"]="Red"
            else :
                self.G.node[n]["color"]="Blue"


