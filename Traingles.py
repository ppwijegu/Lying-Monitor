__author__ = 'pivithuru'


import networkx as nx



def get_all_triangles(G):

    all_triangles={}

    for node in G.nodes():
        all_triangles[node]=[]

        neighbors=G.neighbors(node)



        for i in range(len(neighbors)):
            for j in range(i+1,len(neighbors)):

                    if neighbors[j] in G.neighbors(neighbors[i]):
                        all_triangles[node].append((neighbors[i],neighbors[j]))



    return all_triangles




