__author__ = 'pivithuru'


import networkx as nx



def get_all_triangles(G):

    all_triangles={}

    cycls_3 = [c for c in nx.cycle_basis(G) if len(c)==3]

    for cycle in cycls_3:

        for i in range(len(cycle)):

            if cycle[i] not in all_triangles:

                all_triangles[cycle[i]] = []

            all_triangles[cycle[i]].append((cycle[(i+1)%3],cycle[(i+2)%3]))


    return all_triangles



