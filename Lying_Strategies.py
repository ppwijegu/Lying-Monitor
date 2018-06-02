__author__ = 'pivithuru'

import numpy as np
import networkx as nx

prob_lying_own_color=0

def assign_honesty(G):
    mu,sigma=0.5,0.125
    honesty=np.random.normal(mu,sigma,len(G.nodes()))
    honesty=abs(honesty)

    i=0
    node_honesty={}
    for node in G.nodes():

        node_honesty[node]=honesty[i]
        i+=1
    nx.set_node_attributes(G,'honesty',node_honesty)



#Get neighbor colors based on lying strategy 1
def get_neighbor_colors_setting1(G,node):

    neighbor_colors={}

    for neighbor in G.neighbors(node):

        rand=np.random.random()

        if (G.node[neighbor]['color']=="Blue"):
            prob_lie=(1-G.node[node]['honesty'])


            if prob_lie >rand:

                neighbor_colors[neighbor]="Red"
            else:
                neighbor_colors[neighbor]="Blue"


        elif (G.node[neighbor]['color']=="Red"):

            prob_lie=min((1.0*G.node[neighbor]['centrality']*(1-G.node[node]['honesty']))/G.node[node]['centrality'],1)



            if prob_lie>rand:
                neighbor_colors[neighbor]="Blue"
            else:
                neighbor_colors[neighbor]="Red"




    return neighbor_colors


#Get neighbor colors based on lying strategy 2
def get_neighbor_colors_setting2(G,node):

    neighbor_colors={}
    for neighbor in G.neighbors(node):

        if (G.node[node]['color']=="Blue"):
             neighbor_colors[neighbor]="Blue"



        elif (G.node[node]['color']=="Red"):
            rand=np.random.random()

            if (G.node[neighbor]['color']=="Blue"):
                prob_lie=(1-G.node[node]['honesty'])
                if prob_lie >rand:

                    neighbor_colors[neighbor]="Red"
                else:
                    neighbor_colors[neighbor]="Blue"

            elif (G.node[neighbor]['color']=="Red"):

                prob_lie=min((1.0*G.node[neighbor]['centrality']*(1-G.node[node]['honesty']))/G.node[node]['centrality'],1)

                if prob_lie>rand:
                    neighbor_colors[neighbor]="Blue"
                else:
                    neighbor_colors[neighbor]="Red"
    return neighbor_colors



## Based on the lying strategy we use, this function returns the set of colors all nodes in G  say their neighbors are
def assign_color_I_say(lying_strategy,G):

    all_neighbor_colors={}

    for node in G.nodes():

        colors = lying_strategy(G,node)
        all_neighbor_colors[node] = colors

    return all_neighbor_colors

def get_color_I_am(G,node):

    if G.node[node]['color']=="Blue":
        return "Blue"
    else:

        rand=np.random.random()

        if rand <prob_lying_own_color:
            return "Blue"
        else:
            return "Red"