__author__ = 'pivithuru'

import numpy as np
scores_dict={}
def select_random_monitor(terrorist_graph,nodes_lie):

     if nodes_lie:

        threshold=max_no_monitors_get_true_color(terrorist_graph)

        if threshold<3:
            threshold=3

        choice=[node for node in terrorist_graph if terrorist_graph.node[node]["MonitorNumber"]<threshold+1 and terrorist_graph.node[node]["color"]!="Red" ]
        if len(choice)!=0:
            return np.random.choice(choice)
        else:
            return None

     else:
        unmonitord=[node for node in terrorist_graph if terrorist_graph.node[node]["IsMonitor"]==False ]
        if len(unmonitord)!=0:
            return np.random.choice(unmonitord)
        else:
            return None


def select_next_monitor_most_red_neighbors(terrorist_graph,selected,nodes_lie=False):
    global scores_dict
    max_red_score=0
    nodes_with_max_score=[]

    if nodes_lie:

        threshold=max_no_monitors_get_true_color(terrorist_graph)

        if threshold<3:
            threshold=3


        for node in terrorist_graph.neighbors(selected):
            if not terrorist_graph.node[node]["color"]=="Red" and terrorist_graph.node[node]["MonitorNumber"]<threshold+1:
                red_score=0


                for neighbor in terrorist_graph.neighbors(node):
                    if terrorist_graph.node[neighbor]["color"]=="Red":
                        red_score+=1


                scores_dict[node]=red_score



    else:
        for node in terrorist_graph.neighbors(selected):
            if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
                red_score=0


                for neighbor in terrorist_graph.neighbors(node):
                    if terrorist_graph.node[neighbor]["color"]=="Red":
                        red_score+=1


                scores_dict[node]=red_score

    for key in scores_dict:
            red_score=scores_dict[key]
            if red_score>=max_red_score:
                if red_score > max_red_score:
                    nodes_with_max_score=[]
                max_red_score=red_score
                nodes_with_max_score.append(key)

    try:
        next_node=np.random.choice(nodes_with_max_score)
        del scores_dict[next_node]

    except:
        print len([n for n in terrorist_graph.nodes() if terrorist_graph.node[n]["IsMonitor"]==True])
        next_node=None




    return next_node


def select_next_monitor_most_red_says_red(terrorist_graph,selected_node,nodes_lie=False):
    global scores_dict
    max_red_score=0
    nodes_with_max_score=[]
    if nodes_lie:

        threshold=max_no_monitors_get_true_color(terrorist_graph)

        if threshold<3:
            threshold=3


        for node in terrorist_graph.neighbors(selected_node):
            if not terrorist_graph.node[node]["color"]=="Red" and terrorist_graph.node[node]["MonitorNumber"]<threshold+1:
                red_score=0
                says_red=terrorist_graph.node[node]["SaysRed"]

                for neighbor in says_red:
                    if terrorist_graph.node[neighbor]["color"]=="Red":
                        red_score+=1
                scores_dict[node]=red_score

    else:

        for node in terrorist_graph.neighbors(selected_node):
            if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
                red_score=0
                says_red=terrorist_graph.node[node]["SaysRed"]

                for neighbor in says_red:
                    if terrorist_graph.node[neighbor]["color"]=="Red":
                        red_score+=1
                scores_dict[node]=red_score

    for key in scores_dict:
            red_score=scores_dict[key]
            if red_score>=max_red_score:
                if red_score>max_red_score:
                    nodes_with_max_score=[]

                max_red_score=red_score
                nodes_with_max_score.append(key)


    try:
        next_node=np.random.choice(nodes_with_max_score)
        del scores_dict[next_node]
    except:
        print len([n for n in terrorist_graph.nodes() if terrorist_graph.node[n]["IsMonitor"]==True])
        next_node=None


    return next_node
## Calculate the color scores for red score method
def get_color_score(terrorist_graph,node):
    red_score=0
    blue_score=0
    for n in terrorist_graph.node[node]['SaysRed']:
        red_score+=terrorist_graph.node[n]["EstHonRed"]

    for n in terrorist_graph.node[node]['SaysBlue']:
        blue_score+=terrorist_graph.node[n]["EstHonBlue"]



    conf_red=1.0*red_score/(red_score+blue_score)


    return {"redscore":red_score,"confred":conf_red,"confblue":1-conf_red}

def select_next_monitor_highest_estimated_red_score(terrorist_graph,selected_node,nodes_lie=False):
    global scores_dict

    max_red_score=0

    nodes_with_max_score=[]
    if nodes_lie:

        threshold=max_no_monitors_get_true_color(terrorist_graph)

        if threshold<3:
            threshold=3


        for node in terrorist_graph.neighbors(selected_node):
            if not terrorist_graph.node[node]["color"]=="Red" and terrorist_graph.node[node]["MonitorNumber"]<threshold+1:
                results=get_color_score(terrorist_graph,node)
                red_score=results["redscore"]
                scores_dict[node]=red_score
    else:
        for node in terrorist_graph.neighbors(selected_node):
            if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
                results=get_color_score(terrorist_graph,node)
                red_score=results["redscore"]
                scores_dict[node]=red_score

    for key in scores_dict:
            red_score=scores_dict[key]
            if red_score>=max_red_score:
                if red_score > max_red_score:
                    nodes_with_max_score=[]
                max_red_score=red_score
                nodes_with_max_score.append(key)


    try:
        next_node=np.random.choice(nodes_with_max_score)
        del scores_dict[next_node]
    except:
        print len([n for n in terrorist_graph.nodes() if terrorist_graph.node[n]["IsMonitor"]==True])
        next_node=None


    return next_node


def max_no_monitors_get_true_color(terrorist_graph):
    red_nodes=[n for n in terrorist_graph.nodes() if terrorist_graph.node[n]['color']=="Red"]
    red_monitors=[]

    for node in red_nodes:

        no_monitors=terrorist_graph.node[node]["MonitorNumber"]

        red_monitors.append(no_monitors)


    return max(red_monitors)


def initialize():
    global scores_dict
    scores_dict={}

