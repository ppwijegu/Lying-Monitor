__author__ = 'pivithuru'

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import copy
from scipy.stats import mode

max_centrality=0
min_honesty=0

def read_network_data_gexf(path):
    return nx.read_gexf(path)

def read_network_data_gml(path):
    return nx.read_gml(path)

def draw_network(G):
    # get unique groups
    for node in G.nodes():
        if str(node).split("_")[0]=="R":
            G.node[node]['color']="Red"
        else:
             G.node[node]['color']="Blue"



    #colors=[G[n]['color'] for n in nodes]
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_edges(G, pos, alpha=1)
    # nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors,
    #                             with_labels=True, node_size=100)
    #
    # plt.axis('off')
    # plt.show()


def assign_colors(G):
    node_colors={}
    with open("data//Noordin_Red.csv") as f:
      colors  =csv.DictReader(f)



      for element in colors:

          if element["Meeting  3"]=='1':
            node_colors[element['Name']]="Red"
    for node in G:
        if not node_colors.has_key(node):

            node_colors[node]="Blue"

    nx.set_node_attributes(G,'color',node_colors)







def assign_honesty(G):
    mu,sigma=0.5,0.125
    honesty=np.random.normal(mu,sigma,len(G.nodes()))
    honesty=abs(honesty)



    #honesty=[(n- min(honesty))/(max(honesty)-min(honesty)) for n in honesty]  # range transform the honesty values between 0 and 1

    #print normalized_honesty
    i=0
    node_honesty={}
    for node in G.nodes():

        node_honesty[node]=honesty[i]
        i+=1
    nx.set_node_attributes(G,'honesty',node_honesty)


    global min_honesty
    min_honesty=min(honesty)
def assign_centrality(G, centralityFunction):

    centrality_values=centralityFunction(G)

    nx.set_node_attributes(G,'centrality',centrality_values)

    global max_centrality

    max_centrality=max(centrality_values.values())

def get_neighbor_colors_setting1(G,node):

    neighbor_colors={}

    for neighbor in G.neighbors(node):

        rand=np.random.random()

        if (G.node[neighbor]['color']=="Blue"):
            prob_lie=1.0*min_honesty/G.node[node]['honesty']


            if prob_lie >rand:

                neighbor_colors[neighbor]="Red"
            else:
                neighbor_colors[neighbor]="Blue"



        elif (G.node[neighbor]['color']=="Red"):
            prob_lie=G.node[neighbor]['centrality']*min_honesty/(G.node[node]['honesty']*max_centrality)


            if prob_lie>rand:
                neighbor_colors[neighbor]="Blue"
            else:
                neighbor_colors[neighbor]="Red"




    return neighbor_colors

def get_neighbor_colors_setting2(G,node):

    neighbor_colors={}
    for neighbor in G.neighbors(node):

        if (G.node[node]['color']=="Blue"):
             neighbor_colors[neighbor]="Blue"



        elif (G.node[node]['color']=="Red"):
            rand=np.random.random()

            if (G.node[neighbor]['color']=="Blue"):
                prob_lie=1.0*min_honesty/G.node[node]['honesty']
                if prob_lie >rand:

                    neighbor_colors[neighbor]="Red"
                else:
                    neighbor_colors[neighbor]="Blue"

            elif (G.node[neighbor]['color']=="Red"):
                prob_lie=G.node[neighbor]['centrality']*min_honesty/(G.node[node]['honesty']*max_centrality)

                if prob_lie>rand:
                    neighbor_colors[neighbor]="Blue"
                else:
                    neighbor_colors[neighbor]="Red"
    return neighbor_colors


def get_neighbor_colors_setting3(G,node):

    neighbor_colors={}
    for neighbor in G.neighbors(node):
        rand=np.random.random()
        if (G.node[node]['color']=="Blue"):
             if (G.node[neighbor]['color']=="Blue"):
                neighbor_colors[neighbor]="Blue"
             else:
                prob_lie=G.node[neighbor]['centrality']*min_honesty/(G.node[node]['honesty']*max_centrality)
                if prob_lie>rand:
                    neighbor_colors[neighbor]="Blue"
                else:
                    neighbor_colors[neighbor]="Red"



        elif (G.node[node]['color']=="Red"):


            if (G.node[neighbor]['color']=="Blue"):
                prob_lie=1.0*min_honesty/G.node[node]['honesty']
                if prob_lie >rand:

                    neighbor_colors[neighbor]="Red"
                else:
                    neighbor_colors[neighbor]="Blue"

            elif (G.node[neighbor]['color']=="Red"):
                prob_lie=G.node[neighbor]['centrality']*min_honesty/(G.node[node]['honesty']*max_centrality)

                if prob_lie>rand:
                    neighbor_colors[neighbor]="Blue"
                else:
                    neighbor_colors[neighbor]="Red"
    return neighbor_colors



def get_color_score_1(terrorist_graph,node):
    red_score=1
    blue_score=1

    for n in terrorist_graph.node[node]['SaysRed']:

        red_score*=terrorist_graph.node[n]["EstHonRed"]

        blue_score*=(1-terrorist_graph.node[n]["EstHonRed"])



    for n in terrorist_graph.node[node]['SaysBlue']:

        blue_score*=terrorist_graph.node[n]["EstHonBlue"]

        red_score*=(1-terrorist_graph.node[n]["EstHonBlue"])


    conf_red=red_score
    conf_blue=blue_score


    return {"redscore":red_score,"bluescore":blue_score,"confred":conf_red,"confblue":conf_blue}

def get_color_score_2(terrorist_graph,node):
    red_score=0
    blue_score=0
    for n in terrorist_graph.node[node]['SaysRed']:
        red_score+=terrorist_graph.node[n]["EstHonRed"]

    for n in terrorist_graph.node[node]['SaysBlue']:
        blue_score+=terrorist_graph.node[n]["EstHonBlue"]

    conf_red=0.0001
    conf_blue=0.0001
    if len(terrorist_graph.node[node]['SaysRed'])>0:
        conf_red=1.0*red_score/len(terrorist_graph.node[node]['SaysRed'])
    if len(terrorist_graph.node[node]['SaysBlue'])>0:
        conf_blue=1.0*blue_score/len(terrorist_graph.node[node]['SaysBlue'])

    return {"redscore":red_score,"bluescore":blue_score,"confred":conf_red,"confblue":conf_blue}

def get_color_score_3(terrorist_graph,node):
    red_score=0
    blue_score=0
    max_red_hon=0
    max_blue_hon=0
    for n in terrorist_graph.node[node]['SaysRed']:
        red_score+=terrorist_graph.node[n]["EstHonRed"]

        if terrorist_graph.node[n]["EstHonRed"] >max_red_hon:
            max_red_hon=terrorist_graph.node[n]["EstHonRed"]

    for n in terrorist_graph.node[node]['SaysBlue']:
        blue_score+=terrorist_graph.node[n]["EstHonBlue"]

        if terrorist_graph.node[n]["EstHonBlue"] >max_blue_hon:
            max_blue_hon=terrorist_graph.node[n]["EstHonBlue"]

    conf_red=max_red_hon
    conf_blue=max_blue_hon

    return {"redscore":red_score,"bluescore":blue_score,"confred":conf_red,"confblue":conf_blue}


def select_random_monitor(terrorist_graph):

    return np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])

def select_next_monitor_most_red_neighbors(terrorist_graph):
    max_red_score=0
    next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
            red_score=0


            for neighbor in terrorist_graph.neighbors(node):
                if terrorist_graph.node[neighbor]["color"]=="Red":
                    red_score+=1
            if red_score>max_red_score:
                max_red_score=red_score
                next_node=node

    return next_node

def select_next_monitor_most_red_says_red(terrorist_graph):
    max_red_score=0
    next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
            red_score=0
            says_red=terrorist_graph.node[node]["SaysRed"]

            for neighbor in says_red:
                if terrorist_graph.node[neighbor]["color"]=="Red":
                    red_score+=1
            if red_score>max_red_score:
                max_red_score=red_score
                next_node=node

    #print max_red_score
    return next_node

def select_next_monitor_highest_estimated_red_score(terrorist_graph,color_score_function=get_color_score_2):

    max_red_score=0
    max_conf=0
    next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
            results=color_score_function(terrorist_graph,node)
            rscore=results["redscore"]
            if rscore>max_red_score:
                max_red_score=rscore
                next_node=node


    return next_node




def estimated_honesty_recursive(terrorist_graph,color_score_function):

    color_changed=True
    confidence_changed=True
    iterations=0

    last_10_conf={node:{"Red":[],"Blue":[]} for node in terrorist_graph.nodes()}
    last_10_colors={node:[] for node in terrorist_graph.nodes()}

    last_10_honesty_blue={node:[] for node in terrorist_graph.nodes()}
    last_10_honesty_red={node:[] for node in terrorist_graph.nodes()}
    while (color_changed or confidence_changed) and iterations<100:


        color_changed=False
        confidence_changed=False
        iterations+=1

        for node in terrorist_graph:

            if not terrorist_graph.node[node]["IsMonitor"]:

                if terrorist_graph.node[node]["Discovered"]:



                    results=color_score_function(terrorist_graph,node)
                    red_score=results["redscore"]
                    blue_score=results["bluescore"]
                    confidence_red=results["confred"]
                    confidence_blue=results["confblue"]




                    if red_score>blue_score:
                        if terrorist_graph.node[node]["EstColor"][0]!="Red":
                            color_changed=True

                            terrorist_graph.node[node]["EstColor"]=["Red",confidence_red]

                        elif terrorist_graph.node[node]["EstColor"][1]!=confidence_red:
                            if abs(confidence_red-terrorist_graph.node[node]["EstColor"][1] )>0.1:
                                confidence_changed=True
                                terrorist_graph.node[node]["EstColor"]=["Red",confidence_red]
                        if len(last_10_colors[node]) >=10:
                            last_10_colors[node].pop()
                            last_10_colors[node].append("Red")
                        else:
                            last_10_colors[node].append("Red")
                        if len(last_10_conf[node]["Red"]) >=10:

                            last_10_conf[node]["Red"].pop()
                            last_10_conf[node]["Red"].append(confidence_red)
                        else:
                            last_10_conf[node]["Red"].append(confidence_red)

                    else:

                        if terrorist_graph.node[node]["EstColor"][0]!="Blue":
                            color_changed=True
                            confidence_changed=True

                            terrorist_graph.node[node]["EstColor"]=["Blue",confidence_blue]

                        elif terrorist_graph.node[node]["EstColor"][1]!=confidence_blue:
                            if abs(confidence_blue-terrorist_graph.node[node]["EstColor"][1]) >0.1:

                                confidence_changed=True
                                terrorist_graph.node[node]["EstColor"]=["Blue",confidence_blue]
                        if len(last_10_colors[node]) >=10:
                            last_10_colors[node].pop()
                            last_10_colors[node].append("Blue")
                        else:
                            last_10_colors[node].append("Blue")
                        if len(last_10_conf[node]["Blue"]) >=10:

                            last_10_conf[node]["Blue"].pop()
                            last_10_conf[node]["Blue"].append(confidence_blue)
                        else:
                            last_10_conf[node]["Blue"].append(confidence_blue)

        for nn in terrorist_graph:
            if terrorist_graph.node[nn]["IsMonitor"]:
                discovered_red_neighbors=[n for n in terrorist_graph.neighbors(nn) if terrorist_graph.node[n]['color']=="Red"]
                red_neighbors_said_red=[n for n in discovered_red_neighbors if nn in terrorist_graph.node[n]["SaysRed"]]
                discovered_blue_neighbors=[n for n in terrorist_graph.neighbors(nn) if terrorist_graph.node[n]['color']=="Blue"]
                blue_neighbors_said_blue=[n for n in discovered_blue_neighbors if nn in terrorist_graph.node[n]["SaysBlue"]]

                estimated_blue_neighbors=[n for n in terrorist_graph.neighbors(nn) if not terrorist_graph.node[n]["IsMonitor"] and terrorist_graph.node[n]["EstColor"][0]=="Blue"]
                est_blue_says_blue=[n for n in estimated_blue_neighbors if nn in terrorist_graph.node[n]["SaysBlue"]]
                estimated_red_neighbors=[n for n in terrorist_graph.neighbors(nn) if not terrorist_graph.node[n]["IsMonitor"] and terrorist_graph.node[n]["EstColor"][0]=="Red"]
                est_red_says_red=[n for n in estimated_red_neighbors if nn in terrorist_graph.node[n]["SaysRed"]]


                est_color_hon_red=0
                est_color_hon_blue=0


                if len(discovered_red_neighbors) >0 or len(estimated_red_neighbors)>0:

                    est_color_hon_red=(len(red_neighbors_said_red)+sum([terrorist_graph.node[n]["EstColor"][1] for n in est_red_says_red ]))/(len(discovered_red_neighbors)+len([terrorist_graph.node[n]["EstColor"][1] for n in estimated_red_neighbors ]))


                if len(discovered_blue_neighbors) >0 or len(estimated_blue_neighbors) >0:
                    #prob_true_blue=1.0*len(discovered_blue_neighbors)/(len(discovered_blue_neighbors)+len(estimated_blue_neighbors))

                    est_color_hon_blue=(len(blue_neighbors_said_blue)+sum([terrorist_graph.node[n]["EstColor"][1] for n in est_blue_says_blue ]))/(len(discovered_blue_neighbors)+len([terrorist_graph.node[n]["EstColor"][1] for n in estimated_blue_neighbors ]))

                if est_color_hon_blue>0:
                    terrorist_graph.node[nn]["EstHonBlue"]=est_color_hon_blue
                    if len(last_10_honesty_blue[nn]) >=10:
                        last_10_honesty_blue[nn].pop()
                        last_10_honesty_blue[nn].append(est_color_hon_blue)
                    else:
                      last_10_honesty_blue[nn].append(est_color_hon_blue)

                if est_color_hon_red>0:
                    terrorist_graph.node[nn]["EstHonRed"]=est_color_hon_red
                    if len(last_10_honesty_red[nn]) >=10:
                        last_10_honesty_red[nn].pop()
                        last_10_honesty_red[nn].append(est_color_hon_red)
                    else:
                       last_10_honesty_red[nn].append(est_color_hon_red)



    if iterations==100:
        for n in terrorist_graph:
            if terrorist_graph.node[n]["IsMonitor"]:
                if len(last_10_honesty_blue[n]) >0:
                    terrorist_graph.node[n]["EstHonBlue"]=np.mean(last_10_honesty_blue[n])
                if len(last_10_honesty_red[n])>0:
                    terrorist_graph.node[n]["EstHonRed"]=np.mean(last_10_honesty_red[n])
            if not terrorist_graph.node[n]["IsMonitor"] and terrorist_graph.node[n]["Discovered"] :
                if len(last_10_colors) >0:

                    from collections import Counter
                    data = Counter(last_10_colors[n])
                    color=data.most_common(1)[0][0]

                    conf=np.mean(last_10_conf[n][color])

                    terrorist_graph.node[n]["EstColor"]=[color,conf]





def evaluate_model_monitor(terrorist_graph):
    red_found=0
    for node in terrorist_graph:
        if terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["color"]=="Red":
            red_found+=1

    return red_found

def initialize_terrorist_graph(terrorist_graph):
    nx.set_node_attributes(terrorist_graph, 'SaysBlue', {node:[] for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'SaysRed', {node:[] for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'IsMonitor', {node:False for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'color', {node:"Black" for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstColor', {node:["Black",0] for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'Discovered', {node:False for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstHonRed', {node:0.5 for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstHonBlue', {node:0.5 for node in terrorist_graph.nodes()})



def draw_plots(all_performance,budgets,plot_name):

    #for s in lying_strategies.keys():
        for k in all_performance.keys():
            performance=all_performance[k]
            for monitor_placement in performance.keys():

                    average_per=[np.mean(performance[monitor_placement][budget]) for budget in sorted(budgets)]
                    std_perform=[np.std(performance[monitor_placement][budget]) for budget in sorted(budgets)]

                    plt.plot(sorted(budgets),average_per,label=monitor_placement)

                    print monitor_placement
                    print average_per
                    print std_perform


        plt.legend(loc=4)

        #plt.legend()
        plt.ylabel('# of Red Found')
        plt.xlabel('# of Monitors')
        plt.title(plot_name)
        plt.savefig(plot_name)
        plt.show()

def simulate_lying_monitor(G):

    color_score_functions=[get_color_score_2]
    with_honesty_values=["True","False"]
    lying_strategies={"strategy-1":get_neighbor_colors_setting1 ,"strategy-2":get_neighbor_colors_setting2,"strategy-3":get_neighbor_colors_setting3}
    next_monitor={"Most red neighbors":select_next_monitor_most_red_neighbors,"Highest red score":select_next_monitor_highest_estimated_red_score,"Most red says red":select_next_monitor_most_red_says_red,"Random monitor":select_random_monitor}

    budgets=[i for i in range(5,30,5)]
    all_performance={}


    for strategy_name in lying_strategies.keys():


        #for with_honesty in with_honesty_values:
        with_honesty="False"
        for color_score_func in color_score_functions:
                performance={}


                for monitor_placement in next_monitor.keys():
                    performance[monitor_placement]={}
                    for b in budgets:
                        performance[monitor_placement][b]=[]


                print with_honesty

                for i in range(0,100):
                    print ("Big I", i)
                    assign_centrality(G,nx.degree_centrality)
                    assign_honesty(G)
                    assign_colors(G_1)
                    red_nodes=[node for node in G.nodes() if G.node[node]['color']=="Red"]

                    no_runs=100
                    for j in range(0,no_runs):
                        start_node=np.random.choice(red_nodes)

                        for next_monitor_placement in next_monitor.keys():
                            # if next_monitor=="Highest red score":
                            #     with_honesty="True"
                            for b in budgets:

                                terrorist_graph=nx.Graph()
                                terrorist_graph.add_nodes_from(G.nodes())
                                initialize_terrorist_graph(terrorist_graph)



                                selected_node=start_node

                                budget=copy.deepcopy(b)

                                while budget>0:


                                    terrorist_graph.node[selected_node]["IsMonitor"]=True
                                    terrorist_graph.node[selected_node]['color']=G.node[selected_node]['color']
                                    terrorist_graph.node[selected_node]['Discovered']=True



                                    for edge in G.edges(selected_node):

                                        terrorist_graph.add_edge(edge[0],edge[1])



                                    neighbor_colors=lying_strategies[strategy_name](G,selected_node)


                                    for node in neighbor_colors:
                                        terrorist_graph.node[node]["Discovered"]=True
                                        if neighbor_colors[node]=="Red":
                                            terrorist_graph.node[node]["SaysRed"].append(selected_node)
                                        else:
                                            terrorist_graph.node[node]["SaysBlue"].append(selected_node)

                                    if with_honesty=="True" :
                                        estimated_honesty_recursive(terrorist_graph,color_score_func)

                                    selected_node=next_monitor[next_monitor_placement](terrorist_graph)
                                    budget-=1



                                evaluation=evaluate_model_monitor(terrorist_graph)
                                performance[next_monitor_placement][b].append(evaluation)






                # if color_score_func==get_color_score_2:
                #
                #
                #     all_performance["sum_honesty"]=performance
                # elif color_score_func==get_color_score_3:
                #     all_performance["max_honesty"]=performance
                # else:
                #     all_performance["naive_honesty"]=performance
                all_performance["d"]=performance


        draw_plots(all_performance,budgets,strategy_name+" without honesty")










G_1=nx.Graph(read_network_data_gexf("data//Noordin 14 Layers.gexf"))





simulate_lying_monitor(G_1)


