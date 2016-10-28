__author__ = 'pivithuru'

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import csv
import copy
import PokeC
import pickle
import Noordin
import math
from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

no_red=0
average_centrality=0


def read_network_data_gexf(path):
    return nx.read_gexf(path)

def read_network_data_gml(path):
    return nx.read_gml(path)

def draw_network(G):


    colors_dict={n:G.node[n]['color'] for n in G.nodes()}
    colors=colors_dict.values()
    nodes=colors_dict.keys()


    pos = nx.spring_layout(G)
    nx.draw_networkx_edges(G, pos, alpha=1)
    nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors,
                                with_labels=True, node_size=100)

    plt.axis('off')
    plt.show()



def network_details(G_2):



    G_3=assign_centrality(G_2)
    assign_honesty(G_3)


    print (len([n for n in G_3.nodes() if G_3.node[n]['color']=="Red"]),"# Red")
    print (len(G_3.edges()),"# Edges")
    print (len(G_3.nodes()),"# nodes")
    print average_centrality
    degree_dist={}
    for node in G_3.nodes():
         if G_3.node[node]['color']=="Red":
            deg=len(G_3.neighbors(node))


            if deg not in degree_dist.keys():
                degree_dist[deg]=0

            degree_dist[deg]+=1
    #plt.scatter(ba_x,ba_y,c='r',marker='s',s=50)
    print degree_dist
    plt.scatter(degree_dist.keys(),[math.log10(i) for i in degree_dist.values()],c='b')
    plt.xlabel("Degree")
    plt.ylabel("Log frequency")
    #plt.axis([0,max(degree_dist.keys())+5,0,max(degree_dist.values())+5])
    plt.show()





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



def assign_centrality(G):



    for u in G.nodes():

            #centrality=len([n for n in G.neighbors(u) if G.node[n]['color']=="Red"])
            # if len(G.neighbors(u))>0:
            #     centrality=1.0/len(G.neighbors(u))
            # else:
            #     centrality=0
            centrality=len(G.neighbors(u))
            G.node[u]['centrality']=centrality


    return G
def assign_random_centrality(G):

    for u in G.nodes():

            centrality=np.random.random_integers(1,20)
            G.node[u]['centrality']=centrality


    return G

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


def get_neighbor_colors_setting3(G,node):

    neighbor_colors={}

    for neighbor in G.neighbors(node):
        rand=np.random.random()
        if (G.node[node]['color']=="Blue"):
             if (G.node[neighbor]['color']=="Blue"):
                neighbor_colors[neighbor]="Blue"
             else:
                prob_lie=min((1.0*G.node[neighbor]['centrality']*(1-G.node[node]['honesty']))/G.node[node]['centrality'],1)
                if prob_lie>rand:
                    neighbor_colors[neighbor]="Blue"
                else:
                    neighbor_colors[neighbor]="Red"



        elif (G.node[node]['color']=="Red"):


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



def get_color_score_1(terrorist_graph,node):
    red_score=1
    blue_score=1

    for n in terrorist_graph.node[node]['SaysRed']:

        red_score*=terrorist_graph.node[n]["EstHonRed"]

        blue_score*=(1-terrorist_graph.node[n]["EstHonRed"])



    for n in terrorist_graph.node[node]['SaysBlue']:

        blue_score*=terrorist_graph.node[n]["EstHonBlue"]

        red_score*=(1-terrorist_graph.node[n]["EstHonBlue"])


    conf_red=0

    for n in terrorist_graph.node[node]['SaysRed']:
        red_score+=terrorist_graph.node[n]["EstHonRed"]

    for n in terrorist_graph.node[node]['SaysBlue']:
        blue_score+=terrorist_graph.node[n]["EstHonBlue"]



    conf_red=1.0*red_score/(red_score+blue_score)


    return {"redscore":red_score,"bluescore":blue_score,"confred":conf_red,"confblue":1-conf_red}

def get_color_score_2(terrorist_graph,node):
    red_score=0
    blue_score=0
    for n in terrorist_graph.node[node]['SaysRed']:
        red_score+=terrorist_graph.node[n]["EstHonRed"]

    for n in terrorist_graph.node[node]['SaysBlue']:
        blue_score+=terrorist_graph.node[n]["EstHonBlue"]



    conf_red=1.0*red_score/(red_score+blue_score)


    return {"redscore":red_score,"confred":conf_red,"confblue":1-conf_red}



def select_random_monitor(terrorist_graph,global_parameters):

    return np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])

def select_next_monitor_most_red_neighbors(terrorist_graph,global_parameters):
    max_red_score=0
    nodes_with_max_score=[]
    #next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
            red_score=0


            for neighbor in terrorist_graph.neighbors(node):
                if terrorist_graph.node[neighbor]["color"]=="Red":
                    red_score+=1
            if red_score>=max_red_score:
                if red_score > max_red_score:
                    nodes_with_max_score=[]
                max_red_score=red_score
                nodes_with_max_score.append(node)
                #next_node=node
    if global_parameters!=None and len(nodes_with_max_score)>1:

        next_node=break_ties(nodes_with_max_score,terrorist_graph,global_parameters)

    else:
        next_node=np.random.choice(nodes_with_max_score)
    return next_node

def break_ties(nodes_with_max_score,terrorist_graph,global_parameters):



    max_prob_red=0
    next=np.random.choice(nodes_with_max_score)

    for node in nodes_with_max_score:
        sum_prob_red=0
        count=0


        for n_r in terrorist_graph.node[node]["SaysRed"]:
            if terrorist_graph.node[n_r]["color"]=="Red":

                sum_prob_red+=global_parameters["p_red_ured_saysred"]
                count+=1
            else:
                sum_prob_red+=global_parameters["p_red_ublue_saysred"]
                count+=1
        for n_b in terrorist_graph.node[node]["SaysBlue"]:
             if terrorist_graph.node[n_b]["color"]=="Red":
                sum_prob_red+=global_parameters["p_red_ured_saysblue"]
                count+=1
             else:
                sum_prob_red+=global_parameters["p_red_ublue_saysblue"]
                count+=1
        prob_red=sum_prob_red/count

        if max_prob_red<prob_red:

            max_prob_red=prob_red
            next=node


    return next




def select_next_monitor_most_red_says_red(terrorist_graph,global_parameters):
    max_red_score=0
    nodes_with_max_score=[]
#    next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] :#and terrorist_graph.node[node]["Discovered"]==True:
            red_score=0
            says_red=terrorist_graph.node[node]["SaysRed"]

            for neighbor in says_red:
                if terrorist_graph.node[neighbor]["color"]=="Red":
                    red_score+=1
            if red_score>=max_red_score:
                if red_score>max_red_score:
                    nodes_with_max_score=[]

                max_red_score=red_score
                nodes_with_max_score.append(node)

    if global_parameters!=None and len(nodes_with_max_score)>1:
        next_node=break_ties(nodes_with_max_score,terrorist_graph,global_parameters)

    else:
        try:
            next_node=np.random.choice(nodes_with_max_score)
        except:
            draw_network(terrorist_graph)

    return next_node



def select_next_monitor_highest_estimated_red_score(terrorist_graph,global_parameters,color_score_function=get_color_score_2):

    max_red_score=0

    nodes_with_max_score=[]
    #next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
            results=color_score_function(terrorist_graph,node)
            red_score=results["redscore"]
            if red_score>=max_red_score:
                if red_score > max_red_score:
                    nodes_with_max_score=[]
                max_red_score=red_score
                nodes_with_max_score.append(node)
                #next_node=node
    if global_parameters!=None and len(nodes_with_max_score)>1:
        next_node=break_ties(nodes_with_max_score,terrorist_graph,global_parameters)

    else:
        next_node=np.random.choice(nodes_with_max_score)


    return next_node

def select_next_monitor_red_score_global_prob(terrorist_graph,global_parameters):


    max_red_score=0

    nodes_with_max_score=[]
    next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    if global_parameters!=None:
        for node in terrorist_graph:
            if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:

                count=0
                sum_prob_red=0


                for n_r in terrorist_graph.node[node]["SaysRed"]:
                    if terrorist_graph.node[n_r]["color"]=="Red":
                        sum_prob_red+=global_parameters["p_red_ured_saysred"]
                        count+=1
                    else:
                        sum_prob_red+=global_parameters["p_red_ublue_saysred"]
                        count+=1
                if count>0:
                    red_score=sum_prob_red*1.0/count
                else:
                    red_score=0
                if red_score>=max_red_score:
                    if red_score > max_red_score:
                        nodes_with_max_score=[]
                    max_red_score=red_score
                    nodes_with_max_score.append(node)
                #next_node=node
    if global_parameters!=None and len(nodes_with_max_score)>1:
        next_node=break_ties(nodes_with_max_score,terrorist_graph,global_parameters)

    elif len(nodes_with_max_score) >1:
        next_node=np.random.choice(nodes_with_max_score)


    return next_node
def select_next_monitor_tie_break(terrorist_graph,global_parameters):
    max_red_score=0

    nodes_with_max_score=[]
    #next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:

            red_score=0
            if red_score>=max_red_score:
                if red_score > max_red_score:
                    nodes_with_max_score=[]
                max_red_score=red_score
                nodes_with_max_score.append(node)
                #next_node=node
    if global_parameters!=None and len(nodes_with_max_score)>1:
        next_node=break_ties(nodes_with_max_score,terrorist_graph,global_parameters)

    else:
        next_node=np.random.choice(nodes_with_max_score)
    return next_node
def select_next_monitor_most_red_triangles(terrorist_graph):

    max_triangles=0
    #next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])

    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
            ego_graph=nx.ego_graph(terrorist_graph,node)
            ego_graph.remove_nodes_from([n for n in ego_graph if ego_graph.node[n]['color']=="Blue"])

            no_triangles=nx.triangles(ego_graph,node)

            if no_triangles>max_triangles:
                max_triangles=no_triangles
                next_node=node
    if max_triangles==0:
        node=select_next_monitor_most_red_neighbors(terrorist_graph)
    return node




def redscore_red_neighbors(terrorist_graph,color_score_function=get_color_score_2):

    max_red_score=0
    max_conf=0
    next_node=np.random.choice([node for node in terrorist_graph if terrorist_graph.node[node]["Discovered"]==True and terrorist_graph.node[node]["IsMonitor"]==False ])
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["Discovered"]==True:
            red_neighbors=0
            results=color_score_function(terrorist_graph,node)
            rscore_1=results["redscore"]

            for neighbor in terrorist_graph.neighbors(node):
                if terrorist_graph.node[neighbor]["color"]=="Red":
                    red_neighbors+=1
            rscore=(rscore_1*red_neighbors)
            if rscore>max_red_score:
                max_red_score=rscore
                next_node=node
    return next_node
def estimated_honesty_recursive(terrorist_graph,color_score_function,start_node):


    confidence_changed=True
    it_limit=20
    iterations=0

    last_10_conf={node:{"Red":[],"Blue":[]} for node in terrorist_graph.nodes()}

    last_10_honesty_blue={node:[] for node in terrorist_graph.nodes()}
    last_10_honesty_red={node:[] for node in terrorist_graph.nodes()}


    #affected_nodes=nx.single_source_shortest_path_length(terrorist_graph, start_node, cutoff=4).keys()
    #affected_nodes.append(start_node)

    affected_nodes=terrorist_graph.nodes()
    while (confidence_changed) and iterations<it_limit:



        confidence_changed=False
        iterations+=1

        for node in affected_nodes:

            if not terrorist_graph.node[node]["IsMonitor"]:

                if terrorist_graph.node[node]["Discovered"]:



                    results=color_score_function(terrorist_graph,node)

                    confidence_red=results["confred"]
                    confidence_blue=results["confblue"]







                    if terrorist_graph.node[node]["EstColor"]["Red"]!=confidence_red:

                        if abs(confidence_red-terrorist_graph.node[node]["EstColor"]["Red"] )>0.01:
                            confidence_changed=True
                            terrorist_graph.node[node]["EstColor"]["Red"]=confidence_red

                    if len(last_10_conf[node]["Red"]) >=10:

                        last_10_conf[node]["Red"].pop()
                        last_10_conf[node]["Red"].append(confidence_red)
                    else:
                        last_10_conf[node]["Red"].append(confidence_red)




                    if terrorist_graph.node[node]["EstColor"]["Blue"]!=confidence_blue:
                        if abs(confidence_blue-terrorist_graph.node[node]["EstColor"]["Blue"]) >0.01:

                            confidence_changed=True
                            terrorist_graph.node[node]["EstColor"]["Blue"]=confidence_blue

                    if len(last_10_conf[node]["Blue"]) >=10:

                        last_10_conf[node]["Blue"].pop()
                        last_10_conf[node]["Blue"].append(confidence_blue)
                    else:
                        last_10_conf[node]["Blue"].append(confidence_blue)

        for nn in affected_nodes:
            if terrorist_graph.node[nn]["IsMonitor"]:
                discovered_red_neighbors=[n for n in terrorist_graph.neighbors(nn) if terrorist_graph.node[n]['color']=="Red"]
                red_neighbors_said_red=[n for n in discovered_red_neighbors if nn in terrorist_graph.node[n]["SaysRed"]]
                discovered_blue_neighbors=[n for n in terrorist_graph.neighbors(nn) if terrorist_graph.node[n]['color']=="Blue"]
                blue_neighbors_said_blue=[n for n in discovered_blue_neighbors if nn in terrorist_graph.node[n]["SaysBlue"]]

                neighbors=[n for n in terrorist_graph.neighbors(nn) if not terrorist_graph.node[n]["IsMonitor"] and terrorist_graph.node[n]["Discovered"]==True]
                blue_says_blue=[n for n in neighbors if nn in terrorist_graph.node[n]["SaysBlue"]]
                red_says_red=[n for n in neighbors if nn in terrorist_graph.node[n]["SaysRed"]]


                est_color_hon_red=0
                est_color_hon_blue=0


                if len(discovered_red_neighbors) >0 or sum([terrorist_graph.node[n]["EstColor"]["Red"] for n in neighbors ])>0:

                    est_color_hon_red=(len(red_neighbors_said_red)+sum([terrorist_graph.node[n]["EstColor"]["Red"] for n in red_says_red ]))/(len(discovered_red_neighbors)+sum([terrorist_graph.node[n]["EstColor"]["Red"] for n in neighbors ]))


                if len(discovered_blue_neighbors) >0 or sum([terrorist_graph.node[n]["EstColor"]["Blue"] for n in neighbors ]) >0:

                    est_color_hon_blue=(len(blue_neighbors_said_blue)+sum([terrorist_graph.node[n]["EstColor"]["Blue"] for n in blue_says_blue ]))/(len(discovered_blue_neighbors)+sum([terrorist_graph.node[n]["EstColor"]["Blue"] for n in neighbors ]))

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



    if iterations==it_limit:
        for n in affected_nodes:
            if terrorist_graph.node[n]["IsMonitor"]:
                if len(last_10_honesty_blue[n]) >0:
                    terrorist_graph.node[n]["EstHonBlue"]=np.mean(last_10_honesty_blue[n])
                if len(last_10_honesty_red[n])>0:
                    terrorist_graph.node[n]["EstHonRed"]=np.mean(last_10_honesty_red[n])
            if not terrorist_graph.node[n]["IsMonitor"] and terrorist_graph.node[n]["Discovered"] :



                    conf_blue=np.mean(last_10_conf[n]["Blue"])
                    conf_red=np.mean(last_10_conf[n]["Red"])

                    terrorist_graph.node[n]["EstColor"]["Blue"]=conf_blue
                    terrorist_graph.node[n]["EstColor"]["Red"]=conf_red

def get_global_parameters(terrorist_graph):
    red_ured_saysred=0
    blue_ured_saysred=0
    red_ured_saysblue=0
    blue_ured_saysblue=0
    red_ublue_saysred=0
    blue_ublue_saysred=0
    red_ublue_saysblue=0
    blue_ublue_saysblue=0

    p_red_ured_saysred=0
    p_blue_ured_saysred=0
    p_red_ured_saysblue=0
    p_blue_ured_saysblue=0
    p_red_ublue_saysred=0
    p_blue_ublue_saysred=0
    p_red_ublue_saysblue=0
    p_blue_ublue_saysblue=0

    for node in [n for n in terrorist_graph.nodes() if terrorist_graph.node[n]['IsMonitor']==True] :
        if terrorist_graph.node[node]['color']=="Red":
            for n_red in terrorist_graph.node[node]["SaysRed"]:
                if terrorist_graph.node[n_red]["color"]=="Red":
                    red_ured_saysred+=1
                else:
                    red_ublue_saysred+=1
            for n_blue in terrorist_graph.node[node]["SaysBlue"]:
                if terrorist_graph.node[n_blue]["color"]=="Red":
                    red_ured_saysblue+=1
                else:
                    red_ublue_saysblue+=1
        else:
            for n_red in terrorist_graph.node[node]["SaysRed"]:
                if terrorist_graph.node[n_red]["color"]=="Red":
                    blue_ured_saysred+=1
                else:
                    blue_ublue_saysred+=1
            for n_blue in terrorist_graph.node[node]["SaysBlue"]:
                if terrorist_graph.node[n_blue]["color"]=="Red":
                    blue_ured_saysblue+=1
                else:
                    blue_ublue_saysblue+=1
    if red_ured_saysred+blue_ured_saysred>0:
        p_red_ured_saysred=1.0*red_ured_saysred/(red_ured_saysred+blue_ured_saysred)
        p_blue_ured_saysred=1.0*blue_ured_saysred/(red_ured_saysred+blue_ured_saysred)
    if red_ured_saysblue+blue_ured_saysblue>0:
        p_red_ured_saysblue=1.0*red_ured_saysblue/(red_ured_saysblue+blue_ured_saysblue)
        p_blue_ured_saysblue=1.0*blue_ured_saysblue/(red_ured_saysblue+blue_ured_saysblue)
    if red_ublue_saysred+blue_ublue_saysred>0:
        p_red_ublue_saysred=1.0*red_ublue_saysred/(red_ublue_saysred+blue_ublue_saysred)
        p_blue_ublue_saysred=1.0*blue_ublue_saysred/(red_ublue_saysred+blue_ublue_saysred)
    if red_ublue_saysblue+blue_ublue_saysblue>0:
        p_red_ublue_saysblue=1.0*red_ublue_saysblue/(red_ublue_saysblue+blue_ublue_saysblue)
        p_blue_ublue_saysblue=1.0*blue_ublue_saysblue/(red_ublue_saysblue+blue_ublue_saysblue)

    return {"p_red_ured_saysred":p_red_ured_saysred,
            "p_blue_ured_saysred":p_blue_ured_saysred,
            "p_red_ured_saysblue":p_red_ured_saysblue,
            "p_blue_ured_saysblue":p_blue_ured_saysblue,
            "p_red_ublue_saysred":p_red_ublue_saysred,
            "p_blue_ublue_saysred":p_blue_ublue_saysred,
            "p_red_ublue_saysblue":p_red_ublue_saysblue,
            "p_blue_ublue_saysblue":p_blue_ublue_saysblue}





def learning_model(predict_model,training_set,predict_set):
    max_red_prb=0
    if len(training_set)%10==0:
        x=[]
        y=[]
        for n in training_set:
            x.append(training_set[n][:7])

            y.append(training_set[n][7])
        predict_model = linear_model.LogisticRegression(C=1e5)


        #predict_model.fit(x,y)
        #ranfor=RandomForestClassifier(n_estimators=100)





        predict_model.fit(x[:3*len(x)/4], y[:3*len(x)/4])
        predictions=predict_model.predict(x[3*len(x)/4:])
        prob=predict_model.predict_proba(x[3*len(x)/4:])
        true_class=y[3*len(x)/4:]
        print predictions
        print true_class
        print prob
        #print precision_recall_fscore_support(true_class,predictions)
        # ranfor.fit(x[:3*len(x)/4], y[:3*len(x)/4])
        # print len(x)
        print ("log",predict_model.score(x[3*len(x)/4:],y[3*len(x)/4:]))
        # print ("ran",ranfor.score(x[3*len(x)/4:],y[3*len(x)/4:]))
        #print logreg.feature_importances_

    next=np.random.choice(predict_set.keys())
    probs_array=predict_model.predict_proba(predict_set.values())

    for node in predict_set.keys():
        i=0
        probs=probs_array[i]
        i+=1
        red_prob=probs[1]
        blue_prob=probs[0]

        if blue_prob <red_prob:

            if red_prob>max_red_prb:
                max_red_prb=red_prob
                next=node

    return predict_model,next


def update_features(selected_node,terrorist_graph,training,predict):
    training[selected_node]=predict[selected_node]
    if terrorist_graph.node[selected_node]['color']=="Red":
        training[selected_node].append(1.0)
    else:
        training[selected_node].append(0.0)
    no_red_neighbors=len([n for n in terrorist_graph.neighbors(selected_node) if terrorist_graph.node[n]["color"]=="Red"])
    no_blue_neighbors=len([n for n in terrorist_graph.neighbors(selected_node) if terrorist_graph.node[n]["color"]=="Blue"])
    unknown_neighbors=len([n for n in terrorist_graph.neighbors(selected_node) if terrorist_graph.node[n]["color"]!="Blue" and terrorist_graph.node[n]["color"]!="Red"] )
    for neighbor in terrorist_graph.neighbors(selected_node):
        if neighbor not in predict.keys():
            predict[neighbor]=[0,0,0,0,0,0,0]

        if terrorist_graph.node[selected_node]['color']=="Red":
            predict[neighbor][0]+=1
        else:
            predict[neighbor][1]+=1

        predict[neighbor][2]+=no_red_neighbors
        predict[neighbor][3]+=no_blue_neighbors
        #predict[neighbor][4]+=unknown_neighbors
        predict[neighbor][5]=len(terrorist_graph.node[neighbor]["SaysRed"])
        predict[neighbor][6]=len(terrorist_graph.node[neighbor]["SaysBlue"])
    del predict[selected_node]
    return training,predict




def evaluate_model_monitor(terrorist_graph):
    red_found=0

    for node in terrorist_graph:
        if terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["color"]=="Red":
            red_found+=1


    return red_found
def assign_color_I_say(lying_strategy,G):
    all_neighbor_colors={}
    for node in G.nodes():
        colors=lying_strategy(G,node)
        all_neighbor_colors[node]=colors
    return all_neighbor_colors



def initialize_terrorist_graph(terrorist_graph):
    nx.set_node_attributes(terrorist_graph, 'SaysBlue', {node:[] for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'SaysRed', {node:[] for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'IsMonitor', {node:False for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'color', {node:"Black" for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstColor', {node:{"Blue":0.000001,"Red":0.000001} for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'Discovered', {node:False for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstHonRed', {node:0.5 for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstHonBlue', {node:0.5 for node in terrorist_graph.nodes()})

def initialize_node(terrorist_graph,n):
    terrorist_graph.add_node(n)
    terrorist_graph.node[n]["SaysBlue"]=[]
    terrorist_graph.node[n]["SaysRed"]=[]
    terrorist_graph.node[n]["IsMonitor"]=False
    terrorist_graph.node[n]["Discovered"]=True
    terrorist_graph.node[n]["EstHonRed"]=0.5
    terrorist_graph.node[n]["EstHonBlue"]=0.5
    terrorist_graph.node[n]["color"]="Black"
    terrorist_graph.node[n]["EstColor"]={"Blue":0.000001,"Red":0.000001}




def draw_plots(performance,budgets,plot_name):

    #for s in lying_strategies.keys():

        data=[]
        header=[]
        header.append("budget")
        for monitor_placement in sorted(performance.keys()):
            header.append("Average "+monitor_placement)
            header.append("STD "+monitor_placement)
        data.append(header)

        for budget in sorted(budgets):

            vals=[budget]


            for monitor_placement in sorted(performance.keys()):
                        avg=np.mean(performance[monitor_placement][budget])
                        vals.append(avg)
                        std=np.std(performance[monitor_placement][budget])
                        vals.append(std)

            data.append(vals)




        with open("plots/"+plot_name+".csv","wb") as f:
            csv_writer=csv.writer(f)
            csv_writer.writerows(data)



        #
        #         plt.plot(sorted(budgets),average_per,label=monitor_placement)
        #
        #         print monitor_placement
        #         print average_per
        #         print std_perform
        #
        #
        # plt.legend(loc=4)
        #
        # #plt.legend()
        # plt.ylabel('# of Red Found')
        # plt.xlabel('# of Monitors')
        # plt.ylim((1,no_red))
        # plt.title(plot_name)
        # plt.savefig("plots/"+plot_name)
        # plt.show()

def simulate_lying_monitor(G_1,name):


    lying_strategies={"strategy-2":get_neighbor_colors_setting2,
                      "strategy-3":get_neighbor_colors_setting3,
                      "strategy-1":get_neighbor_colors_setting1}
    next_monitor={"Most red neighbors":select_next_monitor_most_red_neighbors,
                  "tie breaking red neighbors":select_next_monitor_most_red_neighbors,
                  "tie breaking highest score":select_next_monitor_highest_estimated_red_score,
                  "Highest red score":select_next_monitor_highest_estimated_red_score,
                  "Most red says red":select_next_monitor_most_red_says_red,
                  "Random monitor":select_random_monitor
                  }
                  #"Highest red score with est. Honesty":select_next_monitor_highest_estimated_red_score}
                  #"redscore_red_neigh":redscore_red_neighbors}
                  #"Est red neignbors with est hon":select_next_monitor_estimated_red_neighbors,
                  #"Est red neignbors score":select_next_monitor_estimated_red_neighbors_score}
    #next_monitor={"Highest red score":select_next_monitor_highest_estimated_red_score,}

    #"strategy-2":get_neighbor_colors_setting2, ,"strategy-3":get_neighbor_colors_setting3
    G=assign_centrality(G_1)
    red_nodes=[node for node in G.nodes() if G.node[node]['color']=="Red"]
    print len(red_nodes)

    #budgets=[i for i in range(len(red_nodes),6*len(red_nodes),len(red_nodes))]
    budgets=[]
    budgets.append(len(G.nodes())/100)
    budgets.extend([len(G.nodes())*i/100 for i in range(10,60,10)])

    print budgets



    for strategy_name in lying_strategies.keys():


                performance={}
                color_score_func=get_color_score_2
                all_red_found={}

                for monitor_placement in next_monitor.keys():
                    performance[monitor_placement]={}
                    all_red_found[monitor_placement]=[]
                    for b in budgets:
                        performance[monitor_placement][b]=[]




                for i in range(0,10):
                    print ("Big I", i)

                    assign_honesty(G)
                    #assign_colors(G)


                    global no_red
                    no_red=len(red_nodes)
                    no_runs=10
                    for j in range(0,no_runs):

                        start_node=np.random.choice(red_nodes)
                        while len(G.neighbors(start_node))==0:
                            start_node=np.random.choice(red_nodes)
                        all_neighbor_colors=assign_color_I_say(lying_strategies[strategy_name],G)


                        for next_monitor_placement in next_monitor.keys():


                                for b in budgets:

                                    global_parameters=None
                                    terrorist_graph=nx.Graph()

                                    selected_node=start_node
                           
                                    initialize_node(terrorist_graph,selected_node)




                                    budget=copy.deepcopy(b)

                                    no_monitors=0
                                    while budget>0:
                                        no_monitors+=1

                                        terrorist_graph.node[selected_node]["IsMonitor"]=True
                                        terrorist_graph.node[selected_node]['color']=G.node[selected_node]['color']
                                        terrorist_graph.node[selected_node]['Discovered']=True


                                        #draw_network(terrorist_graph)


                                        for edge in G.edges(selected_node):

                                            if not terrorist_graph.has_node(edge[0]):

                                                initialize_node(terrorist_graph,edge[0])

                                            if not terrorist_graph.has_node(edge[1]):

                                                initialize_node(terrorist_graph,edge[1])

                                            terrorist_graph.add_edge(edge[0],edge[1])







                                        neighbor_colors=all_neighbor_colors[selected_node]



                                        for node in neighbor_colors:


                                            if neighbor_colors[node]=="Red":
                                                terrorist_graph.node[node]["SaysRed"].append(selected_node)
                                            else:
                                                terrorist_graph.node[node]["SaysBlue"].append(selected_node)

                                        if no_monitors>10 and (next_monitor_placement=="tie breaking red neighbors" or next_monitor_placement=="tie breaking highest score") :

                                            global_parameters=get_global_parameters(terrorist_graph)


                                        if next_monitor_placement=="Highest red score with est. Honesty":

                                            estimated_honesty_recursive(terrorist_graph,color_score_func,selected_node)




                                        #else:
                                        selected_node=next_monitor[next_monitor_placement](terrorist_graph,global_parameters)
                                        budget-=1



                                    evaluation=evaluate_model_monitor(terrorist_graph)

                                    all_red_found[next_monitor_placement].extend([G.node[n]["centrality"] for n in terrorist_graph if terrorist_graph.node[n]["color"]=="Red"])

                                    performance[next_monitor_placement][b].append(evaluation)




                for mplace in all_red_found.keys():
                    print(mplace,np.mean(all_red_found[mplace]))


                draw_plots(performance,budgets,strategy_name+"_"+name)



def measure_homophily(G):

    red_node_neighbors=[]
    blue_node_neighbors=[]


    for node in G.nodes():
        blue_neigh=0
        red_neigh=0
        if len(G.neighbors(node))>0:
            for neighb in G.neighbors(node):
                if G.node[neighb]['color']=="Blue":
                    blue_neigh+=1
                if G.node[neighb]['color']=="Red":
                    red_neigh+=1
            if G.node[node]["color"]=="Blue":
                blue_node_neighbors.append((blue_neigh,red_neigh))
            else:
                red_node_neighbors.append((blue_neigh,red_neigh))
    red_red_percentage=0
    red_blue_percentage=0

    blue_blue_percentage=0
    blue_red_percentage=0

    for neighbor_stat in red_node_neighbors:
        red_red_percentage+=neighbor_stat[1]*1.0/(neighbor_stat[1]+neighbor_stat[0])
        red_blue_percentage+=neighbor_stat[0]*1.0/(neighbor_stat[1]+neighbor_stat[0])
    print ("Red Red Prob",red_red_percentage/len(red_node_neighbors))
    print ("Red_Blue_prob",red_blue_percentage/len(red_node_neighbors))

    for neighbor_stat in blue_node_neighbors:
        blue_red_percentage+=neighbor_stat[1]*1.0/(neighbor_stat[1]+neighbor_stat[0])
        blue_blue_percentage+=neighbor_stat[0]*1.0/(neighbor_stat[1]+neighbor_stat[0])
    print ("Blue blue Prob",blue_blue_percentage/len(blue_node_neighbors))
    print ("Blue red prob",blue_red_percentage/len(blue_node_neighbors))




def remove_red_edges(G):
    remove=[]
    for edge in G.edges():
        if G.node[edge[0]]['color']=="Red" and G.node[edge[1]]['color']=="Red":
            remove.append(edge)
    G.remove_edges_from(remove)
    return G


#
# G_1=assign_colors_noordin_8k(G_1)
G_2=pickle.load(open("pokec_sampled//Pokec_kosicky_badminton"))


# G_3=PokeC.assign_colors(G_2)
# RandomWalk.save_object(G_3,"pokec_sampled///Pokec_10000_colored")
# print G_2.nodes()
#draw_network(G_1)

#G_2=Noordin.get_noordin8k()
#draw_network(G_2.subgraph([n for n in G_2.nodes() if G_2.node[n]['color']=="Red"]))



#draw_network(G_2.subgraph([n for n in G_2.nodes() if G_2.node[n]['color']=="Red"]))
# name="Noordin_new_l"
#draw_network(nx.Graph(read_network_data_gexf("data//Train_Noordin_ExtComms_Computer-based.gexf")))



#Networks


    # Train_Noordin_ExtComms_Computer-based.gexf
    #Train_Noordin_ExtComms_Videos.gexf
    #Train_Noordin_ExtComms_Support-materials.gexf
    #Train_Noordin_ExtComms_Print-media.gexf
    #Train_Noordin_ExtComms_Unknown-commo.gexf

#G_2=Noordin.get_noordin_network("data//Train_Noordin_ExtComms_Unknown-commo.gexf")
# # for node in G_2:
# #     if G_2.node[node]["color"]=="Red":
# #
# #         print (node,nx.degree(G_2,node))
G_2=remove_red_edges(G_2)

simulate_lying_monitor(G_2,"Pokec_badminton_no_red_edges")


#network_details(G_2)
#draw_network(nx.Graph(read_network_data_gexf("data//Train_Noordin_ExtComms_Computer-based.gexf")))
#network_details(G_2)

#measure_homophily(G_2)