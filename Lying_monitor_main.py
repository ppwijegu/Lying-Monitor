__author__ = 'pivithuru'
__author__ = 'pivithuru'


import networkx as nx

import numpy as np
import csv
import copy
import Noordin

import warnings
import Learning_based_monitors
import baseline_monitor
import Lying_Strategies
import Traingles





warnings.filterwarnings("ignore")


def read_network_data_gexf(path):
    return nx.read_gexf(path)

def read_network_data_gml(path):
    return nx.read_gml(path)






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
    nx.set_node_attributes(terrorist_graph, 'tempColor', {node:"Grey" for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstColor', {node:{"Blue":0.000001,"Red":0.000001} for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'Discovered', {node:False for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstHonRed', {node:0.5 for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'EstHonBlue', {node:0.5 for node in terrorist_graph.nodes()})
    nx.set_node_attributes(terrorist_graph, 'probRed', {node:0.0 for node in terrorist_graph.nodes()})

def initialize_node(terrorist_graph,n):
    terrorist_graph.add_node(n)
    terrorist_graph.node[n]["SaysBlue"]=[]
    terrorist_graph.node[n]["SaysRed"]=[]
    terrorist_graph.node[n]["IsMonitor"]=False
    terrorist_graph.node[n]["Discovered"]=True
    terrorist_graph.node[n]["EstHonRed"]=0.5
    terrorist_graph.node[n]["EstHonBlue"]=0.5
    terrorist_graph.node[n]["color"]="Black"
    terrorist_graph.node[n]["tempColor"]="Grey"
    terrorist_graph.node[n]["EstColor"]={"Blue":0.000001,"Red":0.000001}
    terrorist_graph.node[n]["MonitorNumber"]=0
    terrorist_graph.node[n]["RedConfidence"]=0
    terrorist_graph.node[n]["probRed"]=0.0

def save_feature_importance(feature_importance_data,file_name):

    features=Learning_based_monitors.features
    average_importance={}
    data=[]
    header=[]
    header.append("budget")

    header.extend(features)

    data.append(header)



    for budget in feature_importance_data:
        vals=[budget]
        importance_dict={}

        f_average_importance={}

        for i in range(0,len(features)):
            f_average_importance[i]=[]

        for importance_array in feature_importance_data[budget]:


            for i in range(len(importance_array)):

                value=importance_array[i]


                if np.isnan(value):

                    value=0

                f_average_importance[i].append(abs(value))

        for i in range(len(features)):
            vals.append(np.mean(f_average_importance[i]))
            importance_dict[features[i]]=np.mean(f_average_importance[i])
        average_importance[budget]=importance_dict

        data.append(vals)

    with open("results/journal/"+file_name+".csv","wb") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(data)
    return average_importance









##Save the results in a file
def save_results(performance,performance_reported,budgets,file_name):



    data=[]
    header=[]
    header.append("budget")
    for monitor_placement in sorted(performance.keys()):
        header.append("Average "+monitor_placement)
        header.append("STD "+monitor_placement)
        header.append("Average "+monitor_placement+"reported")
        header.append("STD "+monitor_placement+"reported")

    data.append(header)

    for budget in sorted(budgets):

        vals=[budget]


        for monitor_placement in sorted(performance.keys()):
                    avg=np.mean(performance[monitor_placement][budget])
                    vals.append(avg)
                    std=np.std(performance[monitor_placement][budget])
                    vals.append(std)
                    avg2=np.mean(performance_reported[monitor_placement][budget])
                    vals.append(avg2)
                    std2=np.std(performance_reported[monitor_placement][budget])
                    vals.append(std2)


        data.append(vals)

    with open("results/"+file_name+".csv","wb") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(data)


def process_monitor(G,red_count,terrorist_graph,all_neighbor_colors,selected_node,update,evaluation,evaluation_reported_color,node_lie=False):

                ####Change evaluataion
                # if G.node[selected_node]["color"]=="Red" and terrorist_graph.node[selected_node]["MonitorNumber"]==0:
                #
                #
                #         evaluation+=1

                red_terrorist=len([n for n in terrorist_graph.nodes() if terrorist_graph.node[n]["color"]=="Red"])

                if G.node[selected_node]["color"]=="Red":


                    evaluation+=1


                terrorist_graph.node[selected_node]["IsMonitor"]=True

                if node_lie and len(terrorist_graph.nodes())!=1:
                    terrorist_graph.node[selected_node]['color']=Lying_Strategies.get_color_I_am(G,selected_node)

                else:
                    terrorist_graph.node[selected_node]['color']=G.node[selected_node]['color']
                terrorist_graph.node[selected_node]['tempColor']=terrorist_graph.node[selected_node]['color']

                if terrorist_graph.node[selected_node]["color"]=="Red":


                    evaluation_reported_color+=1



                terrorist_graph.node[selected_node]['tempColor']=terrorist_graph.node[selected_node]['color']

                terrorist_graph.node[selected_node]['Discovered']=True
                terrorist_graph.node[selected_node]["MonitorNumber"]=terrorist_graph.node[selected_node]["MonitorNumber"]+1


                for edge in G.edges(selected_node):


                        if not terrorist_graph.has_node(edge[0]):

                            initialize_node(terrorist_graph,edge[0])


                        if not terrorist_graph.has_node(edge[1]):

                            initialize_node(terrorist_graph,edge[1])


                        terrorist_graph.add_edge(edge[0],edge[1])

                neighbor_colors=all_neighbor_colors[selected_node]



                for node in neighbor_colors:


                        update.append(node)

                        if neighbor_colors[node]=="Red":
                            if selected_node not in terrorist_graph.node[node]["SaysRed"]:
                                terrorist_graph.node[node]["SaysRed"].append(selected_node)
                        else:
                            if selected_node not in terrorist_graph.node[node]["SaysBlue"]:
                                terrorist_graph.node[node]["SaysBlue"].append(selected_node)



                update.append(selected_node)
                return terrorist_graph,update,evaluation,evaluation_reported_color


##Input graph and name of the file you want to store results in
##This method simulate the lying strategies and different monitor placement strategies to get results.
## For changing importance of nodes I have used assign_centrality functions. When you use different importance change this function accordingly.
def simulate_lying_monitor(G_prime,file_name,nodes_lie):

    print nodes_lie


    lying_strategies={
                    "strategy-2":Lying_Strategies.get_neighbor_colors_setting2,
                   "strategy-1":Lying_Strategies.get_neighbor_colors_setting1 }
    next_monitor={"Most red neighbors":baseline_monitor.select_next_monitor_most_red_neighbors,

                    "Highest red score":baseline_monitor.select_next_monitor_highest_estimated_red_score,
                    "Most red says red": baseline_monitor.select_next_monitor_most_red_says_red,
                    "Random monitor":baseline_monitor.select_random_monitor,
                    "Learning":Learning_based_monitors.train_learning_model,
                  }

    #########################
    ####  Only change this line when you develop different node importance functions
    ###########################
    #G=assign_centrality(G)   ##Assign importance to nodes. Change this function to change importance of nodes

    red_nodes=[node for node in G_prime.nodes() if G_prime.node[node]['color']=="Red"]
    print len(red_nodes)
    red_count=len([n for n in red_nodes if G_prime.degree(n)>0])



    #Budget changed
    max_budget=len(G_prime.nodes())/2
    budgets=[i for i in range(1,max_budget+1)]




    for strategy_name in lying_strategies.keys():  #For different lying strategies considered


        prob_performance={}

        for prob in range(0,11,5):  #Change the range for required probabilities



                print strategy_name
                print prob
                performance={}
                performance_reported_color={}

                feature_importance_data={}
                feature_importance_data2={}

                for monitor_placement in next_monitor.keys():
                    performance[monitor_placement]={}
                    performance_reported_color[monitor_placement]={}

                    for b in budgets:
                        performance[monitor_placement][b]=[]
                        performance_reported_color[monitor_placement][b]=[]




                for i in range(0,3): # Number of runs with different honesty


                    G_modify=copy.deepcopy(G_prime)
                    G=remove_red_edges(G_modify,prob*0.1) #Remove homophily

                    Lying_Strategies.assign_honesty(G)


                    no_runs=3
                    for j in range(0,no_runs):  # Number of runs with different start nodes and neighbor colors



                        print j
                        start_node=np.random.choice(red_nodes)
                        while len(G.neighbors(start_node))==0:  #select a node with neighbors as the start node
                            start_node=np.random.choice(red_nodes)

                        all_neighbor_colors=Lying_Strategies.assign_color_I_say(lying_strategies[strategy_name],G)


                        Lying_Strategies.prob_lying_own_color=np.random.normal(0.5,0.125)


                        for next_monitor_placement in next_monitor.keys():          #keep honesty,neighbor colors same for all monitor placement strategies and evaluate them


                                    baseline_monitor.initialize()
                                    Learning_based_monitors.initialize()


                                    terrorist_graph=nx.Graph()

                                    selected_node=start_node

                                    initialize_node(terrorist_graph,selected_node)

                                    budget=max_budget
                                    no_monitors=0


                                    update=[]
                                    evaluation=0
                                    evaluation_reported_color=0

                                    found_red=[start_node]

                                    print next_monitor_placement
                                    while budget>0:


                                        no_monitors+=1

                                        terrorist_graph,update,evaluation,evaluation_reported_color=process_monitor(G,red_count,terrorist_graph,all_neighbor_colors,selected_node,update,evaluation,evaluation_reported_color,nodes_lie)
                                        budget-=1


                                        if next_monitor_placement=="Learning":

                                                leads=False


                                                if next_monitor_placement=="Learning with lead":

                                                    leads=True


                                                if no_monitors>=5: #learn the model after placing 5 monitors


                                                    selected_node,f_importance,f_imp2=Learning_based_monitors.learning_model(G,terrorist_graph,update,True,nodes_lie,collective=False,with_leads=leads)

                                                    if no_monitors not in feature_importance_data.keys():
                                                        feature_importance_data[no_monitors]=[]
                                                        feature_importance_data2[no_monitors]=[]

                                                    feature_importance_data[no_monitors].append(f_importance[0])
                                                    feature_importance_data2[no_monitors].append(f_imp2)
                                                    update=[]

                                                elif no_monitors<5:
                                                    selected_node=baseline_monitor.select_random_monitor(terrorist_graph,nodes_lie)
                                                else:
                                                    selected_node,f_importance,f_imp2=Learning_based_monitors.learning_model(G,terrorist_graph,update,False,nodes_lie,collective=False,with_leads=leads)

                                        #Collective classification based monitor placement

                                        elif next_monitor_placement=="Collective":

                                                if no_monitors>=5: #start the learn model after placing 5 monitors

                                                    selected_node,f_importance,f_imp2=Learning_based_monitors.learning_model(G,terrorist_graph,update,True,nodes_lie,collective=False)
                                                    update=[]

                                                elif no_monitors<5:
                                                    selected_node=baseline_monitor.select_random_monitor(terrorist_graph,nodes_lie)
                                                else:
                                                    selected_node=Learning_based_monitors.learning_model(G,terrorist_graph,update,False,nodes_lie,collective=True)


                                        elif next_monitor_placement=="Random monitor":
                                            selected_node=baseline_monitor.select_random_monitor(terrorist_graph,nodes_lie)




                                        else:
                                            selected_node=next_monitor[next_monitor_placement](terrorist_graph,selected_node,nodes_lie)







                                        if selected_node!=None:
                                            performance[next_monitor_placement][no_monitors].append(evaluation)
                                            performance_reported_color[next_monitor_placement][no_monitors].append(evaluation_reported_color)
                                            if G.node[selected_node]["color"]=="Red":
                                                found_red.append(selected_node)

                                        else:
                                            while budget>0:
                                                no_monitors+=1
                                                performance[next_monitor_placement][no_monitors].append(evaluation)
                                                performance_reported_color[next_monitor_placement][no_monitors].append(evaluation_reported_color)
                                                budget-=1




                if nodes_lie:
                    prob_performance[prob]=performance_reported_color
                else:
                    prob_performance[prob]=performance



                save_results(performance,performance_reported_color,budgets,strategy_name+"_"+file_name+"_remove_edge_prob_"+str(prob))





##Modify this function to change the % of red edges removes from the network. Call this before inputting the graph to simulation function.


def remove_red_edges(G,remove_prob):
    remove=[]

    ## When this prob is 0 we remove all the the edges. If this is 1, we keep all the red edges.
    for edge in G.edges():
        if G.node[edge[0]]['color']=="Red" and G.node[edge[1]]['color']=="Red":
            rand=np.random.random()
            if rand<remove_prob:
                remove.append(edge)
    G.remove_edges_from(remove)
    return G
def remove_red_edges_at_visit(G,remove_prob):

    G_prime=nx.DiGraph(G)

    for node in G_prime:
        for neighbor in G.neighbors(node):
            if G.node[node]['color']=="Red" and G.node[neighbor]['color']=="Red":
                rand=np.random.random()
                if rand<remove_prob:
                    G_prime.remove_edge(node,neighbor)



    return G_prime





noordin_data=["Train_Noordin_ExtComms_Computer-based.gexf","Train_Noordin_ExtComms_Videos.gexf","Train_Noordin_ExtComms_Support-materials.gexf","Train_Noordin_ExtComms_Unknown-commo.gexf","Train_Noordin_ExtComms_Print-media.gexf"]

noordin_data=["Train_Noordin_ExtComms_Computer-based.gexf"]
for network in noordin_data:
    G=Noordin.get_noordin_network("data//"+network)  ##Graph you are selecting to input. Change the path according to your input

    Learning_based_monitors.triangles=Traingles.get_all_triangles(G)
    nodes_lie=False
    simulate_lying_monitor(G,network+"_testing",nodes_lie) # Input the graph and file name to store data









