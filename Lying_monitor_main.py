__author__ = 'pivithuru'

import networkx as nx

import numpy as np
import csv
import copy
import NetworkData
import warnings
import Learning_based_monitors
import baseline_monitor
import Lying_Strategies
import Traingles





warnings.filterwarnings("ignore")

class LyingMonitorMain():

    def __init__(self, G, start_learning, training_it):

        self.G = G

        self.start_learning = start_learning  # this decides when to start training the learning algorithm

        # this decides how often we should retrain the learning algorithm. For small networks set this to 1. For large networks 20.

        self.training_it = training_it

        # if you design new lying strategies add them here
        self.lying_strategies={
                        "strategy-2":Lying_Strategies.get_neighbor_colors_setting2,
                       "strategy-1":Lying_Strategies.get_neighbor_colors_setting1 }

        # if you design new monitor placement algorithms add them here
        self.next_monitor = {"Most red neighbors":baseline_monitor.select_next_monitor_most_red_neighbors,
                        "Highest red score":baseline_monitor.select_next_monitor_highest_estimated_red_score,
                        "Most red says red": baseline_monitor.select_next_monitor_most_red_says_red,
                        "Random monitor":baseline_monitor.select_random_monitor,
                        "Learning":Learning_based_monitors.train_learning_model,
                      }

        # this controls how many monitors to place
        self.max_budget=len(self.G.nodes())/2

        self.budgets = [i for i in range(1,self.max_budget+1)]

        # usually needs to run 10 times and take the average results since the algorithms are probabilistic.
        # make these two variables smaller to run the code faster
        self.runs_honesty = 5
        self.runs_neighbor_color = 5




    def evaluate_model_monitor(self, terrorist_graph):
        red_found=0

        for node in terrorist_graph:
            if terrorist_graph.node[node]["IsMonitor"] and terrorist_graph.node[node]["color"]=="Red":
                red_found+=1


        return red_found




    def initialize_terrorist_graph(self, terrorist_graph):
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

    def initialize_node(self,terrorist_graph,n):
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



    ##Save the results in a file
    def save_results(self,performance,performance_reported,file_name):



        data=[]
        header=[]
        header.append("budget")
        for monitor_placement in sorted(performance.keys()):
            header.append("Average "+monitor_placement)
            header.append("STD "+monitor_placement)
            header.append("Average "+monitor_placement+"reported")
            header.append("STD "+monitor_placement+"reported")

        data.append(header)

        for budget in sorted(self.budgets):

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


    def process_monitor(self,G_changed, terrorist_graph,all_neighbor_colors,selected_node,update,evaluation,evaluation_reported_color,node_lie=False):



                    if G_changed.node[selected_node]["color"]=="Red":


                        evaluation+=1


                    terrorist_graph.node[selected_node]["IsMonitor"]=True

                    if node_lie and len(terrorist_graph.nodes())!=1:
                        terrorist_graph.node[selected_node]['color']=Lying_Strategies.get_color_I_am(G_changed,selected_node)

                    else:
                        terrorist_graph.node[selected_node]['color']=G_changed.node[selected_node]['color']
                    terrorist_graph.node[selected_node]['tempColor']=terrorist_graph.node[selected_node]['color']

                    if terrorist_graph.node[selected_node]["color"]=="Red":


                        evaluation_reported_color+=1



                    terrorist_graph.node[selected_node]['tempColor']=terrorist_graph.node[selected_node]['color']

                    terrorist_graph.node[selected_node]['Discovered']=True
                    terrorist_graph.node[selected_node]["MonitorNumber"]=terrorist_graph.node[selected_node]["MonitorNumber"]+1


                    for edge in G_changed.edges(selected_node):


                            if not terrorist_graph.has_node(edge[0]):

                                self.initialize_node(terrorist_graph,edge[0])


                            if not terrorist_graph.has_node(edge[1]):

                                self.initialize_node(terrorist_graph,edge[1])


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


    ## name of the file you want to store results in
    ##This method simulate the lying strategies and different monitor placement strategies to get results.

    def simulate_lying_monitor(self, results_file_name, nodes_lie):


        red_nodes=[node for node in self.G.nodes() if self.G.node[node]['color']=="Red"]


        for strategy_name in self.lying_strategies.keys():  #For different lying strategies considered


            prob_performance={}

            for prob in range(0,11,5):  #Change the range for required probabilities

                    print "Homophily removal probability " + str(prob*0.1)

                    performance={}
                    performance_reported_color={}

                    feature_importance_data={}
                    feature_importance_data2={}

                    for monitor_placement in self.next_monitor.keys():
                        performance[monitor_placement]={}
                        performance_reported_color[monitor_placement]={}

                        for b in self.budgets:
                            performance[monitor_placement][b]=[]
                            performance_reported_color[monitor_placement][b]=[]

                    for i in range(0, self.runs_honesty): # Number of runs with different honesty

                        print "Honesty run" + str(i)

                        G_modify = copy.deepcopy(self.G)

                        G_changed = self.remove_red_edges(G_modify,prob*0.1)  #Remove homophily

                        Lying_Strategies.assign_honesty(G_changed)


                        for j in range(0, self.runs_neighbor_color):  # Number of runs with different start nodes and neighbor colors
                            print "start node run" +str(j)

                            start_node = np.random.choice(red_nodes)
                            while len(list(G_changed.neighbors(start_node)))==0:  #select a node with neighbors as the start node
                                start_node=np.random.choice(red_nodes)

                            all_neighbor_colors = Lying_Strategies.assign_color_I_say(self.lying_strategies[strategy_name],G_changed)


                            Lying_Strategies.prob_lying_own_color=np.random.normal(0.5,0.125)


                            for next_monitor_placement in self.next_monitor.keys():          #keep honesty,neighbor colors same for all monitor placement strategies and evaluate them


                                        baseline_monitor.initialize()
                                        Learning_based_monitors.initialize()


                                        terrorist_graph=nx.Graph()

                                        selected_node=start_node

                                        self.initialize_node(terrorist_graph,selected_node)

                                        budget = self.max_budget
                                        no_monitors=0


                                        update=[]
                                        evaluation=0
                                        evaluation_reported_color=0

                                        found_red=[start_node]

                                        print next_monitor_placement
                                        while budget>0:


                                            no_monitors+=1

                                            terrorist_graph,update,evaluation,evaluation_reported_color = self.process_monitor(G_changed,terrorist_graph,all_neighbor_colors,selected_node,update,evaluation,evaluation_reported_color,nodes_lie)
                                            budget-=1


                                            if next_monitor_placement=="Learning":

                                                    leads=False


                                                    if next_monitor_placement=="Learning with lead":

                                                        leads=True


                                                    if no_monitors >= self.start_learning and no_monitors% self.training_it == 0: #learn the model after placing start learning number of monitors
                                                                                                                                # relearn the model every training_it number of monitors

                                                        selected_node,f_importance,f_imp2=Learning_based_monitors.learning_model(G_changed,terrorist_graph,update,True,nodes_lie,collective=False,with_leads=leads)

                                                        if no_monitors not in feature_importance_data.keys():
                                                            feature_importance_data[no_monitors]=[]
                                                            feature_importance_data2[no_monitors]=[]

                                                        feature_importance_data[no_monitors].append(f_importance[0])
                                                        feature_importance_data2[no_monitors].append(f_imp2)
                                                        update=[]

                                                    elif no_monitors < self.start_learning:  #at the begining place a small number of monitors randomly
                                                        selected_node=baseline_monitor.select_random_monitor(terrorist_graph,nodes_lie)

                                                    else:
                                                        selected_node,f_importance,f_imp2=Learning_based_monitors.learning_model(G_changed,terrorist_graph,update,False,nodes_lie,collective=False,with_leads=leads)


                                            elif next_monitor_placement=="Random monitor":
                                                selected_node=baseline_monitor.select_random_monitor(terrorist_graph,nodes_lie)

                                            else:
                                                selected_node = self.next_monitor[next_monitor_placement](terrorist_graph,selected_node,nodes_lie)

                                            if selected_node!=None:
                                                performance[next_monitor_placement][no_monitors].append(evaluation)
                                                performance_reported_color[next_monitor_placement][no_monitors].append(evaluation_reported_color)
                                                if G_changed.node[selected_node]["color"]=="Red":
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

                    self.save_results(performance,performance_reported_color,strategy_name+"_"+results_file_name+"_remove_edge_prob_"+str(prob))





    ##Modify this function to change the % of red edges removes from the network. Call this before inputting the graph to simulation function.

    def remove_red_edges(self, G_current,remove_prob):

        remove=[]
        ## When this prob is 0 we remove all the the edges. If this is 1, we keep all the red edges.
        for edge in G_current.edges():
            if G_current.node[edge[0]]['color']=="Red" and G_current.node[edge[1]]['color']=="Red":
                rand=np.random.random()
                if rand<remove_prob:
                    remove.append(edge)
        G_current.remove_edges_from(remove)
        return G_current



if __name__ == "__main__":



    ###################################################################################
    ########################RUN NOORDIN NETWORKS#######################################
    ###################################################################################

    # # get noordin data
    # noordin = NetworkData.NoordinData()
    #
    # noordin_data=["Train_Noordin_ExtComms_Computer-based.gexf","Train_Noordin_ExtComms_Videos.gexf","Train_Noordin_ExtComms_Support-materials.gexf","Train_Noordin_ExtComms_Unknown-commo.gexf","Train_Noordin_ExtComms_Print-media.gexf"]
    #
    # # using one noordin network
    # noordin_data = ["Train_Noordin_ExtComms_Unknown-commo.gexf"]
    # # run simulations for all noordin data
    # for network in noordin_data:
    #
    #     G = noordin.get_network_data("data//Noordin//"+network)  ##Graph you are selecting to input. Change the path according to your input
    #
    #     Learning_based_monitors.triangles=Traingles.get_all_triangles(G)  # pre calculate triangles for learning algorithm
    #
    #     nodes_lie = False  # define which problem setting ( setting 1: Nodes don't lie about own color, setting 2: Nodes lie about own color)
    #
    #     lying_monitor = LyingMonitorMain(G, start_learning=5, training_it=1) # For small networks set training_it to each iteration
    #
    #     lying_monitor.simulate_lying_monitor(network+"_results",nodes_lie) # Input the graph and file name to store results
    #

    ###################################################################################
    ########################RUN POKEC NETWORKS#########################################
    ###################################################################################

    # pokeC = NetworkData.PokeCData()
    #
    # pokeC_path=["Pokec_kosicky_age"]#,"Pokec_kosicky_height"]
    #
    # # run all pokec networks.
    #
    # for network in pokeC_path:
    #
    #     G = pokeC.get_network_data("data//pokec_sampled//"+network)
    #
    #     Learning_based_monitors.triangles = Traingles.get_all_triangles(G)
    #
    #     nodes_lie = False
    #
    #     lying_monitor = LyingMonitorMain(G, start_learning=20, training_it=20) # For large networks set training_it to 20
    #
    #     lying_monitor.simulate_lying_monitor(network+"_results",nodes_lie)



    ###################################################################################
    ########################RUN FACEBOOK100 NETWORKS###################################
    ###################################################################################

    network = "Colgate88" # also can select Amherst41
    attribute="dorm" #select the facebook100 attribute, you can also use "year"
    order = 1 # select the attribute value with highest # of nodes
    facebookData = NetworkData.Facebook100Data(attribute, order)

    G = facebookData.get_network_data("data//facebook100//"+network)

    nodes_lie=False  # define which problem setting ( setting 1: Nodes don't lie about own color, setting 2: Nodes lie about own color)

    Learning_based_monitors.triangles=Traingles.get_all_triangles(G)

    lying_monitor = LyingMonitorMain(G, start_learning=20, training_it=20) # For small networks set training_it to each iteration

    lying_monitor.simulate_lying_monitor(network+"_results",nodes_lie) # Input the graph and file name to store results





