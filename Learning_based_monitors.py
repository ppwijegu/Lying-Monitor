__author__ = 'pivithuru'


from sklearn import linear_model
import numpy as np
from heapq import *
import networkx as nx
import copy
import Isoloation
import operator
from sklearn import linear_model
from  sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import time

predict_model=None
score_heap=[]
training_set={}
predict={}
triangles={}
feature_categories={"answer":["red says red","blue says red","red says blue","blue says blue", "red score","prob red"],"neighbor":["red neighbors","blue neighbors"],"second hop":["second blue","second red"],"tight knight":["red traingles"]}#,"repeated monitor":["No. monitors","red lied about","blue lied about"]}

features=[]



def train_learning_model(training_set):


    predict_model = linear_model.LogisticRegression(class_weight='balanced')


    x=[]
    y=[]

    for point in training_set:

        x.append(training_set[point][0].values())
        keys=training_set[point][0].keys()
        y.append(training_set[point][1])

    if 0 not in y:
        y.append(0)
        x.append([0 for i in x[0]])
    predict_model.fit(x,y)

    return predict_model


def feature_importance():
    x=[]
    y=[]

    for point in training_set:

        x.append(training_set[point][0].values())
        keys=training_set[point][0].keys()
        y.append(training_set[point][1])

    if 0 not in y:
        y.append(0)
        x.append([0 for i in x[0]])

    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(x, y)
    importance=fit.scores_


    return importance



def predict_next_monitor(predict_model,terrorist_graph,predict_set):
    global score_heap

    scores_dict={}


    for node in predict_set.keys():



        probs=predict_model.predict_proba(predict_set[node].values())

        red_prob=probs[0][1]

        if not scores_dict.has_key(-red_prob):

            scores_dict[-red_prob]=[]


        scores_dict[-red_prob].append(node)
        terrorist_graph.node[node]["RedConfidence"]=red_prob

    score_heap=scores_dict.items()
    heapify(score_heap)
    return score_heap


def select_monitor_learning(terrorist_graph,with_leads):

    all_monitors=True

    while all_monitors:
        nodes_with_max=heappop(score_heap)
        for node in nodes_with_max[1]:
            if terrorist_graph.node[node]["IsMonitor"]==False:
                all_monitors=False


    if -nodes_with_max[0]<0.5 and with_leads:

        next=find_good_leader(terrorist_graph)


    else:

        next=np.random.choice(nodes_with_max[1])


        while terrorist_graph.node[next]["IsMonitor"]:

            next=np.random.choice(nodes_with_max[1])
        nodes_with_max[1].remove(next)


        if len(nodes_with_max[1]) >0:
            heappush(score_heap,nodes_with_max)


    return next


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


def get_prob_red(global_parameters,node,terrorist_graph):
    if global_parameters!=None:
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
        if count>0:
            prob_red=sum_prob_red/count
        else:
            prob_red=0
        return prob_red

    else: return 0

def get_second_neighbors(terrorist_graph,node):

    second_neighbors=[]

    for n in terrorist_graph.neighbors(node):
        second_neighbors.extend(terrorist_graph.neighbors(n))

    return set(second_neighbors)


def calculate_features(terrorist_graph,update):
    global predict
    global training_set
    global_parameters=get_global_parameters(terrorist_graph)
    for node in set(update):


        no_red_neighbors=len([n for n in terrorist_graph.neighbors(node) if terrorist_graph.node[n]["color"]=="Red"])
        blue_neighbors=(len(list(terrorist_graph.neighbors(node)))-no_red_neighbors)


        red_says_red=len([n for n in terrorist_graph.node[node]['SaysRed'] if terrorist_graph.node[n]['color']=="Red"])


        blue_says_red=len([n for n in terrorist_graph.node[node]['SaysRed'] if terrorist_graph.node[n]['color']=="Blue"])


        blue_says_blue=len([n for n in terrorist_graph.node[node]['SaysBlue'] if terrorist_graph.node[n]['color']=="Blue"])


        red_says_blue=len([n for n in terrorist_graph.node[node]['SaysBlue'] if terrorist_graph.node[n]['color']=="Red"])

        red_neighbors_lied_about=len([n for n in terrorist_graph.neighbors(node) if (node in terrorist_graph.node[n]['SaysBlue'] and terrorist_graph.node[n]['color']=="Red") ])

        blue_neighbors_lied_about=len([n for n in terrorist_graph.neighbors(node) if (node in terrorist_graph.node[n]['SaysRed'] and terrorist_graph.node[n]['color']=="Blue") ])


        red_score=len(terrorist_graph.node[node]['SaysRed'])

        prob_red=get_prob_red(global_parameters,node,terrorist_graph)


        node_triangles=triangles[node]
        red_triangles=0


        for t in node_triangles:
            try:
                if terrorist_graph.node[t[0]]["color"]=="Red" and terrorist_graph.node[t[1]]["color"]=="Red":
                    red_triangles+=1
            except:
                red_triangles+=0

        second_neighbors=get_second_neighbors(terrorist_graph,node)

        second_red_neighbors=len([n for n in second_neighbors if terrorist_graph.node[n]["color"]=="Red"])
        second_blue_neighbors=len([n for n in second_neighbors if terrorist_graph.node[n]["color"]=="Blue"])



        node_feature_set={"red neighbors":no_red_neighbors,
                           "red says red":red_says_red,
                           "blue says red":blue_says_red,
                           "red says blue":red_says_blue,
                           "blue says blue":blue_says_blue,
                           "red score":red_score,
                           "blue neighbors":blue_neighbors,
                           "red traingles":red_triangles,
                           "prob red":prob_red,
                          "second blue":second_blue_neighbors,
                          "second red":second_red_neighbors}
        ####Add these features when nodes lie about their own color
                           # "No. monitors":terrorist_graph.node[node]["MonitorNumber"],
                           # "red lied about":red_neighbors_lied_about,
                           # "blue lied about":blue_neighbors_lied_about}


        if not terrorist_graph.node[node]['IsMonitor']:
            predict[node]=node_feature_set
        else:
            if node in predict:

                del predict[node]
            if terrorist_graph.node[node]['color']=="Red":
                training_set[node]=(node_feature_set,1.0)
            else:
                training_set[node]=(node_feature_set,0)

        global features
        features=node_feature_set.keys()

    for node in terrorist_graph:
        prob_red=get_prob_red(global_parameters,node,terrorist_graph)
        if node in predict:

            predict[node]["prob red"]=prob_red
        if node in training_set:
            training_set[node][0]["prob red"]=prob_red


def get_predict_when_nodes_lie(train,predict):

    for node in train:
        if train[node][1]==0:
            predict[node]=train[node][0]
    return predict

def select_next_monitor_when_nodes_lie(terrosrist_graph,threshold):

    selected=False

    while(not selected):
        next=select_monitor_learning()
        if terrosrist_graph.node[next]["MonitorNumber"]<threshold+1:
            selected=True
    return next

def find_good_leader(terrorist_graph):

    reg = linear_model.LinearRegression()

    x=[]
    y=[]



    for point in training_set:

        x.append(training_set[point][0].values())
        y.append(training_set[point][1])


    for node in training_set:
        red_neighbors=len([n for n in terrorist_graph.neighbors(node) if terrorist_graph.node[n]["color"]=="Red"])
        blue_neighbors=len([n for n in terrorist_graph.neighbors(node) if terrorist_graph.node[n]["color"]=="Blue"])

        if (red_neighbors+blue_neighbors)==0:
            prob_lead=0
        else:

            prob_lead=1.0*red_neighbors/(red_neighbors+blue_neighbors)

        x.append(training_set[node][0].values())

        y.append(prob_lead)


    reg.fit(x,y)
    max_lead_val=0
    max_leads=[]
    for node in predict:
        if not terrorist_graph.node[node]["IsMonitor"]:
            potential_lead=reg.predict(predict[node].values())
            reg_val=potential_lead[0]

            if reg_val>=max_lead_val:
                if potential_lead>max_lead_val:
                    max_leads=[]
                    max_lead_val=reg_val
                max_leads.append(node)
    #print max_lead_val

    if len(max_leads)==0:
        max_leads=[n for n in terrorist_graph if not terrorist_graph.node[n]["IsMonitor"]]
    return np.random.choice(max_leads)

def find_good_leader_2(terrorist_graph):


    max_lead_val=0
    max_leads=[]
    for node in predict:
        potential_lead=get_potential_lead_score(terrorist_graph,node)

        if potential_lead>=max_lead_val:
            if potential_lead>max_lead_val:
                max_leads=[]
                max_lead_val=potential_lead
            max_leads.append(node)
    #print max_lead_val

    if len(max_leads)==0:
        max_leads=[n for n in terrorist_graph if not terrorist_graph.node[n]["IsMonitor"]]
    return np.random.choice(max_leads)

def get_potential_lead_score(terrorist_graph,node):

    monitored_neighbors=[n for n in terrorist_graph.neighbors(node) if terrorist_graph.node[n]["IsMonitor"]]
    cum_prob=1
    for neighbor in monitored_neighbors:
        red_neighbors=len([n for n in terrorist_graph.neighbors(neighbor) if terrorist_graph.node[n]["color"]=="Red"])
        blue_neighbors=len([n for n in terrorist_graph.neighbors(neighbor) if terrorist_graph.node[n]["color"]=="Blue"])

        prob_lead=1.0*red_neighbors/(red_neighbors+blue_neighbors)

        cum_prob*(1-prob_lead)

    return (1-cum_prob)



def calculate_prediction_accuracy(terrorist_graph,threshold):
    blue=0
    red=0
    for node in terrorist_graph.nodes():
        if terrorist_graph.node[node]["IsMonitor"]:
            if terrorist_graph.node[node]["color"]=="Blue":
                blue+=1
            else:
                red+=1

    accuracy=1.0*red/(red+blue)



def collective_features(terrorist_graph,global_parameters,current_node):




    no_red_neighbors=len([n for n in terrorist_graph.neighbors(current_node) if terrorist_graph.node[n]["tempColor"]=="Red"])
    blue_neighbors=(len(terrorist_graph.neighbors(current_node))-no_red_neighbors)


    red_says_red=len([n for n in terrorist_graph.node[current_node]['SaysRed'] if terrorist_graph.node[n]['tempColor']=="Red"])


    blue_says_red=len([n for n in terrorist_graph.node[current_node]['SaysRed'] if terrorist_graph.node[n]['tempColor']=="Blue"])


    blue_says_blue=len([n for n in terrorist_graph.node[current_node]['SaysBlue'] if terrorist_graph.node[n]['tempColor']=="Blue"])


    red_says_blue=len([n for n in terrorist_graph.node[current_node]['SaysBlue'] if terrorist_graph.node[n]['tempColor']=="Red"])

    red_neighbors_lied_about=len([n for n in terrorist_graph.neighbors(current_node) if (current_node in terrorist_graph.node[n]['SaysBlue'] and terrorist_graph.node[n]['tempColor']=="Red") ])

    blue_neighbors_lied_about=len([n for n in terrorist_graph.neighbors(current_node) if (current_node in terrorist_graph.node[n]['SaysRed'] and terrorist_graph.node[n]['tempColor']=="Blue") ])


    red_score=len(terrorist_graph.node[current_node]['SaysRed'])



    prob_red=get_prob_red(global_parameters,current_node,terrorist_graph)

    second_neighbors=get_second_neighbors(terrorist_graph,current_node)

    second_red_neighbors=len([n for n in second_neighbors if terrorist_graph.node[n]["tempColor"]=="Red"])
    second_blue_neighbors=len([n for n in second_neighbors if terrorist_graph.node[n]["tempColor"]=="Blue"])


    node_triangles=triangles[current_node]
    red_triangles=0


    for t in node_triangles:
        #if terrorist_graph.has_node(t[0]) and terrorist_graph.has_node(t[1]):
        try:
            if terrorist_graph.node[t[0]]["tempColor"]=="Red" and terrorist_graph.node[t[1]]["tempColor"]=="Red":
                red_triangles+=1
        except:
            red_triangles+=0



    node_feature_set={"red neighbors":no_red_neighbors,
                       "red says red":red_says_red,
                       "blue says red":blue_says_red,
                       "red says blue":red_says_blue,
                       "blue says blue":blue_says_blue,
                       "red score":red_score,
                       "blue neighbors":blue_neighbors,
                       "red traingles":red_triangles,
                       "prob red":prob_red,
                       # "No. monitors":terrorist_graph.node[current_node]["MonitorNumber"],
                       # "red lied about":red_neighbors_lied_about,
                       # "blue lied about":blue_neighbors_lied_about,
                       "second blue":second_blue_neighbors,
                        "second red":second_red_neighbors}

    return node_feature_set

def collective_classification(terrorist_graph,predict_model):

    global_param=get_global_parameters(terrorist_graph)


    temp_prob={}

    label_changed=True
    i=0
    no_iterations=100


    for next in terrorist_graph.nodes():
        if not terrorist_graph.node[next]["IsMonitor"]:

            node_features=collective_features(terrorist_graph,global_param,next)


            probs=predict_model.predict_proba(node_features.values())



            prob_red=probs[0][1]



            temp_prob[next]=prob_red
            terrorist_graph.node[next]['probRed']=prob_red
            if prob_red>0.5:
                terrorist_graph.node[next]['tempColor']="Red"
            else:
                terrorist_graph.node[next]['tempColor']="Blue"


    sorted_prob= sorted(temp_prob.items(), key=operator.itemgetter(1),reverse=True)
    while (label_changed and i<no_iterations):
            label_changed=False
            i+=1



            for node_pair in sorted_prob:

                if terrorist_graph.node[node_pair[0]]["IsMonitor"]==False:

                    node_features=collective_features(terrorist_graph,global_param,node_pair[0])


                    probs=predict_model.predict_proba(node_features.values())


                    prob_red=probs[0][1]


                    temp_prob[node_pair[0]]=prob_red



                    terrorist_graph.node[node_pair[0]]['probRed']=prob_red

                    if prob_red>0.5:

                        last_label=terrorist_graph.node[node_pair[0]]['tempColor']

                        if last_label!="Red":

                            label_changed=True

                        terrorist_graph.node[node_pair[0]]['tempColor']="Red"
                    else:

                        last_label=terrorist_graph.node[node_pair[0]]['tempColor']
                        if last_label!="Blue":
                            label_changed=True


                        terrorist_graph.node[node_pair[0]]['tempColor']="Blue"



            sorted_prob= sorted(temp_prob.items(), key=operator.itemgetter(1),reverse=True)

    max_prob=0

    nodes_max_prob=[]
    for node in terrorist_graph:
        if not terrorist_graph.node[node]["IsMonitor"]:
            if terrorist_graph.node[node]["probRed"]>=max_prob:
                if terrorist_graph.node[node]["probRed"]>max_prob:
                    nodes_max_prob=[]

                nodes_max_prob.append(node)


                max_prob=terrorist_graph.node[node]["probRed"]
    next_monitor=np.random.choice(nodes_max_prob)

    return next_monitor


def learning_model(G,terrorist_graph,update,train_again,nodes_lie=False,collective=False,with_leads=False):
    global score_heap
    global predict_model
    global training_set,predict

    if nodes_lie and train_again:

        threshold=max_no_monitors_get_true_color(terrorist_graph)

        if threshold<3:
            threshold=3


        calculate_features(terrorist_graph,update)
        training_set,predict,terrorist_graph=Isoloation.remove_outliers(G,terrorist_graph,threshold,training_set,predict)

        predict_model=train_learning_model(training_set)

        score_heap=predict_next_monitor(predict_model,terrorist_graph,predict)

        next_monitor=select_monitor_learning(terrorist_graph,with_leads)


    elif train_again:

        if collective:
            calculate_features(terrorist_graph,update)
            predict_model=train_learning_model(training_set)
            next_monitor=collective_classification(terrorist_graph,predict_model)


        else:


            calculate_features(terrorist_graph,update)

            predict_model=train_learning_model(training_set)

            score_heap=predict_next_monitor(predict_model,terrorist_graph,predict)

            next_monitor=select_monitor_learning(terrorist_graph,with_leads)
    else:

        next_monitor=select_monitor_learning(terrorist_graph,with_leads)

    f_import2=feature_importance()



    return next_monitor,predict_model.coef_,f_import2


def max_no_monitors_get_true_color(terrorist_graph):
    red_nodes=[n for n in terrorist_graph.nodes() if terrorist_graph.node[n]['color']=="Red"]
    red_monitors=[]

    for node in red_nodes:

        no_monitors=terrorist_graph.node[node]["MonitorNumber"]

        red_monitors.append(no_monitors)

    return max(red_monitors)


def initialize():
    global predict_model
    global score_heap
    global training_set
    global predict

    predict_model=None
    score_heap=[]
    training_set={}
    predict={}




