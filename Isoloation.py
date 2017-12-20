__author__ = 'isira'
print(__doc__)
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import sklearn.ensemble as en
from sklearn.metrics import accuracy_score
import sklearn.metrics
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sklearn.neighbors as neigh

def DBScan(G,training_data):

    X_Blue={}
    X_Red={}
    keys=[]
    train=[]

    for point in training_data:
        #if training_data[point][1]==0:
            X_Blue[point]=training_data[point][0].values()
            train.append(training_data[point][0].values())
            keys.append(point)


        # else:
        #     X_Red[point]=training_data[point][0].values()
        #     train.append(training_data[point][0].values())
        #     keys.append(point)




    db = DBSCAN(eps=0.3, min_samples=10).fit(train)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    print core_samples_mask
    core_samples_mask[db.core_sample_indices_] = True
    labels=db.labels_


    for i in range(len(keys)):
        node=keys[i]
        print (i,labels[i],G.node[node]["color"],training_data[node][1])
    labels_true=[G.node[n]["color"] for n in keys]
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    # print("Silhouette Coefficient: %0.3f"
    #       % metrics.silhouette_score(training_data, labels))



def remove_outliers(G,terrorist_graph,threshold,train_set,predict_set):


    rng = np.random.RandomState(42)
    # fit the model
    # X_Red=[X_train[i] for i in range(len(X_train)) if Y_train[i]==1 ]
    X_Blue=[]
    X_Blue={}
    X_Red={}
    keys=[]
    for point in train_set:
        if train_set[point][1]==0:
            X_Blue[point]=train_set[point][0].values()
            keys=train_set[point][0].keys()
        else:
            X_Red[point]=train_set[point][0].values()


    blue_train=[X_Blue[i] for i in X_Blue.keys()]

    blue_keys=X_Blue.keys()

    if len(blue_train)>0:
        #clf=neigh.LocalOutlierFactor()

        clf = en.IsolationForest(max_samples=100, random_state=rng)
        clf.fit(blue_train)
        y_pred_train = clf.predict(blue_train)

        no_red_among_blue=[n for n in blue_keys if G.node[n]['color']=="Red" ]
        no_identified=0
        no_removed=0

        for j in range(len(y_pred_train)):

            if y_pred_train[j]!=1:

                if blue_keys[j]  in no_red_among_blue:
                    no_identified+=1
                no_monitors=terrorist_graph.node[blue_keys[j]]["MonitorNumber"]
                # print no_monitors
                # prob_red=pow(prob_lying,no_monitors)
                # print prob_red
                #if terrorist_graph.node[blue_keys[j]]["MonitorNumber"]<4: #or terrorist_graph.node[blue_keys[j]]["RedConfidence"]>0.1 :

                if no_monitors<threshold+1:
                    #print no_monitors

                    predict_set[blue_keys[j]]=train_set[blue_keys[j]][0]
                    no_removed+=1
                    #predict_set.append()
                    del train_set[blue_keys[j]]
                    terrorist_graph.node[blue_keys[j]]["color"]="Black"
                    terrorist_graph.node[blue_keys[j]]['tempColor']="Black"
                    terrorist_graph.node[blue_keys[j]]["IsMonitor"]=False

        #print (no_identified,len(no_red_among_blue),no_removed)



    return train_set,predict_set,terrorist_graph



