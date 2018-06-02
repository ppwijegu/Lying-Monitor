# Lying-Monitor

1) Create a folder called "results" to store results of the algorithms
2) Run the code: run Lying_monitor_main.py
3) Change simulation parameters in the init method of Lying_monitor_main class
	a) add new lying strategies 
	b) add new monitor placement algorithms
	c) change monitor budget
	d) how many runs of the algorithm
4) All datasets we use are in the data folder
	Take a look at NetworkData script to see how we assign red nodes and node centrality values
	You can see an example of how to get the social networks in Lying_monitor_main script main 		method
5) Requirements
	Python 2.7, NetworkX, numpy, sklearn, matplotlib
	
	
	
#Example

If you  want add a new lying strategy following are the changes you need to make

1) Implement the new lying strategy in Lying_Strategies.py script
	Your method need to return the colors of the neighbors of a given node ( see lying_strategy_sample method for an example)

2) Add the new lying strategy to Lying_monitor_main.py init method

3) Select datasets in Lying_monitor_main.py, main method

4) When you run the code results will be save to results folder
	a) The results file will have the average performance and standard deviation for # of runs .
	
	b) Results reported color column is only applicable to problem setting 2 ( when nodes lie about their own color)
	
	c) results looks like follows:
	Monitor_placement_avg || Monitor_placement_STD || Monitor_placement_reported || Monitor_placement_reported_STD for each alforithm
	