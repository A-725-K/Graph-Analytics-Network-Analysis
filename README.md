# Graph-Analytics-Network-Analysis
A complete study on characteristics, robustness and contagion behavior on the American Tv-Show Facebook pages graph

## Getting started
To run this project is sufficient to clone or download this repository, with the command:
```
git clone https://github.com/A-725-K/Graph-Analytics-Network-Analysis.git
```
There are some external dependencies to satisfy, so you can run:
```
pip install -r requirements.txt
```

## How the repo works
First of all you have to set up the environment, so I provide You a *bash* script to ease the job:
```
cd bash_utils
./configure
```
In the directory *lib* there is the code on which rely all the experiments, while in *bash_utils* there are some utilities to automatize some scut work.

There are three different programs you can run, each one explore a different aspect of the network.

<ol>
  <li> <b>Expolore your Graph</b>: extract some <i>values, metrics</i> and analyze <i>communities</i> in the network
  <li> <b>Network Robustness</b>: try to mine the <i>connectivity</i> of the graph with some target attacks
  <li> <b>Social Contagion</b>: a study on how a <i>contagion</i> spreads inside a social network
</ol>

## How to launch the programs
You can simply run:
```
python3 lab1.py [--interactive]
python3 lab2.py [--small]
python3 lab3.py [--small]
```
The arguments between [ ] are optional.

## Authors

* **<i>Andrea Canepa</i>** - Computer Science, UNIGE - *Graph Analytics a.y. 2018/2019*
