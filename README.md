# PLSBM (Partially Labeled Stochastic Blockmodel)

The implementation of the proposed method in our paper "When Does Label Propagation Fail? A View from a Network Generative Model", which will be published upcoming IJCAI 2017.

## Usage

You'll need to prepare the input graph file and the input label file (file format shown below).
No parameter is needed in the algorithm.

```
$ python main.py -h
usage: main.py [-h] -g GRAPHFILE -l LABELFILE -o OUTFILE -m METHOD [-v]
               [-e EPS]

optional arguments:
  -h, --help            show this help message and exit
  -g GRAPHFILE, --graphfile GRAPHFILE
                        input graph file
  -l LABELFILE, --labelfile LABELFILE
                        input label file
  -o OUTFILE, --outfile OUTFILE
                        output file
  -m METHOD, --method METHOD
                        plsbm or sbm
  -v, --verbose         verbosity
  -e EPS, --eps EPS     threshold of residual error for convergence
```

## Input grpah file

The input graph file is the edge list separated by space.

For each line the first number is the id of the source node, and the second number of the id of the destination node.

Node ids are zero-based.

```
$ cat graph.dat
0 1
1 2
1 3
2 4
2 6
3 5
4 5
```
## Input label file

The input label file is the list of the label id.

Label ids are zero-based.

The i-th line contains the label id of the node i.

```
$ cat label.dat
0
1
1
2
0
```

In this case, for example, node 3 has label 2.


## Reference
Yuto Yamaguchi, Kohei Hayashi, "When Does Label Propagation Fail? A View from a Network Generative Model", International Joint Conference on Artificial Intelligence (IJCAI), Melbourne, Australia, Aug 19-25, 2017.
