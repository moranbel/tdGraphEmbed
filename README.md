# tdGraphEmbed
=======

=======

[Link](http://www.kiraradinsky.com/files/Temporal_Dynamic_Graph_Embedding__CIKM.pdf?fbclid=IwAR30gmFRxA8jqjOppnL1kGhUpwXKMQ1aJ1hUBR4lGprSTeroEHl7eTtAT0w
) to the paper describing our method (published in CIKM'20).
Temporal dynamic graphs are graphs whose topology evolves over time, 
which is important in numerous fields - the World Wide Web evolution, 
social and communication networks, scientific citations, terrorist analysis, 
biological graphs, etc.  We share an approach to represent such graphs for 
anomaly detection, trends analysis, graph classification and more.


### tdGraphEmbed- main ###

Creating temporal embedding for each graph time step. 
To run the embedding, use the main file and update the path file to your graph data. 
The data should include - source node, target node, time of interaction, weight(optional).
`model.get_embeddings()` - > numpy array of shape (number of time steps, graph vector dimension size)

### TdGraphEmbed.get_documents_from_graph ###

According to the method describing in our paper, each graph time step is converted to a list of sentences 
of type `[TaggedDocument(doc, [time])]`. 

<img src="https://i.ibb.co/ZfxYvtB/graph2doc.png" style="max-width: 600px"/>

You can control the graph to document building process by updating the parameters in the config file: 
- `p` and `q` parameters affect the traverse method in the graph as explained in node2vec.
- `walk_length`(L), each sentence in the document max length
- `num_walks` (gamma)- number of walks starting from each node,
 will affect the number of sentences in the document representing the graph. 

### Training the model ###

We train our model described in the paper, using the following architecture:
<img src="https://i.ibb.co/Z8g3qt7/g2v.png" style="max-width: 600px"/>

We use doc2vec code in order to apply this architecture.
You can control the doc2vec training parameters by updating the parameters in the config file.


### Generated data ###
To achieve structural changes in time to the graph, which can be caused by nodes’ changing of communities, we generated data for this use case.
We created a synthetic dataset that illustrates the effectiveness of our proposed graph embedding approach for anomaly detection.

The first graph in the sequence of times is generated using the Lancichinetti-Fortunato-Radicchi (LFR) algorithm,
which generates benchmark networks, i.e., artificial networks that resemble real-world networks.

We initiate the graph with two communities and in each time step, 
we randomly choose one of the two communities and sample m nodes (using normal distribution) 
from this community. We then change the labels of these nodes and assign them to the other community (referred to as "community shifting"). 
We repeat this process for 100 iterations (i.e., time steps).

We inject anomalies in time by changing the normal distribution parameters, 
such that in abnormal time steps more nodes are likely to change labels. 
The following figure presents a visualization of the graph during its first 20 time steps. 
The color of each nodes indicates the node's community, where the position of each node is determined using 2-Laplacian Eigenmaps. 
The nodes selected for a community shift appear in red. The anomaly we injected at t=13 is clearly visible in the graph.

<img src="https://i.ibb.co/7k3bhsr/communities.jpg" style="max-width: 600px"/>


### Data ###
All our data is accessible in the "data" folder. 
The `_dynamic` suffix stands for dynamic graphs, having different number of nodes per time step.

The  `_static` suffix stands for static graph, having same number of nodes per time step. 
To achieve that we created all nodes in each time steps, nodes that do not exist at that time step are isolated.


### Citing ###
If you find tdGraphEmbed useful for your research, please consider citing the following paper:

tdGraphEmbed: Temporal Dynamic Graph-Level Embedding
Moran Beladev, Lior Rokach, Gilad Katz, Ido Guy, Kira Radinsky
CIKM’20 – October 2020, Galway, Ireland

For questions, please contact me at `moranbeladev90@gmail.com`.
