# tdGraphEmbed

This repository provides a reference implementation of *tdGraphEmbed* as described in the paper:<br>
> tdGraphEmbed: Temporal Dynamic Graph-Level Embedding.<br>
> Moran Beladev, Lior Rokach, Gilad Katz, Ido Guy, Kira Radinsky.<br>
> CIKM’20 – October 2020, Galway, Ireland. <br>
> [Link](http://www.kiraradinsky.com/files/Temporal_Dynamic_Graph_Embedding__CIKM.pdf?fbclid=IwAR30gmFRxA8jqjOppnL1kGhUpwXKMQ1aJ1hUBR4lGprSTeroEHl7eTtAT0w)

### Requirements
    python>=3.6
    networkx
    numpy
    pandas
    gensim
    node2vec
    matplotlib
    holoviews
    sklearn
    scipy

### Basic Usage

#### Example
To run *tdGraphEmbed* you can follow the main.py for full flow example.
```python
    df = pd.read_table(r"data/facebook/facebook-wall.txt", sep = '\t', header = None)
    df.columns = ['source', 'target', 'time']
    temporal_g = TemporalGraph(data = df, time_granularity = 'months')
    graphs = temporal_g.get_temporal_graphs(min_degree = 10)
    model = TdGraphEmbed(dataset_name = "facebook")
    documents = model.get_documents_from_graph(graphs)
    model.run_doc2vec(documents)
    graph_vectors = model.get_embeddings()
   
```

#### Input
The data should include - source node, target node, time of interaction, weight(optional).
	
	node1_id_int node2_id_int time_timestamp <weight_float, optional>

#### output
`model.get_embeddings()` - > numpy array of shape (number of time steps, graph vector dimension size)

	shape = [num_of_time_steps, dim_of_representation]

#### TdGraphEmbed.get_documents_from_graph

According to the method describing in our paper, each graph time step is converted to a list of sentences 
of type `[TaggedDocument(doc, [time])]`. 

<img src="https://i.ibb.co/ZfxYvtB/graph2doc.png" width="600">

You can control the graph to document building process by updating the parameters in the config file: 
- `p` and `q` parameters affect the traverse method in the graph as explained in node2vec.
- `walk_length`(L), each sentence in the document max length
- `num_walks` (gamma)- number of walks starting from each node,
 will affect the number of sentences in the document representing the graph. 

#### Training the model ####

We train our model described in the paper, using the following architecture:
<img src="https://i.ibb.co/Z8g3qt7/g2v.png" width="400"/>

We use doc2vec code in order to apply this architecture.
You can control the doc2vec training parameters by updating the parameters in the config file.


#### Generated data ####
To achieve structural changes in time to the graph, we generated data by changing the nodes’ communities in time.
We use the Lancichinetti-Fortunato-Radicchi (LFR) algorithm to generate the graph, and 
injected anomalies in time by changing the amount of nodes changing their communities. 
To use this data generator use:
```python
    temporal_LFR_anomalies(n, tau1, tau2, mu, timesteps, anomaly_times)
```

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
