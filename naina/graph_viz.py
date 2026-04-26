import networkx as nx
import matplotlib.pyplot as plt

import pickle

with open("outputs/graph.gpickle", "rb") as f:
    G = pickle.load(f)# color nodes
colors = []
for n in G.nodes():
    label = G.nodes[n].get('label', -1)
    if label == 1:
        colors.append('red')     # fraud
    elif label == 0:
        colors.append('blue')    # legit
    else:
        colors.append('gray')    # unknown

# take small subgraph
nodes = list(G.nodes())[:200]
subG = G.subgraph(nodes)

sub_colors = [colors[list(G.nodes()).index(n)] for n in nodes]

plt.figure(figsize=(8,6))
nx.draw(subG, node_color=sub_colors, node_size=30, with_labels=False)
plt.title("Fraud Graph Visualization")
plt.show()