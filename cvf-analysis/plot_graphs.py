import networkx as nx

from matplotlib import pyplot as plt

graph_names = [
    # "graph_1",
    # "graph_2",
    # "graph_3",
    # "graph_6",
    # "graph_6b",
    # "graph_7",
    # "graph_8",
    "graph_4",
    "graph_5",
]

planar = True
fontsize = 20

for gname in graph_names:
    G = nx.read_adjlist(f"graphs/{gname}.txt")
    fig = plt.figure(figsize=(10, 5), constrained_layout=True)
    if planar:
        pos = nx.planar_layout(G)
        nx.draw_networkx(G, pos=pos, node_color="black", font_color="white", ax=fig.add_subplot(), font_size=30, node_size=1200)
    else:
        nx.draw_networkx(G, node_color="black", font_color="white", ax=fig.add_subplot(), font_size=30, node_size=1200)

    fig.suptitle(f"Graph {gname.split("_")[1]}", fontsize=fontsize)
    fig.savefig(f"graph_images/{gname}.png")
    # plt.show()
