from .common import *
from .graph_utils import print_statistics


def compute_triangles(G):
    trg = nx.triangles(G)
    return sum(trg.values()) // 3 # because each vertex is counted in each triangle in which appears
    

def print_graph_info(G):
    # name
    print(PURPLE + '/' + '*'*(len(G.name)+2) + '\\')
    print(PURPLE + '|', YELLOW + G.name, PURPLE + '|')
    print(PURPLE + '\\' + '*'*(len(G.name)+2) + '/\n')

    # graph info
    print(RED + '  > ' + BLUE + 'Nodes:' + WHITE, G.nnodes)
    print(RED + '  > ' + BLUE + 'Edges:' + WHITE, G.nedges)
    print(RED + '  > ' + BLUE + 'Type of edges:' + WHITE, 'Directed' if nx.is_directed(G) else 'Undirected')
    print(RED + '  > ' + BLUE + 'Average degree:\n\t<k> = ' + WHITE + '{:.3f}'.format(G.avg_degree))
    print(RED + '  > ' + BLUE + 'Average clustering coefficient:\n\tC = ' + WHITE + '{:.3f}'.format(nx.average_clustering(G)))
    print(RED + '  > ' + BLUE + 'Density:\n\trho = ' + WHITE + '{:.3f}'.format(nx.density(G)))
    print(RED + '  > ' + BLUE + 'Number of triangles:' + WHITE, compute_triangles(G))
    print(RED + '  > ' + BLUE + 'Connectivity:' + WHITE, 'Connected' if nx.is_connected(G) else 'Disconnected')
    print(RED + '  > ' + BLUE + 'Assortativity:\n\tr = ' + WHITE + '{:.3f}'.format(nx.degree_assortativity_coefficient(G)))
    print(RED + '  > ' + BLUE + 'Giant component coverage: ' + WHITE + '{:.2f}%\n'.format(max([len(cc) for cc in nx.connected_components(G)])/G.nnodes*100) + RESET)


def compute_historgram(degs_dict):
    normalized_histogram = {}

    # count frequencies
    for deg in sorted([deg for _, deg in degs_dict]):
        if deg in normalized_histogram:
            normalized_histogram[deg] += 1
        else:
            normalized_histogram[deg] = 1
    
    # normalize the histogram
    for k, v in normalized_histogram.items():
        normalized_histogram[k] = v / len(degs_dict)

    return normalized_histogram


def degree_distribution(G):
    print_statistics(sorted(dict(G.degree).items(), key=lambda p: p[1]), 'Degree distribution', G.mapping)
    
    hist = compute_historgram(G.degree)

    x = [int(k) for k in hist.keys()]
    y = [float(v) for v in hist.values()]

    # real power law
    max_y = max(y)
    pow_law = []
    for i in range(1, max(x)):
        pow_law += [i**(-1) * max_y]

    plt.figure()
    plt.title('Degree Distribution and Power Law')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.bar(x, y, width=0.7, color='#FF6F00')
    plt.plot(pow_law, 'blue', linewidth=2.5)
    plt.savefig(IMG_DIR + 'deg_distribution' + EXT)


def assortativity_matrix(G):
    ass_matrix = nx.degree_mixing_matrix(G)

    fig = plt.figure()
    plt.title('Degree Mixing Analysis')
    plt.tight_layout()
    ax = plt.imshow(ass_matrix, cmap='coolwarm', interpolation='nearest', origin='lower')
    fig.colorbar(ax, label='\nDegree Correlation Matrix')
    plt.savefig(IMG_DIR + 'deg_mix' + EXT)
    