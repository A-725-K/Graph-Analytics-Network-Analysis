import matplotlib

from lib.common import *
from lib.graph_utils import *
from lib.general_utils import timeit, initialize_graph


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


@timeit
def main():
    G = initialize_graph()

    # create a new random graph for debug purpose
    # G = generate_random_graph(100, 0.2)

    print_graph_info(G)

    layout_thread = th.Thread(target=compute_layout, args=(G,))
    diameter_thread = th.Thread(target=compute_distances, args=(G,))
    
    # heavy computation done in parallel
    diameter_thread.start()
    layout_thread.start()

    # wait for all threads to terminate, due to plt sharing
    diameter_thread.join()
    layout_thread.join()

    # centrality measures and graph exploration
    draw_graph(G, 'big_graph')
    degree_distribution(G)
    assortativity_matrix(G)
    analyze_communities(G)
    compute_metrics(G, 'clustering', True)
    compute_metrics(G, 'hits', True)
    compute_metrics(G, 'pagerank', True)
    compute_metrics(G, 'betweenness', True)
    compute_metrics(G, 'closeness', True)


def interactive_main():
    G = initialize_graph()
    
    while True:
        user_input = input('{}Tell me which node you want to know [0-{}] (END to exit):{}\t'.format(BLUE, G.nnodes-1, RESET))
        if user_input == 'END':
            break
        
        try:
            choice = int(user_input)    
            print(PURPLE + 'Node {} is the show:'.format(choice) + YELLOW + '\t' + G.mapping[choice] + RESET)
        except Exception as _:
            continue
    

if __name__ == "__main__":
    args = sys.argv
    argc = len(args)
    
    if argc > 2 or argc == 0:
        exit(RED + 'Usage: python3 lab1.py [--interactive]' + RESET)
    
    elif argc == 2:
        if args[1] != '--interactive':
            exit(RED + 'Usage: python3 lab1.py [--interactive]' + RESET)

        interactive_main()
    
    else:
        matplotlib.use('Agg')   # to avoid concurrency problem with matplotlib,
                                # run non graphical thread in background instead
                                # (cannot show plots, only save images on file)
        main()