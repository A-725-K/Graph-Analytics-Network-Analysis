from .common import *
from .general_utils import timeit, print_title, print_statistics


SAMPLES = 25
LAYOUT  = None


def analyze_communities(G):
    global LAYOUT

    # Clauset-Newman-Moore algorithm to detect communities
    communities = nx.algorithms.community.greedy_modularity_communities(G)

    print_title('Communities')
    print(RED + '\t+++')
    print(RED + '\t |- ' + YELLOW + 'Number of Communities:' + WHITE, len(communities))
    print(RED + '\t |- ' + YELLOW + 'Performance: p =' + WHITE, nx.algorithms.community.quality.performance(G, communities))
    print(RED + '\t |- ' + YELLOW + 'Modularity: Q =' + WHITE, nx.algorithms.community.quality.modularity(G, communities))
    print(RED + '\t+++' + RESET)

    comm_colors = []
    for n in G.nodes():
        for idx, comm in enumerate(communities):
            if n in comm:
                comm_colors += [idx]
                break
    max_col = len(np.unique(comm_colors))

    plt.figure()
    plt.title('Communities')
    nx.draw_networkx(G, pos=LAYOUT, node_color=comm_colors, cmap='rainbow', vmin = 1, vmax=max_col, node_size=10, width=0.1, with_labels=False)
    plt.tight_layout()
    plt.savefig(IMG_DIR + 'communities' + EXT)
    

def plot_metrics(measure, label, color, samples=SAMPLES, xlabel='nodes'):
    measure = measure[-samples:][::-1]
    [xs, ys] = zip(*measure)
    
    plt.figure()
    plt.title(label.title())
    plt.xlabel(xlabel)
    plt.ylabel(label)

    ind = range(samples)
    plt.bar(ind, ys, color=color, alpha=0.8, width=0.9)
    if label != 'clustering':
        plt.xticks(ind, xs, rotation=90)

    plt.tight_layout()
    plt.savefig(IMG_DIR + label + EXT)


def plot_hits(hubs, authorities, label, color, samples=SAMPLES, threshold=1e-3):
    cc = color.split(' ')
    ll = label.split(' ')
    color_h = cc[0]
    color_a = cc[1]
    label_h = ll[0]
    label_a = ll[1]
    
    # separate hubs from authorities
    [xs_h, ys_h] = zip(*(sorted(hubs.items(), key=lambda p: p[1])[-samples:]))
    [xs_a, ys_a] = zip(*(sorted(authorities.items(), key=lambda p: p[1])[-samples:]))

    # filter values too small
    ys_h = list(filter(lambda x: x > threshold, ys_h))
    ys_a = list(filter(lambda x: x > threshold, ys_a))

    # choose one cardinality to show
    l = min(len(ys_a), len(ys_h))
    if len(ys_a) != len(ys_h):
        ys_a = ys_a[-l:]
        ys_h = ys_h[-l:]

    ind = range(l)

    # create labels
    x_lab = []
    for i in ind:
        x_lab += ['{}\n{}'.format(xs_h[i], xs_a[i])]
    
    plt.figure()
    _, ax = plt.subplots()
    width = 0.35
    
    ax.bar(ind, ys_h, width=width, label=label_h, color=color_h)
    ax.bar([i + width for i in ind], ys_a, width=width, label=label_a, color=color_a)

    ax.set_title('Hits')
    ax.set_xlabel('nodes')
    ax.legend()
    ax.autoscale_view()
    plt.xticks(ind, x_lab, rotation=90, fontsize='xx-small')
    plt.savefig(IMG_DIR + 'hits' + EXT)


def compute_metrics(G, metrics, plot=False):
    m = {}
    # color = ''

    if metrics == 'betweenness':
        m = nx.betweenness_centrality(G)
        # color = 'black' 
    elif metrics == 'closeness':
        m = nx.closeness_centrality(G)
        # color = 'red'
    elif metrics == 'pagerank':
        m = nx.pagerank(G, alpha=0.8, max_iter=1500, tol=1e-03)
        # color = 'green'
    elif metrics == 'clustering':
        m = nx.clustering(G)
        # color = 'purple'
    elif metrics == 'hits':
        (hubs, authorities) = nx.hits(G, max_iter=5000, tol=1e-02, normalized=True)
        # color = 'blue orange'
        if plot:
            h_s = sorted(hubs.items(), key=lambda p: p[1])
            a_s = sorted(authorities.items(), key=lambda p: p[1])

            print_statistics(h_s, 'Hubs', G.mapping)
            print_statistics(a_s, 'Authorities', G.mapping)
            plot_hits(hubs, authorities, 'Hubs Authorities', METRIC_COLOR[metrics], samples=20)
        
        return (hubs, authorities)
    else:
        print(RED + 'ERROR: Metric "{}" does not implemented !'.format(metrics) + RESET)
        exit(1)
    
    if plot:
        m = sorted(m.items(), key=lambda p: p[1])
        print_statistics(m, metrics.title(), G.mapping)
        if metrics == 'clustering':
            plot_metrics(m, metrics, METRIC_COLOR[metrics], samples=len(m))
        else:
            plot_metrics(m, metrics, METRIC_COLOR[metrics])
    
    return m


#@timeit
def compute_distances(G, show=True):
    if show:
        print(GREEN + 'Computing diameter of the graph...')
        print(GREEN + 'Computing average shortest path length of the graph...' + RESET)

    if not nx.is_connected(G):
        print(RED + '[!!] Graph not connected, considering its giant component...' + RESET)
        ccs = [G.subgraph(cc) for cc in nx.connected_components(G)]
        ccs_sz = [len(cc) for cc in ccs]
        max_idx = ccs_sz.index(max(ccs_sz))
        G = ccs[max_idx]

    d = nx.diameter(G)
    avg_sp = nx.average_shortest_path_length(G)
    
    if show:
        print(RED + '  > ' + BLUE + 'Diameter:\n\td = ' + WHITE + str(d))
        print(RED + '  > ' + BLUE + 'Average shortest path length: ' + WHITE + '{:.3f}'.format(avg_sp) + RESET)

    return d, avg_sp


@timeit
def compute_layout(G):
    global LAYOUT
    print(GREEN + 'Computing graph layout...' + RESET)
    LAYOUT = nx.kamada_kawai_layout(G)


def draw_graph(G, filename, name=GRAPH_NAME, node_col='blue', xx=0.69, verbose=True):
    global LAYOUT

    if verbose:
        print(YELLOW + '++++++++++++++++++++++++ Drawing ' + filename + '...' + RESET)
    
    plt.figure()
    nx.draw(G, pos=LAYOUT, node_size=10, width=0.3, node_color=node_col)
    plt.suptitle(name, fontsize=15, color='#116B17', x=xx, y=0.05)
    plt.savefig(filename + EXT)
    plt.close()


def generate_random_graph(N, p):
    G = nx.gnp_random_graph(N, p)
    G.mapping = {}
    G.name = 'Random Graph, N = {}, p = {}'.format(N, p)
    G.nnodes = len(G.nodes())
    G.nedges = len(G.edges())
    G.avg_degree = 2*G.nedges / G.nnodes if not nx.is_directed(G) else G.nedges / G.nnodes
    for i in range(len(G.nodes())):
        G.mapping[i] = i

    return G


def generate_connected_random_graph(n, p):
    G = nx.Graph()
    fst = True
    while len(G) < 1 or not nx.is_connected(G):
        G = generate_random_graph(n, p)
        if not fst:
            print(RED + 'The graph is not connected... Trying again...' + RESET)
        fst = False
    print(PURPLE + '\n\tN = {}\t\tlog(N) = {:.3f}\t\tlog(N)/N = {:.3f}\tp = {:.2f}\n'.format(n, np.log(n), np.log(n)/n, p) + RESET)

    return G
