import time
import threading as th

import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


SAMPLES = 25
EXT = '.png'
DATASET_FILE = 'datasets/fb-pages-tvshow.edges'
MAPPING_FILE = 'datasets/fb-pages-tvshow.nodes'
GRAPH_NAME = 'American TV Shows Facebook pages'


def get_show(mapping, key):
    print('Node {} is the show: {}'.format(key, mapping[key]))


def format_time(t):
    ss = t % 60
    mm = int((t // 60) % 60)
    hh = int(t // 3600)
    return '{} h {} m {:.3f} s'.format(hh, mm, ss)


def timeit(f):
    def timed_foo(*args, **kw):
        start = time.time()
        ret_val = f(*args, **kw)
        end = time.time()
        print('Function [{}]\telapsed time: {}'.format(f.__name__, format_time(end - start)))
        return ret_val

    return timed_foo


def print_statistics(values, measure, mapping):
    max = values[-1]
    min = values[0]
    lst = [i for _, i in values]

    print('\n' + '#'*(len(measure) + 8))
    print('#'*3 + ' ' + measure + ' ' + '#'*3)
    print('#'*(len(measure) + 8))
    print('\tMaximum:', max[1], '-->', mapping[max[0]])
    print('\tMinimum:', min[1], '-->', mapping[min[0]])
    print('\tAverage: {:.3f}'.format(sum(lst)/len(lst)))
    print('\tVariance: {:.3f}\n'.format(np.var(lst)))


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
    plt.savefig('imgs/' + label + EXT)


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
    plt.suptitle('Hits', fontsize=16)
    _, ax = plt.subplots()
    width = 0.35
    
    ax.bar(ind, ys_h, width=width, label=label_h, color=color_h)
    ax.bar([i + width for i in ind], ys_a, width=width, label=label_a, color=color_a)

    ax.set_xlabel('nodes')
    ax.legend()
    ax.autoscale_view()
    plt.xticks(ind, x_lab, rotation=90, fontsize='xx-small')
    plt.savefig('imgs/hits' + EXT)


def compute_metrics(G, metrics, plot=False):
    m = {}
    color = ''

    if metrics == 'betweenness':
        m = nx.betweenness_centrality(G)
        color = 'black' 
    elif metrics == 'closeness':
        m = nx.closeness_centrality(G)
        color = 'red'
    elif metrics == 'pagerank':
        m = nx.pagerank(G, alpha=0.8, max_iter=1500, tol=1e-03)
        color = 'green'
    elif metrics == 'clustering':
        m = nx.clustering(G)
        color = 'purple'

        if plot:
            m = sorted(m.items(), key=lambda p: p[1])
            plot_metrics(m, metrics, color, samples=len(m))
            
        return m
    elif metrics == 'hits':
        (hubs, authorities) = nx.hits(G, max_iter=100, tol=1e-03, normalized=True)
        color = 'blue orange'
        if plot:
            h_s = sorted(hubs.items(), key=lambda p: p[1])
            a_s = sorted(authorities.items(), key=lambda p: p[1])

            print_statistics(h_s, 'Hubs', G.mapping)
            print_statistics(a_s, 'Authorities', G.mapping)
            plot_hits(hubs, authorities, 'Hubs Authorities', color, samples=20)

        return (hubs, authorities)
    else:
        print('ERROR: Metric "{}" does not implemented !'.format(metrics))
        exit(1)
    
    if plot:
        m = sorted(m.items(), key=lambda p: p[1])
        print_statistics(m, metrics, G.mapping)
        plot_metrics(m, metrics, color)
    
    return m


def compute_triangles(G):
    trg = nx.triangles(G)
    return sum(trg.values()) // 3 # because each vertex is counted in each triangle in which appears
    

def print_graph_info(G):
    # name
    print('/' + '*'*(len(G.name)+2) + '\\')
    print('|', G.name, '|')
    print('\\' + '*'*(len(G.name)+2) + '/\n')

    # graph info
    print('  > Nodes:', G.nnodes)
    print('  > Edges:', G.nedges)
    print('  > Type of edges:', 'Directed' if nx.is_directed(G) else 'Undirected')
    print('  > Average degree:\n\t<k> = {:.3f}'.format(G.avg_degree))
    print('  > Average clustering coefficient:\n\tC = {:.3f}'.format(nx.average_clustering(G)))
    print('  > Density:\n\trho = {:.3f}'.format(nx.density(G)))
    print('  > Number of triangles:', compute_triangles(G))
    print('  > Connectivity:', 'Connected' if nx.is_connected(G) else 'Disconnected')
    print('  > Assortativity:\n\tr = {:.3f}'.format(nx.degree_assortativity_coefficient(G)))
    print('  > Giant component coverage: {:.2f}%\n'.format(max([len(cc) for cc in nx.connected_components(G)])/G.nnodes*100))


def initialize_graph():
    G = nx.read_edgelist(DATASET_FILE, nodetype=int, delimiter=',')
    G.name = GRAPH_NAME
    G.nnodes = len(G.nodes())
    G.nedges = len(G.edges())
    G.avg_degree = 2*G.nedges / G.nnodes if not nx.is_directed(G) else G.nedges / G.nnodes
    G.mapping = {}

    # in order to know which node corresponds to which show
    with open(MAPPING_FILE, 'r') as mf:
        line = mf.readline()
        while line:
            if line.startswith('#'):
                line = mf.readline()
                continue
            fields = line.split(',')
            show_name = fields[1]
            node = int(fields[2])
            G.mapping[node] = show_name

            line = mf.readline()

    return G


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
    plt.savefig('imgs/deg_stribution' + EXT)


@timeit
def draw_graph(G, filename):
    print('Drawing graph...')
    
    plt.figure()
    nx.draw(G, pos=nx.kamada_kawai_layout(G), node_size=5, linewidths=0.5, node_color='blue')
    plt.suptitle(GRAPH_NAME, fontsize=15, color='#116B17', x=0.69, y=0.05)
    plt.savefig('imgs/' + filename + EXT)


@timeit
def compute_distances(G):
    print('Computing diameter of the graph...')
    print('Computing average shortest path length of the graph...')

    if not nx.is_connected(G):
        print('Graph not connected, considering its giant component...')
        ccs = [G.subgraph(cc) for cc in nx.connected_components(G)]
        ccs_sz = [len(cc) for cc in ccs]
        max_idx = ccs_sz.index(max(ccs_sz))
        G = ccs[max_idx]

    d = nx.diameter(G)
    avg_sp = nx.average_shortest_path_length(G)
    print('  > Diameter:\n\td = {}'.format(d))
    print('  > Average shortest path length: {:.3f}'.format(avg_sp))


@timeit
def main():
    G = initialize_graph()
    # G = nx.gnp_random_graph(30, 0.2)
    # G.mapping = {}
    # G.nnodes = len(G.nodes())
    # for i in range(len(G.nodes())):
    #     G.mapping[i] = i


    print_graph_info(G)

    draw_thread = th.Thread(target=draw_graph, args=(G, 'big_graph',))
    diameter_thread = th.Thread(target=compute_distances, args=(G,))
    
    # heavy computation done in parallel
    diameter_thread.start()
    draw_thread.start()

    # wait for all threads to terminate, due to plt sharing
    diameter_thread.join()
    draw_thread.join()

    # centrality measures and graph exploration
    degree_distribution(G)
    compute_metrics(G, 'clustering', True)
    compute_metrics(G, 'hits', True)
    compute_metrics(G, 'pagerank', True)
    compute_metrics(G, 'betweenness', True)
    compute_metrics(G, 'closeness', True)
    

if __name__ == "__main__":
    matplotlib.use('Agg') # to avoid concurrency problem with matplotlib,
                          # run non graphical thread in background instead
                          # (cannot show plots, only save images on file)
    main()