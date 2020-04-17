import matplotlib

from lib.common import *
from lib.general_utils import timeit, initialize_graph
from lib.graph_utils import generate_random_graph, compute_metrics, compute_distances, draw_graph, compute_layout, print_title


GLOBAL_RESULTS = {}
IS_FINISHED = False


def print_dots():
    global IS_FINISHED

    while True:
        print(RED + '.' + RESET, end='')
        sys.stdout.flush()
        time.sleep(10)

        if IS_FINISHED:
            break


@timeit
def attack_network(G, atk_type, tick=0):
    d, avg_sp = compute_distances(G, False)
    n0 = len(G.nodes())

    results = {'diameter': [d],
               'avg_sp': [avg_sp],
               'dim_gc': [n0],    # because from Lab1 I know that GC == graph (coverage = 100%)
               '<k>': [2*len(G.edges()) / n0 if not nx.is_directed(G) else len(G.edges()) / n0],
               'rel_dim': [len(G.nodes()) / n0 * 100]}

    # only if single-thread code
    # draw_graph(G, 'rnd/{}{:04d}'.format(atk_type, 0), atk_type, 'red') # to show the starting graph

    iterations = 1
    # img_idx = 1
    while len(G.nodes()) > 1:
        if atk_type == 'random':
            to_remove = np.random.choice(G.nodes())
        elif atk_type == 'hits':
            hubs, _ = compute_metrics(G, atk_type)
            to_remove = sorted(hubs.items(), key=lambda p: p[1], reverse=True)[0][0]
        else:
            target = compute_metrics(G, atk_type)
            to_remove = sorted(target.items(), key=lambda p: p[1], reverse=True)[0][0]

        G.remove_node(to_remove)
        G = nx.Graph(max([G.subgraph(cc) for cc in nx.connected_components(G)], key=len))
        n = len(G.nodes())
        d, avg_sp = compute_distances(G, False)

        results['diameter'] += [d]
        results['avg_sp'] += [avg_sp]
        results['dim_gc'] += [n]
        results['<k>'] += [2*len(G.edges()) / n if not nx.is_directed(G) else len(G.edges()) / n]
        results['rel_dim'] += [n/n0 * 100]

        # only if single-thread code
        # if iterations % tick == 0:
        #     draw_graph(G, 'rnd/{}{:04d}'.format(atk_type, img_idx), atk_type.title(), 'red')
        #     print('\nIteration: {}\t\t|V| = {}'.format(iterations, n))
        #     img_idx += 1

        iterations += 1
    print()

    results['iter'] = iterations
    return results


def runnable(G, atk, lock):
    global GLOBAL_RESULTS

    result = attack_network(G, atk)
    with lock:
        GLOBAL_RESULTS[atk] = result
        print(GREEN + 'Finish: ' + atk + RESET)


# compute attacks in parallel
def parallelize_attacks(G, attacks):
    global GLOBAL_RESULTS
    global IS_FINISHED

    attacks_threads = []
    dot_th = th.Thread(target=print_dots)
    lock = th.Lock()

    for a in attacks:
        attacks_threads += [th.Thread(target=runnable, args=(G.copy(), a, lock,))]
    
    dot_th.start()
    for at in attacks_threads:
        at.start()

    for at in attacks_threads:
        at.join()
    IS_FINISHED = True

    dot_th.join()

    print(GLOBAL_RESULTS)
    


# TODO: Not Implemented !!
# def plot_results():
#     global GLOBAL_RESULTS
#     pass


@timeit
def main():
    print_title('Network Robustness')

    G = initialize_graph()

    # G = nx.Graph()
    # G.name = 'Random Graph, N = 400, p = 0.02'
    # fst = True
    # while len(G) < 1 or not nx.is_connected(G):
    #     G = generate_random_graph(400, 0.02)
    #     if not fst:
    #         print(RED + 'The graph is not connected... Trying again...' + RESET)
    #     fst = False
    
    #compute_layout(G)
    
    attacks = ['random', 'hits', 'closeness', 'betweenness', 'pagerank', 'clustering']
    parallelize_attacks(G, attacks)
    #plot_results()


if __name__ == '__main__':
    main()
