import copy

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


def init_attack(G):
    print(PURPLE + 'Initializing the attack vector...' + RESET)
    d, avg_sp = compute_distances(G, False)
    n0 = len(G.nodes())

    init = {'diameter': [d],
            'avg_sp': [avg_sp],
            'dim_gc': [n0],    # because from Lab1 I know that GC == graph (coverage = 100%)
            '<k>': [2*len(G.edges()) / n0 if not nx.is_directed(G) else len(G.edges()) / n0],
            'rel_dim': [len(G.nodes()) / n0 * 100]}
    
    return init, n0


def update_trend(G, n, n0, results):
    d, avg_sp = compute_distances(G, False)

    results['diameter'] += [d]
    results['avg_sp'] += [avg_sp]
    results['dim_gc'] += [n]
    results['<k>'] += [2*len(G.edges()) / n if not nx.is_directed(G) else len(G.edges()) / n]
    results['rel_dim'] += [n/n0 * 100]


def adjust_how_much(n):
    if n > 1500:
        return 200
    if n > 800:
        return 100
    if n > 400:
        return 50
    if n > 200:
        return 10
    return 1


@timeit
def attack_network(G, atk_type, results, n0):
    graphs_for_videos = {0: G.copy()}
    how_much = adjust_how_much(n0)
    iterations = 1
    img_idx = 1
    while True:
        if atk_type == 'random':
            to_remove = list(np.random.choice(G.nodes(), how_much, False))
        elif atk_type == 'hits':
            # when the graph is too small it could be disconnected
            # then I interrupt the algorithm
            try:
                hubs, _ = compute_metrics(G, atk_type)
            except ZeroDivisionError as _:
                # this means that the max value of HITS in G is 0
                break
            to_remove = list(map(lambda p: p[0], list(hubs.items())[::-1]))
        else:
            target = compute_metrics(G, atk_type)
            to_remove = list(map(lambda p: p[0], list(target.items())[::-1]))

        # remove interesting nodes from the graph
        for i, node in enumerate(to_remove):
            if i == how_much:
                break
            G.remove_node(node)

        n = len(G.nodes())
        how_much = adjust_how_much(n)
        if n <= 1:
            break

        G = nx.Graph(max([G.subgraph(cc) for cc in nx.connected_components(G)], key=len))
        update_trend(G, n, n0, results)

        if iterations % 10 == 0:
            print('\nAttack type: {}\t\tIteration: {}\t\t|V| = {}'.format(atk_type, iterations, n))
            graphs_for_videos[img_idx] = G.copy()
            img_idx += 1

        iterations += 1

    results['iters'] = iterations
    results['graphs'] = graphs_for_videos
    
    return results


def make_videos():
    global GLOBAL_RESULTS

    for m, res in GLOBAL_RESULTS.items():
        for idx, g in res['graphs'].items():
            draw_graph(g, 'videos/{}{:04d}'.format(m, idx), m.title() + ' Attack', 'red')


def runnable(G, atk, lock, init, n0):
    global GLOBAL_RESULTS

    result = attack_network(G, atk, init, n0)
    with lock:
        GLOBAL_RESULTS[atk] = result
        print(GREEN + 'Finish: ' + atk + RESET)


# compute attacks in parallel
def parallelize_attacks(G, attacks):
    global GLOBAL_RESULTS, IS_FINISHED

    attacks_threads = []
    dot_th = th.Thread(target=print_dots)
    lock = th.Lock()
    init, n0 = init_attack(G)

    print('\n' + RED + ':::::::::::::::::::::::::::::' + RESET)
    print(RED + '::: STARTING THE ATTACK ! :::' + RESET)
    print(RED + ':::::::::::::::::::::::::::::' + RESET + '\n')
    for a in attacks:
        attacks_threads += [th.Thread(target=runnable, args=(G.copy(), a, lock, copy.deepcopy(init), n0,))]
    
    dot_th.start()
    for at in attacks_threads:
        at.start()

    for at in attacks_threads:
        at.join()
    IS_FINISHED = True

    dot_th.join()

    print('\n', GLOBAL_RESULTS)
    make_videos()
    #plot_results()


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
    #     G = generate_random_graph(50, 0.08)
    #     if not fst:
    #         print(RED + 'The graph is not connected... Trying again...' + RESET)
    #     fst = False
    # print(PURPLE + '\tN = {}\tlog(N) = {:.3f}\t log(N)/N = {:.3f}'.format(len(G), np.log(len(G)), len(G)/np.log(len(G))) + RESET)
    
    compute_layout(G)

    attacks = ['random', 'hits', 'closeness', 'betweenness', 'pagerank', 'clustering']
    parallelize_attacks(G, attacks)


if __name__ == '__main__':
    main()