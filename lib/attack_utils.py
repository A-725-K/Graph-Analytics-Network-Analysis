import copy

import matplotlib

from .common import *
from .general_utils import timeit, initialize_graph
from .graph_utils import generate_connected_random_graph, compute_metrics, compute_distances, draw_graph, compute_layout, print_title


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
        return 100
    if n > 800:
        return 50
    if n > 400:
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
            try:
                to_remove = list(np.random.choice(G.nodes(), how_much, False))
            except Exception as _:
                # this means there are too few nodes
                break
        elif atk_type == 'hits':
            # when the graph is too small it could be disconnected
            # then I interrupt the algorithm
            try:
                hubs, _ = compute_metrics(G, atk_type)
            except ZeroDivisionError as _:
                # this means that the max value of HITS in G is 0
                break
            to_remove = list(map(lambda p: p[0], sorted(hubs.items(), key=lambda p: p[1], reverse=True)))
        else:
            target = compute_metrics(G, atk_type)
            to_remove = list(map(lambda p: p[0], sorted(target.items(), key=lambda p: p[1], reverse=True)))

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

        if iterations % 2 == 0:    
            graphs_for_videos[img_idx] = G.copy()
            img_idx += 1

        iterations += 1

    results['iters'] = iterations
    results['graphs'] = graphs_for_videos
    
    return results


def make_videos(small):
    global GLOBAL_RESULTS

    pre = None
    if small:
        pre = 'small_'
    else:
        pre = 'big_'

    for m, res in GLOBAL_RESULTS.items():
        for idx, g in res['graphs'].items():
            draw_graph(g, ATK_VID_DIR + pre + '{}{:04d}'.format(m, idx), m.title() + ' Attack', METRIC_COLOR[m] if m != 'hits' else METRIC_COLOR[m].split(' ')[0])


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
    
    # for DEBUG purpose
    # print('\n', GLOBAL_RESULTS)


def compare_gc_dim(attack, dim_gc, rel_gc_dim, small):
    pre = None
    if small:
        pre = 'small_'
    else:
        pre = 'big_'

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    fig.suptitle('{} Attack'.format(attack.title()))
    
    ax1.plot(dim_gc, color='purple')
    ax1.grid()
    ax1.set(xlabel='#iterations', ylabel='dim of GC [#nodes]')
    
    ax2.plot(rel_gc_dim, color='orange')
    ax2.grid()
    ax2.set(xlabel='#iterations', ylabel='realtive dim of GC [%]')
    ax2.set_yticks(np.arange(0, 101, 10))
    ax2.set_yticklabels(list(map(lambda n: '{}%'.format(n), np.arange(0, 101, 10))), rotation='45')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(ATK_DIR + pre + attack + '_dims' + EXT)
    plt.close()


def compare_distances(measure, diameter, avg_k, avg_sp, small):
    pre = None
    if small:
        pre = 'small_'
    else:
        pre = 'big_'

    plt.figure()
    plt.title('Evolution of Distances in {} Attack'.format(measure.title()))
    plt.xlabel('#iterations')
    plt.ylabel('measures')
    plt.plot(diameter, color='green')
    plt.plot(avg_k, color='red')
    plt.plot(avg_sp, color='orange')
    plt.legend(['diameter', '<k>', 'average shortest path length'], loc='best', fancybox=True, shadow=True)
    plt.grid()
    plt.tight_layout()
    plt.savefig(ATK_DIR + pre + measure + '_dist' + EXT)
    plt.close()


def compare_attacks(attacks, small):
    legend_labels = attacks.keys()
    pre = None
    if small:
        pre = 'small_'
    else:
        pre = 'big_'

    plt.figure()
    plt.title('Comparison of Attacks')
    plt.xlabel('#iterations')
    plt.ylabel('dim of GC [#nodes]')
    for m, atk in attacks.items():
        plt.plot(atk, color=(METRIC_COLOR[m] if m != 'hits' else METRIC_COLOR[m].split(' ')[0]))
    plt.grid()
    plt.tight_layout()
    plt.legend(legend_labels, loc='best', fancybox=True, shadow=True)
    plt.savefig(ATK_DIR + pre + 'all' + EXT)
    plt.close()


def plot_attack_results(small):
    global GLOBAL_RESULTS

    all_atks = {}
    for measure, result in GLOBAL_RESULTS.items():
        compare_gc_dim(measure, result['dim_gc'], result['rel_dim'], small)
        compare_distances(measure, result['diameter'], result['<k>'], result['avg_sp'], small)
        all_atks[measure] = result['dim_gc']
    compare_attacks(all_atks, small)
