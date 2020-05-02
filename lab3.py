import queue

from lib.common import *
from lib.general_utils import initialize_graph, check_cli_arguments, timeit
from lib.graph_utils import generate_connected_random_graph, compute_layout, compute_metrics, draw_graph, print_title


def init_contagion(G, cont_type, how_much=250):
    # in principle, all nodes are sane
    nx.set_node_attributes(G, False, 'infected')

    # select the first infected nodes
    metric = None
    if cont_type == 'random':
        metric = list(np.random.choice(G.nodes(), how_much, False))
        return metric
    elif cont_type == 'hits':
        metric, _ = compute_metrics(G, cont_type)
    else:
        metric = compute_metrics(G, cont_type)

    metric = sorted(metric.items(), key=lambda p: p[1], reverse=True)
    metric = list(map(lambda p: p[0], metric))
        
    return metric[:how_much]


def get_neighbors_list(G, n):
    return list(G[n].keys())


def set_property(G, nodes, property):
    for n in nodes:
        G.nodes[n][property] = True


def draw_outbreak(G, timestamp):
    colors = []
    for n in G.nodes():
        if G.nodes[n]['infected']:
            colors.append('red')
        else:
            colors.append('blue')
    draw_graph(G, CONT_DIR + 'test{:03d}'.format(timestamp), 't = {}'.format(timestamp), colors)


def infect_single_node(G, n, threshold, cont_type):
    if cont_type == 'simple':
        simple_infection(G, n, threshold)
    else:
        complex_infection(G, n, threshold)


def visit_neighbourhood(G, seed, Q, cont_type, threshold):
    infect_single_node(G, seed, threshold, cont_type)
    for adj in get_neighbors_list(G, seed):
        if G.nodes[adj]['visited']:
            continue

        G.nodes[adj]['visited'] = True
        Q.put(adj)

        infect_single_node(G, adj, threshold, cont_type)
      

# type = simple|complex
#   - simple : infect each neighbor node with a probability p
#   - complex: get infected if #neighbor infected is > p
def spread_contagion(G, cont_type, threshold, verbose=False):
    if cont_type != 'simple' and cont_type != 'complex':
        exit(RED + 'Type of contagion not recognized !' + RESET)

    timestamp = 1   
    id_vid = 2      # identifier of frame
    tick = 1        # to avoid drawing too much pictures

    draw_outbreak(G, 1)

    Q = queue.SimpleQueue()
    nx.set_node_attributes(G, False, 'visited') # at the beginning no node is visited
    
    patients_zero = get_infected(G)
    old_infected_size = 0
    new_infected_size = len(patients_zero)

    # chose the first node of the visit
    G.nodes[patients_zero[0]]['visited'] = True
    Q.put(patients_zero[0])
    
    # approach similar to a BFS visit
    N = len(G.nodes())
    while new_infected_size < N:
        seed = Q.get()

        visit_neighbourhood(G, seed, Q, cont_type, threshold)
    
        infected = get_infected(G)
        old_infected_size = new_infected_size
        new_infected_size = len(infected)

        if old_infected_size != new_infected_size:
            if verbose:
                print(GREEN + 'iteration:', timestamp, '\t\tINFECTED:', new_infected_size, RESET)
            tick += 1
            if tick < 10 or tick % 5 == 0:
                draw_outbreak(G, id_vid)
                id_vid += 1
        else:
            if verbose:
                print(RED + 'iteration:', timestamp, '\t\tINFECTED:', new_infected_size, RESET)

        timestamp += 1

    print(PURPLE + '\tGenerations:', timestamp, '\tInfected:', len(get_infected(G)), RESET)


def count_infected(G, node_list):
    count = 0
    for n in node_list:
        if G.nodes[n]['infected']:
            count += 1
    return count


def complex_infection(G, adj, threshold):
    nn = get_neighbors_list(G, adj)
    nn_sz = len(nn)

    infected_ratio = count_infected(G, nn)/nn_sz
    if nn_sz > 0 and infected_ratio >= threshold:
        G.nodes[adj]['infected'] = True
    else:
        G.nodes[adj]['visited'] = False


def simple_infection(G, adj, threshold):
    p_infection = rnd.random()
    if p_infection <= threshold:
        G.nodes[adj]['infected'] = True
    else:
        G.nodes[adj]['visited'] = False


def get_infected(G):
    return [n for n in G.nodes if G.nodes[n]['infected']]


def infect_network(G, selection_method, small, n_payoffs=3):
    patients_zero = init_contagion(G, selection_method, 5)
    set_property(G, patients_zero, 'infected')  # infect the nodes that start spreading the disease
    print(RED + '\nPatients 0s:' + BLUE, len(get_infected(G)), RESET)    

    print('Type: SIMPLE')
    print('Target:', selection_method, 'P = 0.1')
    spread_contagion(G.copy(), 'simple', 0.1)
    print('Type: COMPLEX')
    i = 5
    for _ in range(n_payoffs):
        # generate payoff matrix
        pmatrix = np.eye(2)
        pmatrix[0,0] = i
        p_infection = pmatrix[1,1]/(pmatrix[0,0] + pmatrix[1,1])

        print('Target:', selection_method, 'P = {}/{}'.format(int(pmatrix[1,1]), int(pmatrix[0,0] + pmatrix[1,1])))
        spread_contagion(G.copy(), 'complex', p_infection)

        i *= 5

    
@timeit       
def main(small=False):
    print_title('Social Contagion')

    G = None
    if small:
        n = 300
        m = 4
        G = nx.barabasi_albert_graph(n, m, 12)
        
        G.nnodes = len(G.nodes())
    else:
        G = initialize_graph()
    
    compute_layout(G)
    contagions = ['random', 'hits', 'closeness', 'betweenness', 'pagerank', 'clustering']

    for cont in contagions:
        infect_network(G, cont, small)

    
if __name__ == '__main__':
    main(check_cli_arguments('small'))