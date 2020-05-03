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


def draw_outbreak(G, timestamp, ctx):
    colors = []
    ctx_fields = ctx.split('_')
    selection_method = ctx_fields[4]

    for n in G.nodes():
        if G.nodes[n]['infected']:
            colors.append(METRIC_COMPLEMENTAR[selection_method] if selection_method != 'hits' else METRIC_COMPLEMENTAR[selection_method].split(' ')[0])
        else:
            colors.append(METRIC_COLOR[selection_method] if selection_method != 'hits' else METRIC_COLOR[selection_method].split(' ')[0])

    picture_name = ctx_fields[1].title() + ', P = {}/{}'.format(ctx_fields[2], ctx_fields[3]) + ', Type: ' + ctx_fields[4] + ', t = ' + str(timestamp)
    draw_graph(G, CONT_VID_DIR + ctx + '{:05d}'.format(timestamp), picture_name, colors, 0.4, False)


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
def spread_contagion(G, cont_type, threshold, ctx, verbose=False):
    if cont_type != 'simple' and cont_type != 'complex':
        exit(RED + 'Type of contagion not recognized !' + RESET)

    timestamp = 1   
    id_vid = 2      # identifier of frame
    tick = 1        # to avoid drawing too much pictures
    infected_per_iter = []  # keep track of the evolution of the disease

    draw_outbreak(G, 1, ctx) # starting situation

    Q = queue.SimpleQueue()
    nx.set_node_attributes(G, False, 'visited') # at the beginning no node is visited
    
    patients_zero = get_infected(G)
    old_infected_size = 0
    new_infected_size = len(patients_zero)
    infected_per_iter += [new_infected_size]

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

        # update results only if something happened in the network
        if old_infected_size != new_infected_size:
            infected_per_iter += [new_infected_size]

            if verbose:
                print(GREEN + 'iteration:', timestamp, '\t\tINFECTED:', new_infected_size, RESET)
            tick += 1
            if tick < 10 or tick % 20 == 0:
                draw_outbreak(G, id_vid, ctx)
                id_vid += 1
        else:
            if verbose:
                print(RED + 'iteration:', timestamp, '\t\tINFECTED:', new_infected_size, RESET)

        timestamp += 1

    print(PURPLE + '\tGenerations:', timestamp, '\tInfected:', len(get_infected(G)), RESET, '\n')
    return infected_per_iter


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


def plot_contagion(infected, context, simple):
    pass


def build_context(small, p, selection_method, cont_type):
    name = 'small_' if small else 'big_'
    name += cont_type + '_'
    name += ('1_' if cont_type == 'complex' else '0_') + str(p) + '_'
    name += selection_method + '_'
    
    return name


def infect_network(G, selection_method, small, n_payoffs=3):
    simple_trend = {}
    complex_trend = {}

    patients_zero = init_contagion(G, selection_method, 5 if small else 50)
    set_property(G, patients_zero, 'infected')  # infect the nodes that start spreading the disease
    print(RED + '\nPatients 0s:' + BLUE, len(get_infected(G)), RESET)    

    print(YELLOW + '    ### ' + PURPLE + 'SIMPLE' + YELLOW + ' ###' + RESET)
    print(RED + '\t|- ' + BLUE + 'Target: ' + GREEN +  selection_method + RESET)
    print(RED + '\t|- ' + BLUE + 'P = ' + GREEN + '0.15' + RESET)
    ctx = build_context(small, 15, selection_method, 'simple')

    simple_trend['0.15'] = spread_contagion(G.copy(), 'simple', 0.15, ctx)
    #plot_contagion(simple_trend, context, True)

    print(YELLOW + '    ### ' + PURPLE + 'COMPLEX' + YELLOW + ' ###' + RESET)
    weight = 5    
    for i in range(n_payoffs):
        # generate payoff matrix
        pmatrix = np.eye(2)
        pmatrix[0,0] = weight
        p_infection = pmatrix[1,1]/(pmatrix[0,0] + pmatrix[1,1])
        p_str = '{}/{}'.format(int(pmatrix[1,1]), int(pmatrix[0,0] + pmatrix[1,1]))

        print(RED + '\t|- ' + BLUE + 'Target: ' + GREEN + selection_method + RESET)
        print(RED + '\t|- ' + BLUE + 'P = ' + GREEN + p_str + RESET)
        ctx = build_context(small, int(pmatrix[0,0]), selection_method, 'complex')
        
        complex_trend['p_str'] = spread_contagion(G.copy(), 'complex', p_infection, ctx)

        if i % 2 == 0:
            weight += 5
        else:
            weight *= 10
    #plot_contagion(complex_trend, context, False)

    
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
    
    # compute_layout(G)
    contagions = ['random', 'hits', 'closeness', 'betweenness', 'pagerank', 'clustering']

    for cont in contagions:
        infect_network(G, cont, small)

    
if __name__ == '__main__':
    main(check_cli_arguments('small'))