import queue

from .common import *
from .graph_utils import generate_connected_random_graph, compute_metrics, draw_graph, get_neighbors_list, set_property


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


def draw_outbreak(G, timestamp, ctx):
    colors = []
    ctx_fields = ctx.split('_')
    selection_method = ctx_fields[4]

    for n in G.nodes():
        if G.nodes[n]['infected']:
            colors.append(METRIC_COMPLEMENTAR[selection_method] if selection_method != 'hits' else METRIC_COMPLEMENTAR[selection_method].split(' ')[0])
        else:
            colors.append(METRIC_COLOR[selection_method] if selection_method != 'hits' else METRIC_COLOR[selection_method].split(' ')[0])

    p_str = ', P = {}{}{}'.format(ctx_fields[2], '/' if ctx_fields[1] != 'simple' else '.', ctx_fields[3])
    picture_name = ctx_fields[1].title() + p_str + ', Type: ' + ctx_fields[4] + ', t = ' + str(timestamp)
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
def spread_contagion(G, cont_type, threshold, ctx, small, verbose=False):
    if cont_type != 'simple' and cont_type != 'complex':
        exit(RED + 'Type of contagion not recognized !' + RESET)

    timestamp = 1   
    id_vid = 2      # identifier of frame
    tick = 1        # to avoid drawing too much pictures
    infected_per_iter = []  # keep track of the evolution of the disease
    same_sz = 0     # heuristic optimization

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
    
    # approach similar to a BFS visit, upperbound 75k iterations
    # if after 5k time units there is no change it breaks
    N = len(G.nodes())
    while new_infected_size < N and timestamp < 75000:
        seed = Q.get()

        visit_neighbourhood(G, seed, Q, cont_type, threshold)
    
        infected = get_infected(G)
        old_infected_size = new_infected_size
        new_infected_size = len(infected)

        # update results only if something happened in the network
        if old_infected_size != new_infected_size:
            infected_per_iter += [new_infected_size]
            same_sz = 0     # reset the counter

            if verbose:
                print(GREEN + 'iteration:', timestamp, '\t\tINFECTED:', new_infected_size, RESET)
            tick += 1

            if tick_draw(G, id_vid, tick, ctx, 20 if small else 50):
                id_vid += 1
        else:
            if verbose:
                print(RED + 'iteration:', timestamp, '\t\tINFECTED:', new_infected_size, RESET)

            # if it does not happen anything for too long, unlikely it happens in the future
            same_sz += 1
            if same_sz == 5000:
                break

        timestamp += 1
        
    draw_outbreak(G, timestamp, ctx) # final situation
    print(PURPLE + '\tGenerations:', timestamp, '\tInfected:', len(get_infected(G)), RESET, '\n')
    return infected_per_iter


def tick_draw(G, id_vid, tick, ctx, tick_cnt):
    if tick < tick_cnt/2 or tick % tick_cnt == 0:
        draw_outbreak(G, id_vid, ctx)
        return True

    return False


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


def build_context(small, p, selection_method, cont_type):
    name = 'small_' if small else 'big_'
    name += cont_type + '_'
    name += ('1_' if cont_type == 'complex' else '0_') + str(p) + '_'
    name += selection_method + '_'
    
    return name


def infect_network(G, selection_method, small, n_payoffs=3):
    simple_trend = {}
    complex_trend = {}

    patients_zero = init_contagion(G, selection_method, 5 if small else 250)
    set_property(G, patients_zero, 'infected')  # infect the nodes that start spreading the disease
    print(RED + '\nPatients 0s:' + BLUE, len(get_infected(G)), RESET)    

    p_simple = 0.15 if small else 0.4
    print(YELLOW + '    ### ' + PURPLE + 'SIMPLE' + YELLOW + ' ###' + RESET)
    print(RED + '\t|- ' + BLUE + 'Target: ' + GREEN +  selection_method + RESET)
    print(RED + '\t|- ' + BLUE + 'P = ' + GREEN + str(p_simple) + RESET)
    ctx = build_context(small, int(p_simple*100), selection_method, 'simple')

    simple_trend[str(p_simple)] = spread_contagion(G.copy(), 'simple', p_simple, ctx, small)
    plot_contagion(len(G), simple_trend, ctx, True, small, selection_method)

    print(YELLOW + '    ### ' + PURPLE + 'COMPLEX' + YELLOW + ' ###' + RESET)
    weight = 5 if small else 8
    for i in range(n_payoffs):
        # generate payoff matrix
        pmatrix = np.eye(2)
        pmatrix[0,0] = weight
        p_infection = pmatrix[1,1]/(pmatrix[0,0] + pmatrix[1,1])
        p_str = '{}/{}'.format(int(pmatrix[1,1]), int(pmatrix[0,0] + pmatrix[1,1]))

        print(RED + '\t|- ' + BLUE + 'Target: ' + GREEN + selection_method + RESET)
        print(RED + '\t|- ' + BLUE + 'P = ' + GREEN + p_str + RESET)
        ctx = build_context(small, int(pmatrix[0,0]), selection_method, 'complex')
        
        complex_trend[p_str] = spread_contagion(G.copy(), 'complex', p_infection, ctx, small)

        # those values are chosen empirically
        if i % 2 == 0:
            weight += (2 if small else 3)
        else:
            weight *= (5 if small else 2)

    plot_contagion(len(G), complex_trend, ctx, False, small, selection_method)


def plot_contagion(n, trends, context, simple, small, selection_method):
    legend = None
    filename = ('small_' if small else 'big_') + ('simple_' if simple else 'complex_') + selection_method
    title = 'Type of Contagion: {}, Target: ' + selection_method.title()
    color = METRIC_COLOR[selection_method] if selection_method != 'hits' else METRIC_COLOR[selection_method].split(' ')[0]
   
    plt.figure()
    plt.grid()

    # the lambda expression permits to show the number of susceptible nodes, instead of the infected ones
    if simple:
        title = title.format('Simple')
        p_str = list(trends.keys())[0]
        legend = ['P(c) = ' + p_str]
        plt.plot(list(map(lambda  x: n-x, trends[p_str])), color=color)
    else:
        title = title.format('Complex')
        legend = list(map(lambda s: 'b/b+a = ' + s, trends.keys()))   
        for p_str, trend in trends.items():
            plt.plot(list(map(lambda  x: n-x, trend)))

    plt.title(title)
    plt.xlabel('#iterations')
    plt.ylabel('#susceptible')
    plt.legend(legend, loc='best', fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig(CONT_DIR + filename + EXT)
    plt.close()
