from .common import *


def print_title(title):
    print('\n' + BLUE + '#'*(len(title) + 8))
    print(BLUE + '#'*3 + ' ' + YELLOW + title + ' ' + BLUE + '#'*3)
    print(BLUE + '#'*(len(title) + 8) + RESET)


def print_statistics(values, measure, mapping):
    max = values[-1]
    min = values[0]
    lst = [i for _, i in values]

    print_title(measure)
    print(RED + '\t+++')
    print(RED + '\t |- ' + YELLOW + 'Maximum:', WHITE + str(max[1]), '--> (', mapping[max[0]], ')')
    print(RED + '\t |- ' + YELLOW + 'Minimum:', WHITE + str(min[1]), '--> (', mapping[min[0]], ')')
    print(RED + '\t |- ' + YELLOW + 'Average: {}'.format(WHITE + str(sum(lst)/len(lst))))
    print(RED + '\t |- ' + YELLOW + 'Variance: {}'.format(WHITE + str(np.var(lst))))
    print(RED + '\t+++\n' + RESET)


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
        print(YELLOW + 'Function [{}]\telapsed time: {}'.format(f.__name__, BLUE + format_time(end - start)) + RESET)
        return ret_val

    return timed_foo


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
