from lib.common import nx
from lib.contagion_utils import infect_network
from lib.graph_utils import compute_layout, print_title
from lib.general_utils import initialize_graph, check_cli_arguments, timeit


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
