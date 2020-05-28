import matplotlib

from lib.common import *
from lib.general_utils import timeit, initialize_graph, check_cli_arguments
from lib.graph_utils import draw_graph, compute_layout, compute_distances, compute_metrics
from lib.analysis_utils import degree_distribution, assortativity_matrix, analyze_communities, print_graph_info


@timeit
def main():
    G = initialize_graph()

    # create a new random graph for debug purpose
    # G = generate_random_graph(100, 0.2)

    print_graph_info(G)

    layout_thread = th.Thread(target=compute_layout, args=(G,))
    diameter_thread = th.Thread(target=compute_distances, args=(G,))
    
    # heavy computation done in parallel
    diameter_thread.start()
    layout_thread.start()

    # wait for all threads to terminate, due to plt sharing
    diameter_thread.join()
    layout_thread.join()

    # centrality measures and graph exploration
    draw_graph(G, 'imgs/big_graph')
    degree_distribution(G)
    assortativity_matrix(G)
    analyze_communities(G)
    compute_metrics(G, 'clustering', True)
    compute_metrics(G, 'hits', True)
    compute_metrics(G, 'pagerank', True)
    compute_metrics(G, 'betweenness', True)
    compute_metrics(G, 'closeness', True)


def interactive_main():
    G = initialize_graph()
    
    while True:
        user_input = input('{}Tell me which node you want to know [0-{}] (END to exit):{}\t'.format(BLUE, G.nnodes-1, RESET))
        if user_input == 'END':
            break
        
        try:
            choice = int(user_input)    
            print(PURPLE + 'Node {} is the show:'.format(choice) + YELLOW + '\t' + G.mapping[choice] + RESET)
        except Exception as _:
            continue
    

if __name__ == "__main__":
    if check_cli_arguments('interactive'):
        interactive_main()
    else:
        matplotlib.use('Agg')   # to avoid concurrency problem with matplotlib,
                                # run non graphical thread in background instead
                                # (cannot show plots, only save images on file)
        main()
        