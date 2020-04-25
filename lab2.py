from lib.attack_utils import *


@timeit
def main(small=False):
    print_title('Network Robustness')

    G = None

    if small:
        n = 100
        p = 0.05

        G = generate_connected_random_graph(n, p)
    else:
        G = initialize_graph()

    compute_layout(G)

    attacks = ['random', 'hits', 'closeness', 'betweenness', 'pagerank', 'clustering']
    parallelize_attacks(G, attacks)
    plot_attack_results(small)
    make_videos(small)


if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    matplotlib.use('Agg')   # to avoid concurrency problem with matplotlib,
                            # run non graphical thread in background instead
                            # (cannot show plots, only save images on file)

    if argc > 2 or argc == 0:
        exit(RED + 'Usage: python3 lab2.py [--small]' + RESET)
    
    elif argc == 2:
        if args[1] != '--small':
            exit(RED + 'Usage: python3 lab2.py [--small]' + RESET)
        main(True)
        
    else:
        main()
