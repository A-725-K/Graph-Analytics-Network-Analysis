from lib.attack_utils import *
from lib.general_utils import check_cli_arguments


@timeit
def main(small=False):
    print_title('Network Robustness')

    G = None

    if small:
        n = 500
        p = 0.02

        G = generate_connected_random_graph(n, p)
    else:
        G = initialize_graph()

    compute_layout(G)

    attacks = ['random', 'hits', 'closeness', 'betweenness', 'pagerank', 'clustering']
    parallelize_attacks(G, attacks)
    plot_attack_results(small)
    make_videos(small)


if __name__ == '__main__':
    matplotlib.use('Agg')   # to avoid concurrency problem with matplotlib,
                            # run non graphical thread in background instead
                            # (cannot show plots, only save images on file)

    main(check_cli_arguments('small'))