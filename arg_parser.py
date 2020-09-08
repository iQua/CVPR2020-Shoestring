import argparse

def parameter_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir",
                        nargs = "?",
                        default = "result.txt",
	                    help = "Store the result.")

    parser.add_argument("--k",
                        nargs="+",
                        type=float,
                        help = "Select the top k probs for each class as the unlabeled data. Between 1 and 0. Default is 0.")

    parser.add_argument("--lam",
                        nargs="+",
                        type=float,
                        help = "Weight for similarity calculated using distance. Default is 0.01.")

    parser.add_argument("--pset",
                        nargs="+",
                        type=str,
                        help = "Train size and parameters. Options are: config_citation.one_label_set, config_citation.two_label_set, config_citation.five_label_set. Default is config_citation.one_label_set.")

    parser.add_argument("--dataset",
                        nargs="+",
                        type=str,
                        help = "Dataset to train. Options are: cora, large_cora, citeseer, pubmed. Default is cora.")

    parser.add_argument("--method",
                        nargs="+",
                        type=str,
                        help = "Method to calculate the distance. Options are l1, l2, cos. Default is cos.")

    parser.set_defaults(k = [0])
    parser.set_defaults(lam = [0.01])
    parser.set_defaults(pset = ['config_citation.one_label_set'])
    parser.set_defaults(dataset = ['cora'])
    parser.set_defaults(method = ['cos'])
    
    return parser.parse_args()