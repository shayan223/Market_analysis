import argparse





def __main__():

    parser = argparse.ArgumentParser(description='Used to set options, train, and test a Deep-Q agent')

    parser.add_argument('--generate-plots', type=bool,
                        help='')

    parser.add_argument('--train-models')

    parser.add_argument('--epochs', type=int,
                        help='')