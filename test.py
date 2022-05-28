import argparse


parser = argparse.ArgumentParser(description='Training of Neural Networks, the Barkley Diver')
parser.add_argument('-depth', '--depth', type=int)

_args = parser.parse_args()
args = vars(_args)

print(args['depth'], type(args['depth']))