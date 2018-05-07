import argparse

def str2bool(arg):
    arg = arg.lower()
    if arg in {'1', 't', 'true', 'y', 'yes'}:
        return True
    elif arg in {'0', 'f', 'false', 'n', 'no'}:
        return False
    else:
        raise  argparse.ArgumentTypeError('Boolean value expected. %s was given' % arg)
