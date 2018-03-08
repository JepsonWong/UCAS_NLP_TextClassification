import argparse
import os

from utils import util


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
            '--name',
            type=str,
            default='./ucas_nlp_textclassification',
            help='name for this execution'
        )
        self.parser.add_argument(
            '--checkpoints_dir',
            type=str,
            default='./checkpoints',
            help='path to save options for this execution'
        )
        self.parser.add_argument(
            '--encoding',
            type=str,
            required=False,
            default='gb18030',
            help='file encoding of documents'
        )
        self.parser.add_argument(
            '--suffix_accepted',
            type=str,
            default='txt,TXT',
            help='file with suffix_accepted will be read'
        )

        self.initialized = True

    def parse_arguments(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
