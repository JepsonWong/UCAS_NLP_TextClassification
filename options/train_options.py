from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--corpus_root',
            type=str,
            required=False,
            default='./corpus_tiny',
            help='path to documents(should have subfolders C1-Class1Name, C2-Class2Name,...,Cn-ClassnName)'
        )
        self.parser.add_argument(
            '--tfidf_top_k',
            type=int,
            default=5000,
            help='features with tfidf value within top_k will be selected'
        )
        self.parser.add_argument(
            '--path_to_save_model',
            type=str,
            required=True,
            help='path to save the model'
        )
        self.parser.add_argument(
            '--which_filter',
            type=str,
            default='mi',
            help='mi or gbdt to filter the features'
        )
        self.parser.add_argument(
            '--which_classifier',
            type=str,
            default='svm',
            help='svm or gbdt classifier'
        )
        self.parser.add_argument(
            '--mi_threshold',
            type=float,
            default=0.06,
            help='threshold value used by mutual information feature selection'
        )
        self.isTrain = True
