# coding=utf8
from utils.writer import write_results

class Solver():

    def __init__(self, model, optimizer, exp_path, logger, device=None):
        """
            model: main model to train and evaluate
            optimizer: optimizer for training
            exp_path: export path to write test results, best model
            logger: training logger to record history
            device: torch.device()
        """
        super(Solver, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.exp_path = exp_path
        self.logger = logger
        self.device = device
    
    def train_and_decode(self, *args, **kargs):
        '''
            Training and test mode, pick the best model on dev dataset and 
            evaluate on test set.
        '''
        raise NotImplementedError

    def decode(self, *args, **kargs):
        '''
            Only evaluate on dataset and write down results
        '''
        raise NotImplementedError

class QNLISolver(Solver):

