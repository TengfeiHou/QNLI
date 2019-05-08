# coding=utf8
import time
import numpy as np
from utils.writer import write_results
from utils.batch import to_device

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

    def __init__(self, *args, **kargs):
        super(QNLISolver, self).__init__(*args, **kargs)
        self.history = {
            "best_epoch": 0, "dev_acc": 0.,
            "dev_loss": 1e12, "train_losses": []
        }

    def train_and_decode(train_loader, dev_loader, test_loader, max_epoch):

        for epoch in range(max_epoch):
            self.model.train()
            for data_batch in train_loader:
                self.optimizer['all'].zero_grad()
                word_ids, segment_ids, masks, labels = to_device(data_batch, self.device)
                loss = self.model(word_ids, segment_ids, masks, labels)
                loss.backward()
                self.optimizer['all'].step()
                self.history['train_losses'].append(loss.item())

            if epoch < 5: # start to evaluate after some epochs
                continue
            dev_acc, dev_loss = self.decode(dev_loader, labeled=True, add_loss=True)
            if dev_acc > self.history['dev_acc'] or (dev_acc == self.history['dev_acc'] and dev_loss < self.history['dev_loss']):
                self.history['dev_acc'] = dev_acc
                self.history['dev_loss'] = dev_loss
                self.history['best_epoch'] = epoch
                self.model.save_model(os.path.join(self.exp_path, 'model.pkl'))
                self.logger.info('NEW BEST:\tEpoch : %d\tBest Valid Loss/Acc : %.4f/%.4f' % (epoch, dev_loss, dev_acc))
    
        self.model.load_model(os.path.join(self.exp_path, 'model.pkl'))
        self.logger.info('FINAL BEST RESULT: \tEpoch : %d\tBest Valid (Loss: %.5f Acc : %.4f)' 
                % (self.history['epoch'], self.history['dev_loss'], self.history['dev_acc']))
        test_file = os.path.join(self.exp_path, 'QNLI.tsv')
        test_results = self.decode(test_loader, labeled=False, add_loss=False)
        self.logger.info('Start writing test predictions to file %s ...' % (test_file))
        write_results(test_results, test_file)

    def decode(data_loader, labeled=True, add_loss=False):
        correct, predictions, total_loss = 0, [], 0.
        for data_zip in data_loader:
            self.model.eval()
            if labeled:
                word_ids, segment_ids, masks, labels = to_device(data_zip, self.device)
            else:
                word_ids, segment_ids, masks = to_device(data_zip, self.device)
            predict_labels = self.model.predict(word_ids, segment_ids, masks).cpu()
            if labeled:
                correct += int(np.sum(predict_labels.numpy()==labels.cpu().numpy()))
            predictions.extend(predict_labels.tolist())
            if add_loss:
                assert labeled
                self.model.train()
                loss = self.model(word_ids, segment_ids, masks, labels).cpu().item()
                if self.model.reduction == 'mean':
                    loss = loss * word_ids.size(0)
                total_loss += loss

        if labeled:
            total_size = len(data_loader.dataset)
            predictions = float(correct)/total_size
        if add_loss:
            return predictions, total_loss
        return predictions




