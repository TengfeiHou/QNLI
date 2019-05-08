#coding=utf8
import os, sys, argparse, time, json
from torch.utils.data import DataLoader
from utils.dataset import QNLIDataset
from utils.batch import *
from utils.solver.solver_qnli import QNLISolver
from models.bert_wrapper import BertBinaryClassification
from utils.writer import set_logger
from utils.seed import set_random_seed
from utils.gpu import set_torch_device
from utils.hyperparam import set_hyperparam_path
from utils.optim import *

parser = argparse.ArgumentParser()
parser.add_argument('--bert', default='bert-base-uncased', choices=['bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased'])
parser.add_argument('--loss', choices=['bce', 'ce'], default='ce')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--batch_size', default=10, type=int)
parser.add_argument('--drop', default=0.1, type=float)
parser.add_argument('--optim', default=['adam', 'bertadam'], default='adam')
parser.add_argument('--reduction', default='mean', choices=['none', 'mean', 'sum'])
parser.add_argument('--label_smoothing', default=0.0, type=float)
parser.add_argument('--warmup', default=0.1, type=float, help='warmup ratio')
parser.add_argument('--schedule', default='warmup_linear', choices=['warmup_linear', 'warmup_constant', 'warmup_cosine', 'none'])
parser.add_argument('--init_weights', default=0.1, type=float)
parser.add_argument('--max_epoch', default=100, type=int)
parser.add_argument('--deviceId', type=int, default=-1, help='train model on ith gpu. -1:cpu, 0:auto_select')
parser.add_argument('--seed', type=int, default=999)
args = parser.parse_args()

exp_path = set_hyperparam_path(args)
logger = set_logger(exp_path)
device = set_torch_device(args.deviceId)
set_random_seed(args.seed, device=device)

logger.info("Parameters:" + str(json.dumps(vars(args), indent=4)))
logger.info("Experiment path: %s" % (exp_path))
logger.info(time.asctime(time.localtime(time.time())))

train_loader = DataLoader(QNLIDataset('train'), batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_labeled)
dev_loader = DataLoader(QNLIDataset('dev'), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_labeled)
test_loader = DataLoader(QNLIDataset('test'), batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_unlabeled)
logger.info('Training set size %s; Dev set size %s; Test set size %s' % (len(train_loader.dataset), len(dev_loader.dataset), len(test_loader.dataset)))

train_model = BertBinaryClassification(
    args.bert, dropout=args.dropout, bce=(args.loss == 'bce'), 
    reduction=args.reduction, label_smoothing=args.label_smoothing, device=device
)
self.train_model.init_weights(args.init_weights)
if args.optim == 'bertadam':
    t_total = int((len(train_loader.dataset) + args.batch_size - 1) / args.batch_size) * args.max_epoch
    bert_optimizer = set_bertadam_optimizer(train_model.bert_parameters(), args.lr, t_total, warmup=args.warmup, schedule=args.schedule, weight_decay=args.schedule)
    clsfy_optimizer = set_bertadam_optimizer(train_model.clsfy_parameters(), args.lr, t_total, warmup=args.warmup, schedule=args.schedule, weight_decay=args.schedule)
    all_optimizer = set_bertadam_optimizer(list(train_model.parameters()), args.lr, t_total, warmup=args.warmup, schedule=args.schedule, weight_decay=args.schedule)
else:
    bert_optimizer = set_adam_optimizer(train_model.bert_parameters(), args.lr, args.weight_decay)
    clsfy_optimizer = set_adam_optimizer(train_model.clsfy_parameters(), args.lr, args.weight_decay)
    all_optimizer = set_adam_optimizer(list(train_model.parameters()), args.lr, args.weight_decay)
optimizer = {
    'bert': bert_optimizer,
    'clsfy': clsfy_optimizer,
    'all': all_optimizer
}
train_solver = QNLISolver(train_model, optimizer, exp_path, logger, device)
logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
train_solver.train_and_decode(train_loader, dev_loader, test_loader, args.max_epoch)