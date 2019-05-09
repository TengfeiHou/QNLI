#coding=utf8
import sys, logging, coloredlogs

def set_logger(log_path, testing=False):
    # %(asctime)s,%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s'
    logger = logging.getLogger('mylogger')
    coloredlogs.install(logger=logger, fmt='%(asctime)s %(message)s')
    logFormatter = logging.Formatter('%(asctime)s %(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger.setLevel(logging.DEBUG)
    if testing:
        fileHandler = logging.FileHandler('%s/log_test.txt' % (log_path), mode='w')
    else:
        fileHandler = logging.FileHandler('%s/log_train.txt' % (log_path), mode='w') # override written
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    return logger

def write_results(results, file_path):
    with open(file_path, 'w') as outfile:
        outfile.write('index\tprediction\n')
        for idx, prob in enumerate(results):
            label = 'entailment' if prob > 0.5 else 'not_entailment'
            if prob == 0.5:
                print('[WARNING]: index of %d is ambiguous!' % (idx))
            outfile.write(str(idx) + '\t' + label + '\n')
