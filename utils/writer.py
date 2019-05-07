#coding=utf8
import sys, logging

def set_logger(log_path, testing=False, noStdout=False):
    logFormatter = logging.Formatter('%(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    if testing:
        fileHandler = logging.FileHandler('%s/log_test.txt' % (log_path), mode='w')
    else:
        fileHandler = logging.FileHandler('%s/log_train.txt' % (log_path), mode='w') # override written
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    if not noStdout:
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)
    return logger

def write_results(results, file_path):
    with open(file_path, 'w') as outfile:
        outfile.write('index\tprediction\n')
        for idx, prob in enumerate(results):
            label = 'entailment' if prob > 0.5 else 'not_entailment'
            if prob == 0.5:
                print('[WARNING]: index of %d is ambiguous!' % (idx))
            outfile.write(str(idx) + '\t' + label + '\n')
