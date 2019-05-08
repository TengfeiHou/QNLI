#coding=utf8

def set_hyperparam_path(options):
    task_path = 'exp/task_QNLI__%s' % (options.bert)

    exp_name = ''
    exp_name += 'optim_%s__' % (options.optim)
    if options.optim == 'bertadam':
        exp_name += 'warmup_%s__' % (options.warmup)
        exp_name += 'schedule_%s__' % (options.schedule)
    exp_name += 'lr_%s__' % (option.lr)
    exp_name += 'l2_%s__' % (options.weight_decay)
    exp_name += 'bsize_%s__' % (options.batch_size)
    exp_name += 'loss_%s__' % (options.loss)
    exp_name += 'reduce_%s' % (options.reduction)
    exp_name += 'me_%s__' % (options.max_epoch)
    exp_name += 'smooth_%s' % (option.label_smoothing)

    return os.path.join(task_path, exp_name)