#coding=utf8
import os, sys, argparse, time
import urllib.request

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    # 'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    # 'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    # 'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
    # 'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
    # 'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
    # 'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
}

START_TIME = time.time()

def get_vocabs_and_models(names):
    names = names.split(',')
    tasks = []
    if 'all' in names:
        for each in PRETRAINED_VOCAB_ARCHIVE_MAP:
            tasks.append((each, PRETRAINED_VOCAB_ARCHIVE_MAP[each], PRETRAINED_MODEL_ARCHIVE_MAP[each]))
    else:
        for each in names:
            each = each.strip()
            if each in PRETRAINED_VOCAB_ARCHIVE_MAP:
                tasks.append((each, PRETRAINED_VOCAB_ARCHIVE_MAP[each], PRETRAINED_MODEL_ARCHIVE_MAP[each]))
            else:
                print('[WARNING]: Unknown bert model name %s' % (each))
    return tasks

def hook(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - START_TIME)
    speed_str = (" Speed: %s" % format_size(speed)).ljust(20)
    recv_size = blocknum * blocksize
    f = sys.stdout
    pervent = recv_size / totalsize
    percent_str = "Progress: %.2f%%" % (pervent * 100)
    n = round(pervent * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(percent_str.ljust(18, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    f.write('\r')
 
def format_size(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("Format passed is wrong!")
        return "Error"
    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%.3fG" % (G)
        else:
            return "%.3fM" % (M)
    else:
        return "%.3fK" % (kb)

def download(url, data_dir, proxy=None):
    if proxy:
        proxy = urllib.request.ProxyHandler({'http': proxy})
        opener = urllib.request.build_opener(proxy)
        urllib.request.install_opener(opener)
    file_path = os.path.join(data_dir, url.split('/')[-1])
    if os.path.exists(file_path):
        print('File %s already exists in %s, just skip!' % (os.path.basename(file_path), os.path.dirname(file_path)))
    else:
        print('Start downloading %s to diretory %s ...' % (os.path.basename(file_path), os.path.dirname(file_path)))
        global START_TIME
        START_TIME = time.time()
        urllib.request.urlretrieve(url, file_path, hook)
        print('\nFile %s has been finished!' % (os.path.basename(file_path)))

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='directory to save data to', type=str, default='data/.cache/')
    parser.add_argument('--model', help='bert vocab and model to download from remote url, a comma separated string',
                        type=str, default='all', choices=['all', 'bert-base-uncased', 'bert-base-cased', 'bert-large-uncased', 'bert-large-cased'])
    parser.add_argument('--proxy', type=str, default='https://127.0.0.1:1080', help='use proxy since those urls are not available in some area')
    args = parser.parse_args(arguments)

    if not os.path.isdir(args.data_dir):
        os.mkdir(args.data_dir)
    tasks = get_vocabs_and_models(args.model)

    if args.proxy:
        print('Download all files using proxy %s ...' % (args.proxy))
    for name, vocab, model in tasks:
        download(vocab, args.data_dir, proxy=args.proxy)
        download(model, args.data_dir, proxy=args.proxy)
    print('All vocabs and models have been downloaded!!!')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))