import numpy
import time

import gzip

import shuffle
from util import load_dict

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

class TextIterator:
    """Simple Multitext iterator."""
    def __init__(self, datasets,
                 dicts,
                 n_words_dicts=None,
                 batch_size=128,
                 maxlen=100,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 factors=1,
                 outputs=1,
                 maxibatch_size=20):

        if shuffle_each_epoch:
            self.datasets_orig = datasets
            self.datasets = shuffle.main(datasets, temporary=True)
        else:
            self.datasets = [fopen(fp, 'r') for fp in datasets]

        self.dicts = []
        for dict_ in dicts:
            self.dicts.append(load_dict(dict_))

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.factors = factors
        self.outputs = outputs

        assert len(datasets) == 1 + outputs, 'Datasets and dictionaries mismatch'

        self.n_words_dicts = n_words_dicts

        if self.n_words_dicts:
            for d, max_ in zip(self.dicts, self.n_words_dicts):
                for key, idx in d.items():
                    if idx >= max_:
                        del d[key]

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.buffers = [[] for _ in range(len(datasets))]
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.datasets = shuffle.main(self.datasets_orig, temporary=True)
        else:
            for dataset in self.datasets:
                dataset.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        # creating list of empty for later filling
        batches = [[] for _ in range(len(self.datasets))]
        # fill buffer, if it's empty
        len_buffers = [len(buffer) for buffer in self.buffers]

        assert (min(len_buffers) == max(len_buffers)), 'Buffer size mismatch!'


        if min(len_buffers) == 0:
            print 'feeding buffer'
            start = time.time()
            for sss in zip(*self.datasets): # now the datasets are in a list, parallel iteration with zip()
                # forcing same length over buffers
                new_sss = []
                for idx, ss in enumerate(sss):
                    ss = ss.split()
                    if self.skip_empty and len(ss) == 0:
                        break
                    if len(ss) > self.maxlen:
                        break
                    new_sss.append(ss)
                if len(new_sss) == len(self.buffers):
                    [self.buffers[idx].append(ss) for idx, ss in enumerate(new_sss)]

                if len(self.buffers[0]) == self.k:
                    break

            if len(self.buffers[0]) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.buffers[1]])
                tidx = tlen.argsort()

                for idx, buffer in enumerate(self.buffers):
                    _buf = [buffer[i] for i in tidx]
                    self.buffers[idx] = _buf

            else:
                for buffer in self.buffers:
                    buffer.reverse()
            print "elapsed time: {}".format(time.time()-start)

        start = time.time()
        try:
            # actual work here
            buffer_empty = False
            while not buffer_empty:
                # read from source file and map to word index
                for idx, buffer in enumerate(self.buffers):
                    try:
                        ss = buffer.pop()
                    except IndexError:
                        buffer_empty = True
                        break
                    tmp = []
                    for w in ss:
                        if idx == 0:
                            if self.factors > 1:
                                w = [self.dicts[i][f] if f in self.dicts[i] else 1 for (i,f) in enumerate(w.split('|'))]
                            else:
                                w = [self.dicts[idx][w] if w in self.dicts[idx] else 1]
                            tmp.append(w)
                        else:
                            w = [self.dicts[idx][w] if w in self.dicts[idx] else 1]
                            tmp.extend(w)
                    ss = tmp
                    batches[idx].append(ss)

                len_batches = [len(batch) for batch in batches]

                if any(x > self.batch_size for x in len_batches):
                    break

        except IOError:
            self.end_of_data = True

        print 'returned in {}'.format(start-time.time())
        return batches
