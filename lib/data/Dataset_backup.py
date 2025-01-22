from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import  lib as lb


class Dataset(object):
    def __init__(self, data, batchSize, cuda, name= None, eval=False, gold=None):
        self.src  = data["src"]
        self.tgt  = data["tgt"]
        self.pos  = data["pos"]
        self.data = name
        print("Dataset",self.data)

        if self.data in ['train_pg']:
          self.label = data["label"]
          self.gold  = gold
        assert(len(self.src) == len(self.tgt))
        self.cuda = cuda

        self.batchSize = batchSize
        self.numBatches = math.ceil(len(self.src)/batchSize)
        self.eval = eval

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]   #  here data is len BS obtained from original data, where each sequence is of different length
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(lb.Constants.PAD)  # Now each input is of shape BS * SL as all sequence padeded to same lenght
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths   # Shape out=BS * SL, lenghts=BS (1 d tensor containing lenght of each seq)
        else:
            return out    # BS * SL

    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)

        
        srcBatch, lengths = self._batchify(self.src[index*self.batchSize:(index+1)*self.batchSize],
            include_lengths=True)
        if self.data in ['train_pg'] and self.gold is not None:
             labels=self.label[index*self.batchSize:(index+1)*self.batchSize]
             assert len(labels) == len(srcBatch)

        tgtBatch = self._batchify(self.tgt[index*self.batchSize:(index+1)*self.batchSize])

        # within batch sort by decreasing length.
        indices = range(len(srcBatch)) # 0 to (BS-1) i.e indices for 64  complete sentences as each sentence is padded to be fixed length 
        if self.data in ['train_pg'] and self.gold is not None:
             batch = zip(indices, labels, srcBatch, tgtBatch)
             batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
             indices, labels, srcBatch, tgtBatch = zip(*batch)  # indices here are sorted depending on lenght of sentence within this batch
        else :
             batch = zip(indices, srcBatch, tgtBatch)
             batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
             indices, srcBatch, tgtBatch = zip(*batch)  # indices here are sorted depending on lenght of sentence within this batch

        def wrap(b):
            b = torch.stack(b, 0).t().contiguous() # transposed output is SL * BS
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.eval)
            return b
        if self.data in ['train_pg'] and self.gold is not None:
           return (wrap(srcBatch), lengths, labels), wrap(tgtBatch), indices
        return (wrap(srcBatch), lengths), wrap(tgtBatch), indices # lenght will be used by pad pak to tkonw how much padding is in SL and is sorted index based on lenght
                                                                  #indices is NO of sentences in a batch and  these indices are sorted based on length 

    def __len__(self):
        return self.numBatches

    def shuffle(self):
        data = list(zip(self.src, self.tgt, self.pos))
        random.shuffle(data)
        self.src, self.tgt, self.pos = zip(*data)

    def restore_pos(self, sents):
        sorted_sents = [None] * len(self.pos)
        for sent, idx in zip(sents, self.pos):
          sorted_sents[idx] = sent
        return sorted_sents
