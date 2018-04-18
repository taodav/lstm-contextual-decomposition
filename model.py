import numpy as np
import os
import pandas as pd
import torch
from argparse import ArgumentParser
import torch.nn as nn
from torch.autograd import Variable
from torchtext import data, datasets
import torch.nn.functional as F
import torch.optim as O
import time

# initializing and training LSTM
# reference from https://github.com/clairett/pytorch-sentiment-classification


class LSTMSentiment(nn.Module):

    def __init__(self, args):
        super(LSTMSentiment, self).__init__()
        self.hidden_dim = args["d_hidden"]
        self.vocab_size = args["n_embed"]
        self.emb_dim = args["d_embed"]
        self.num_out = args["d_out"]
        self.batch_size = args["batch_size"]
        self.use_gpu = True  # config.use_gpu
        self.num_labels = 2
        self.embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.lstm = nn.LSTM(input_size=self.emb_dim,
                            hidden_size=self.hidden_dim)
        self.hidden_to_label = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, batch):
        if self.use_gpu:
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()),
                           Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim).cuda()))
        else:
            self.hidden = (Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)),
                           Variable(torch.zeros(1, batch.text.size()[1], self.hidden_dim)))

        vecs = self.embed(batch.text)
        lstm_out, self.hidden = self.lstm(vecs, self.hidden)
        logits = self.hidden_to_label(lstm_out[-1])
        log_probs = F.log_softmax(logits)
        #     return logits, log_probs
        return logits
