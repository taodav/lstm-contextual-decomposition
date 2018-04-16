import numpy as np
import os
import pandas as pd
import torch
import time
from data_utils import load_sst, get_args
from model import LSTMSentiment
from logit import logit_clf
from scipy.special import expit as sigmoid
from numpy import *
import random
from scipy.special import expit as sigmoid


args = get_args()

inputs, outputs, train_iter, valid_iter, train, valid = load_sst(args)

args["n_embed"] = len(inputs.vocab)
args["d_out"] = len(outputs.vocab)

clf, vectorizer = logit_clf(inputs, outputs)

model = LSTMSentiment(args)
model.cuda()
model.load_state_dict(torch.load(
    "results/best_snapshot.pt"))

class CD:
    def __init__(self, model, inputs, clf, data, vectorizer):
        self.model = model
        self.inputs = inputs
        self.data = data
        self.vectorizer = vectorizer
        self.generate_data(data)
        self.generate_dissenting(clf)
        self.clf = clf

    def decomp_three(self, a, b, c, activation):
        a_contrib = 0.5 * (activation(a + c) - activation(c) +
                           activation(a + b + c) - activation(b + c))
        b_contrib = 0.5 * (activation(b + c) - activation(c) +
                           activation(a + b + c) - activation(a + c))
        return a_contrib, b_contrib, activation(c)

    def decomp_tanh_two(self, a, b):
        return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))

    def generate_data(self, data):
        valid_data = []
        for d in self.data.items():
            text = []
            for i in d[1].text:
                text.append(inputs.vocab.itos[int(i)])
            valid_data.append(text)
        self.valid_data = valid_data

    def context_decomp(self, batch, start, stop):
        weights = self.model.lstm.state_dict()

        # Index one = word vector (i) or hidden state (h), index two = gate
        W_ii, W_if, W_ig, W_io = np.split(weights['weight_ih_l0'], 4, 0)
        W_hi, W_hf, W_hg, W_ho = np.split(weights['weight_hh_l0'], 4, 0)
        b_i, b_f, b_g, b_o = np.split(weights['bias_ih_l0'].cpu(
        ).numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)
        word_vecs = self.model.embed(batch.text)[:, 0].data
        T = word_vecs.size(0)
        relevant = np.zeros((T, self.model.hidden_dim))
        irrelevant = np.zeros((T, self.model.hidden_dim))
        relevant_h = np.zeros((T, self.model.hidden_dim))
        irrelevant_h = np.zeros((T, self.model.hidden_dim))
        for i in range(T):
            if i > 0:
                prev_rel_h = relevant_h[i - 1]
                prev_irrel_h = irrelevant_h[i - 1]
            else:
                prev_rel_h = np.zeros(self.model.hidden_dim)
                prev_irrel_h = np.zeros(self.model.hidden_dim)

            rel_i = np.dot(W_hi, prev_rel_h)
            rel_g = np.dot(W_hg, prev_rel_h)
            rel_f = np.dot(W_hf, prev_rel_h)
            rel_o = np.dot(W_ho, prev_rel_h)
            irrel_i = np.dot(W_hi, prev_irrel_h)
            irrel_g = np.dot(W_hg, prev_irrel_h)
            irrel_f = np.dot(W_hf, prev_irrel_h)
            irrel_o = np.dot(W_ho, prev_irrel_h)

            if i >= start and i <= stop:
                rel_i = rel_i + np.dot(W_ii, word_vecs[i])
                rel_g = rel_g + np.dot(W_ig, word_vecs[i])
                rel_f = rel_f + np.dot(W_if, word_vecs[i])
                rel_o = rel_o + np.dot(W_io, word_vecs[i])
            else:
                irrel_i = irrel_i + np.dot(W_ii, word_vecs[i])
                irrel_g = irrel_g + np.dot(W_ig, word_vecs[i])
                irrel_f = irrel_f + np.dot(W_if, word_vecs[i])
                irrel_o = irrel_o + np.dot(W_io, word_vecs[i])

            rel_contrib_i, irrel_contrib_i, bias_contrib_i = self.decomp_three(
                rel_i, irrel_i, b_i, sigmoid)
            rel_contrib_g, irrel_contrib_g, bias_contrib_g = self.decomp_three(
                rel_g, irrel_g, b_g, np.tanh)

            relevant[i] = rel_contrib_i * \
                (rel_contrib_g + bias_contrib_g) + \
                bias_contrib_i * rel_contrib_g
            irrelevant[i] = irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (
                rel_contrib_i + bias_contrib_i) * irrel_contrib_g

            if i >= start and i < stop:
                relevant[i] += bias_contrib_i * bias_contrib_g
            else:
                irrelevant[i] += bias_contrib_i * bias_contrib_g

            if i > 0:
                rel_contrib_f, irrel_contrib_f, bias_contrib_f = self.decomp_three(
                    rel_f, irrel_f, b_f, sigmoid)
                relevant[i] += (rel_contrib_f +
                                bias_contrib_f) * relevant[i - 1]
                irrelevant[i] += (rel_contrib_f + irrel_contrib_f + bias_contrib_f) * \
                    irrelevant[i - 1] + irrel_contrib_f * relevant[i - 1]

            o = sigmoid(
                np.dot(W_io, word_vecs[i]) + np.dot(W_ho, prev_rel_h + prev_irrel_h) + b_o)
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = self.decomp_three(
                rel_o, irrel_o, b_o, sigmoid)
            new_rel_h, new_irrel_h = self.decomp_tanh_two(
                relevant[i], irrelevant[i])
            #relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
            #irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
            relevant_h[i] = o * new_rel_h
            irrelevant_h[i] = o * new_irrel_h

        W_out = self.model.hidden_to_label.weight.data

        # Sanity check: scores + irrel_scores should equal the LSTM's output minus model.hidden_to_label.bias
        scores = np.dot(W_out, relevant_h[T - 1])
        irrel_scores = np.dot(W_out, irrelevant_h[T - 1])

        return scores, irrel_scores

    def CD_word(self, num):
        res = []
        for i, word in enumerate(self.valid_data[self.dissenting[num]]):
            rel, irr = self.context_decomp(
                self.data[self.dissenting[num]], i, i)
            rel_calc = round(rel[0] - rel[1], 3)
            phrase = word
            print(rel_calc, word)
            res.append((rel_calc, word))
        return res

    def generate_dissenting(self, clf):
        self.dissenting = [i for i, val in enumerate(clf.decision_function(
            self.vectorizer.transform(self.valid_data))) if abs(val) < 1.5]

    def splits_and_CD(self, idx):
        text = self.valid_data[self.dissenting[idx]]
        start = 0
        end = 0
        commas = [i for i, v in enumerate(text) if v == ","] + [len(text)]
        splits = [(0, commas[0] - 1)]
        for i in range(len(commas) - 1):
            splits.append((commas[i] + 1, commas[i + 1] - 1))
        return splits

    def CD_phrase(self, num):
        splits = self.splits_and_CD(num)
        res = []
        for (i, j) in splits:
            rel, irr = self.context_decomp(
                self.data[self.dissenting[num]], i, j)
            rel_calc = round(rel[0] - rel[1], 3)
            phrase = " ".join(self.valid_data[self.dissenting[num]][i:j + 1])
            print(rel_calc, phrase)
            res.append((rel_calc, phrase))
        return res


def get_batches(batch_nums, train_iterator, dev_iterator, dset='train'):
    print('getting batches...')
    np.random.seed(13)
    random.seed(13)

#     # pick data_iterator
#     if dset=='train':
#         data_iterator = train_iterator
#     elif dset=='dev':
#         data_iterator = dev_iterator

    data_iterator = dev_iterator

    # actually get batches
    num = 0
    batches = {}
    data_iterator.init_epoch()
    for batch_idx, batch in enumerate(data_iterator):
        if batch_idx == batch_nums[num]:
            batches[batch_idx] = batch
            num += 1

        if num == max(batch_nums):
            break
        elif num == len(batch_nums):
            print('found them all')
            break
    return batches


batch_nums = list(range(6920))
data = get_batches(batch_nums, train_iter, valid_iter)
cd = CD(model, inputs, clf, data, vectorizer)

cd.CD_phrase(10)
