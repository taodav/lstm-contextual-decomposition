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



def Lsig3(y1, y2, y3):
  y1_cont = 0.5 * (sigmoid(y1 + y3) - sigmoid(y3) +
                   sigmoid(y1 + y2 + y3) - sigmoid(y2 + y3))
  y2_cont = 0.5 * (sigmoid(y2 + y3) - sigmoid(y3) +
                   sigmoid(y1 + y2 + y3) - sigmoid(y1 + y3))
  y3_cont = sigmoid(y3)
  return y1_cont, y2_cont, y3_cont


def Ltanh3(y1, y2, y3):
  y1_cont = 0.5 * (tanh(y1 + y3) - tanh(y3) +
                   tanh(y1 + y2 + y3) - tanh(y2 + y3))
  y2_cont = 0.5 * (tanh(y2 + y3) - tanh(y3) +
                   tanh(y1 + y2 + y3) - tanh(y1 + y3))
  y3_cont = tanh(y3)
  return y1_cont, y2_cont, y3_cont


def Ltanh2(y1, y2):
  y1_cont = 0.5 * (tanh(y1) + tanh(y1 + y2) - tanh(y2))
  y2_cont = 0.5 * (tanh(y2) + tanh(y2 + y1) - tanh(y1))
  return y1_cont, y2_cont

class CD:
    def __init__(self, model, inputs, clf, data, vectorizer, valid_data):
        self.model = model
        self.inputs = inputs
        self.data = data
        self.valid_data = valid_data
        self.vectorizer = vectorizer
        # self.generate_data(data)
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

    # def generate_data(self, data):
        # valid_data = []
        # print("generating data")
        # for d in data.items():
        #     df = d[1].text
        #     text = []
        #     # this is every batch
        #     for i in range(df.shape[1]):
        #         text = []
        #         for j in range(df[:, i].shape[0]):
        #             if int(df[j, i]) != 1:
        #                 text.append(self.inputs.vocab.itos[int(df[j, i])])
        #         valid_data.append(text)
        # self.valid_data = valid_data

    #we want to return the CD score W*hc[T]
    def context_decomp(self, batch, start, stop):

        # these are the weights and bias learnt for the gates for each layer during LSTM
        weights = self.model.lstm.state_dict()

        # refer variable names to paper and http://pytorch.org/docs/master/nn.html
        # split equally into 4 (150,300)
        Wi, Wf, Wg, Wo = np.split(weights['weight_ih_l0'], 4, 0)
        Vi, Vf, Vg, Vo = np.split(weights['weight_hh_l0'], 4, 0)
        # split equally into 4 (150,)
        bi, bf, bg, bo = np.split(weights['bias_ih_l0'].cpu(
        ).numpy() + weights['bias_hh_l0'].cpu().numpy(), 4)

        #word embedding model as specified in LSTMSentiment
        word_embedding = self.model.embed(batch.text)[:, 0].data
        #T = number of time steps
        T = word_embedding.size(0)

        #initialize beta, beta_c, gamma, gamma_c all to zeeros
        #contributions of given phrase / elements outside of given phrase made to cell state
        Bc = np.zeros((T, self.model.hidden_dim))
        Gc = np.zeros((T, self.model.hidden_dim))

        #contributions of given phrase / elements outside of given phrase made to hidden state
        B = np.zeros((T, self.model.hidden_dim))
        G = np.zeros((T, self.model.hidden_dim))

        #temp variables: prev_B = B_(t-1), prev_G = G_(t-1)
        prev_B = np.zeros(self.model.hidden_dim)
        prev_G = np.zeros(self.model.hidden_dim)

        #assume we have a way of write each of the gates in eq 2 3 4 as linear sum of contributions from each of their inputs
        #recursively compute the decomposition using linearizing activation functions: see section 3.2.1
        for i in range(T):
            if i != 0:
                prev_B = B[i - 1]
                prev_G = G[i - 1]

            Bi = np.dot(Vi, prev_B)
            Bg = np.dot(Vg, prev_B)
            Bf = np.dot(Vf, prev_B)
            Bo = np.dot(Vo, prev_B)
            Gi = np.dot(Vi, prev_G)
            Gg = np.dot(Vg, prev_G)
            Gf = np.dot(Vf, prev_G)
            Go = np.dot(Vo, prev_G)

            #if the current time step is contained within the phrase, get what was let through
            if i >= start and i <= stop:
                Bi = Bi + np.dot(Wi, word_embedding[i])
                Bg = Bg + np.dot(Wg, word_embedding[i])
                Bf = Bf + np.dot(Wf, word_embedding[i])
                Bo = Bo + np.dot(Wo, word_embedding[i])

            #if the current time step is NOT contained in the phrase, get what was let though
            else:
                Gi += np.dot(Wi, word_embedding[i])
                Gg += np.dot(Wg, word_embedding[i])
                Gf += np.dot(Wf, word_embedding[i])
                Go += np.dot(Wo, word_embedding[i])

            Bi_cont, Gi_cont, bi_cont = Lsig3(Bi, Gi, bi)
            Bg_cont, Gg_cont, bg_cont = Ltanh3(Bg, Gg, bg)

            Bc[i] = Bi_cont * (Bg_cont + bg_cont) + bi_cont * Bg_cont
            Gc[i] = Gi_cont * (Bg_cont + Gg_cont + bg_cont) + \
                (Bi_cont + bi_cont) * Gg_cont

            #if the current time step is contained within the phrase
            if i >= start and i < stop:
                Bc[i] += bi_cont * bg_cont

            #if the current time step is NOT contained in the phrase
            else:
                Gc[i] += bi_cont * bg_cont

            if i != 0:
                Bf_cont, Gf_cont, bf_cont = Lsig3(Bf, Gf, bf)
                Bc[i] += (Bf_cont + bf_cont) * Bc[i - 1]
                Gc[i] += (Bf_cont + Gf_cont + bf_cont) * \
                    Gc[i - 1] + Gf_cont * Bc[i - 1]

            o = sigmoid(
                np.dot(Wo, word_embedding[i]) + np.dot(Vo, prev_B + prev_G) + bo)

            Bo_cont, Go_cont, bo_cont = Lsig3(Bo, Go, bo)
            new_Bh, new_Gh = Ltanh2(Bc[i], Gc[i])

            B[i] = o * new_Bh
            G[i] = o * new_Gh

        scores = np.dot(self.model.hidden_to_label.weight.data, B[T - 1])
        return scores[0] - scores[1]

    def grab_phrase(self, num):
        text = []
        for i in self.data[num].text:
            text.append(self.inputs.vocab.itos[int(i)])
        return text

    def CD_word(self, num):
        res = []
        text = self.grab_phrase(num)
        for i, word in enumerate(text):
            rel_calc = self.context_decomp(self.data[num], i, i)
            print(rel_calc, word)
            res.append((rel_calc, word))
        return res

    def generate_dissenting(self, clf):
        print("generating dissenting subphrases")
        self.dissenting = [i for i, val in enumerate(clf.decision_function(
            self.vectorizer.transform(self.valid_data))) if abs(val) < 1.5]

    def splits_and_CD(self, idx):
        text = self.grab_phrase(idx)
        commas = [i for i, v in enumerate(text) if v == ","] + [len(text)]
        splits = [(0, commas[0] - 1)]
        for i in range(len(commas) - 1):
            splits.append((commas[i] + 1, commas[i + 1] - 1))
        return splits

    def CD_phrase(self, num):
        splits = self.splits_and_CD(num)
        text = self.grab_phrase(num)
        res = []
        for (i, j) in splits:
            rel_calc = self.context_decomp(self.data[num], i, j)
            phrase = " ".join(text[i:j + 1])
            print(rel_calc, phrase)
            res.append((rel_calc, phrase))
        return res

    def generate_score_tuple(self, batch, text, start, end):
        return (" ".join(text[start:end]), self.context_decomp(batch, start, end - 1))

    # CD_subphrases takes in the index and start/end of dissenting subphrase within the sentence
    # returns an array of CD score before, of the phrase itself, and after
    def CD_diss_subphrases(self, batch, start, end):
        text = []
        for i in batch.text:
            text.append(self.inputs.vocab.itos[int(i)])
        res_text = [
            self.generate_score_tuple(batch, text, 0, start),
            self.generate_score_tuple(batch, text, start, end + 1),
            self.generate_score_tuple(batch, text, end + 1, len(text))
        ]
        return res_text

    def CD_negating_subphrases(self, batch, first_start, first_end, second_start, second_end):
        text = []
        for i in batch.text:
            text.append(self.inputs.vocab.itos[int(i)])
        res_text = {
            "negation_phrase": self.generate_score_tuple(batch, text, first_start, second_end + 1),
            "negation_term": self.generate_score_tuple(batch, text, first_start, first_end),
            "negated_phrase": self.generate_score_tuple(batch, text, second_start, second_end + 1),
            "overall": self.generate_score_tuple(batch, text, 0, len(text))
        }
        return res_text



def get_batches(batch_nums, train_iterator, dev_iterator, dset='train'):
    print('getting batches...')
    data_iterator = train_iterator
    # pick data_iterator
    if dset == 'train':
        data_iterator = train_iterator
    elif dset == 'valid':
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


# batch_nums = list(range(6920))
# data = get_batches(batch_nums, train_iter, valid_iter)
# cd = CD(model, inputs, clf, data)

# cd.CD_phrase(10)
