import numpy as np
import os
import pandas as pd
import torch
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
import time
from data_utils import makedirs, load_sst, get_args
from model import LSTMSentiment


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def train_model(train_iter, valid_iter, inputs, outputs, args):
    model = LSTMSentiment(args)
    if args["word_vectors"]:
        model.embed.weight.data = inputs.vocab.vectors
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    opt = O.Adam(model.parameters())

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    all_break = False
    header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
    dev_log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
    log_template = ' '.join(
        '{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    makedirs(args["save_path"])
    print(header)

    for epoch in range(args["epochs"]):
        if all_break:
            break
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for i, batch in enumerate(train_iter):
            model.train()
            opt.zero_grad()
            iterations += 1

            # forward pass
            answer = model(batch)

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(answer, 1)
                          [1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total

            loss = criterion(answer, batch.label)
            loss.backward()
            opt.step()

        #       if iterations % args["save_every"] == 0:
        #         snapshot_prefix = os.path.join(args["save_path"], 'snapshot')
        #         snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)

        #         torch.save(model, snapshot_path)

            if iterations % args["dev_every"] == 0:
                model.eval()
                valid_iter.init_epoch()
                dev_acc = 0
                n_dev_correct, dev_loss = 0, 0
                for dev_batch_idx, dev_batch in enumerate(valid_iter):
                    answer = model(dev_batch)
                    n_dev_correct += (torch.max(answer, 1)[1].view(
                        dev_batch.label.size()).data == dev_batch.label.data).sum()
                    dev_loss = criterion(answer, dev_batch.label)
                dev_acc = 100. * n_dev_correct / len(valid_iter)

                print(dev_log_template.format(time.time() - start,
                                              epoch, iterations, 1 +
                                              i, len(train_iter),
                                              100. * (1 + i) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))

                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    snapshot_prefix = os.path.join(
                        args["save_path"], 'best_snapshot')
                    snapshot_path = snapshot_prefix + \
                        '_devacc_{}_devloss_{}_iter_{}_model.pt'.format(
                            dev_acc, dev_loss.data[0], iterations)

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model.state_dict(), snapshot_path)
                if iterations == 10000:
                    return model
                #           print( os.getcwd() )
                #           print( os.listdir() )
                #           files.download(snapshot_path)

            elif iterations % args["log_every"] == 0:
                # print progress message
                print(log_template.format(time.time() - start,
                                          epoch, iterations, 1 +
                                          i, len(train_iter),
                                          100. * (1 + i) / len(train_iter), loss.data[0], ' ' * 8, n_correct / n_total * 100, ' ' * 12))
    return model


args = get_args()


torch.cuda.set_device(0)
inputs, outputs, train_iter, valid_iter, train, valid = load_sst(args)
args["n_embed"] = len(inputs.vocab)
args["d_out"] = len(outputs.vocab)
model = train_model(train_iter, valid_iter, inputs, outputs, args)
