# LSTM Contextual Decomposition

Reproducing [Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs](https://arxiv.org/abs/1801.05453)

To run our scripts, you first have to train your LSTM with the torchtext preloaded data:
```
python train.py
```

Then we can run our CD algorithm on a few selected dissenting subphrases from our validation set. Change the value of either `CD_parse` or `CD_word` (depending on if you want CD scores of subphrases between commas, or word-level CD scores).
```
python CD.py
```

In this, we find dissenting phrases as phrases with a confidence score below a certain threshold (1.5 in this case), which we have inferred from the paper (when they mention "absolute score of 1.5").