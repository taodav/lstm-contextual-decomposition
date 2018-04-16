import os
import torch
from torchtext import data, datasets

def get_args():
    EPOCHS = 20
    USE_GPU = torch.cuda.is_available()
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 150
    BATCH_SIZE = 50

    config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "d_embed": EMBEDDING_DIM,
        "d_hidden": HIDDEN_DIM,
        'word_vectors': "glove.6B.300d",
        "vector_cache": os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'),
        "save_path": "results",
        "dev_every": 1000,
        "save_every": 1000,
        "log_every": 1000
    }

    return config

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os
    import errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise

# load in Stanford Sentiment Treebank data
# Our first task should be to train an LSTM on this data


def load_sst(args):
    # first create two fields for data, text and labels. Check https://github.com/pytorch/text#data for an example
    text = data.Field()
    labels = data.Field(sequential=False, unk_token=None)

    # we first build our dataset with all subtrees to build our vocab
    train_v, valid_v, test_v = datasets.SST.splits(
        text, labels, train_subtrees=True, filter_pred=lambda ex: ex.label != 'neutral')

    text.build_vocab(train_v, valid_v, test_v)
    if args["word_vectors"]:
        if os.path.isfile(args["vector_cache"]):
            text.vocab.vectors = torch.load(args["vector_cache"])
        else:
            text.vocab.load_vectors(args["word_vectors"])
            makedirs(os.path.dirname(args["vector_cache"]))
            torch.save(text.vocab.vectors, args["vector_cache"])
        labels.build_vocab(train_v)

    # Next we build our datasets without all subtrees
    train, valid, test = datasets.SST.splits(text, labels, fine_grained=False, train_subtrees=False,
                                             filter_pred=lambda ex: ex.label != 'neutral')

    train_iter, valid_iter, test_iter = data.BucketIterator.splits(
        (train, valid, test), batch_size=1, device=0)

    return text, labels, train_iter, valid_iter, train, valid
