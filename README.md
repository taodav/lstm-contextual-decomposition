# LSTM Contextual Decomposition

Reproducing [Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs](https://arxiv.org/abs/1801.05453)

To begin, install all the dependencies in `requirements.txt`. Most importantly, you'll need Pytorch with cuda support.

To run our scripts, please run the files in the demo jupyter notebook labelled:
```
CD_demo.ipynb
```

It should contain all scripts required to return all results we've managed to reproduce.

In this, we find dissenting phrases as phrases with a confidence score below a certain threshold (1.5 in this case), which we have inferred from the paper (when they mention "absolute score of 1.5").

Reference to reproducibility results in the paper labelled `final report 550.docx` in the report folder.
