import torch
import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self):
        """
        """
        super(Word2Vec, self).__init__()

    def forward(self):
        pass


if __name__ == '__main__':
    embeddings = nn.Embedding(3, 10)
    print(embeddings)
    print(embeddings(0))


