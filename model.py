import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


class CNNSentenceClassifier(nn.Module):

    def __init__(self, opt, vocab, pre_trained_embeddings=None):
        super(CNNSentenceClassifier, self).__init__()

        if pre_trained_embeddings is not None:
            self.num_embeddings, self.embedding_dim = pre_trained_embeddings.shape
        else:
            self.num_embeddings = vocab.get_vocab_size("tokens")
            self.embedding_dim = opt['emb_dim']
        self.embedding_mode = opt['mode']
        self.FILTER_SIZES = opt['FILTER_SIZES']
        self.dropout_rate = opt['dropout_rate']
        self.output_size = vocab.get_vocab_size("labels")

        padding_idx = vocab.get_token_index(vocab._padding_token)
        self.embed = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=padding_idx)

        if self.embedding_mode != 'rand':
            self.embed.weight.data.copy_(torch.from_numpy(pre_trained_embeddings))

        if self.embedding_mode in ('static', 'multichannel'):
            self.embed.weight.requires_grad = False

        if self.embedding_mode == 'multichannel':
            self.embed_multi = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=0)
            self.embed_multi.weight.data.copy_(torch.from_numpy(pre_trained_embeddings))
            self.in_channels = 2
        else:
            self.in_channels = 1

        for filter_size in self.FILTER_SIZES:
            conv = nn.Conv1d(self.in_channels, 100, self.embedding_dim * filter_size, stride=self.embedding_dim)
            setattr(self, 'conv_' + str(filter_size), conv)

        self.fc = nn.Linear(len(self.FILTER_SIZES) * 100, self.output_size)

    def forward(self, sentence):
        batch_size = sentence.size()[0]
        sentence_len = sentence.size()[1]

        x = self.embed(sentence).view(batch_size, 1, -1)
        if self.embedding_mode == 'multichannel':
            x_multi = self.embed_multi(sentence).view(batch_size, 1, -1)
            x = torch.cat((x, x_multi), 1)

        conv_result = [
            F.max_pool1d(F.relu(getattr(self, 'conv_' + str(filter_size))(x)), sentence_len - filter_size + 1).view(-1,
                                                                                                                    100)
            for filter_size in self.FILTER_SIZES]

        x = torch.cat(conv_result, 1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        result = self.fc(x)

        return result













