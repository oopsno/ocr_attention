# encoding: UTF-8

import torch
import torch.nn as nn
from torch.nn.functional import tanh, softmax
from torch.nn.parameter import Parameter


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, out_channels):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, out_channels)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        S, N, C = recurrent.size()
        t_rec = recurrent.view(S * N, C)
        output = self.embedding(t_rec)
        output = output.view(S, N, -1)
        return output


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=None):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size
        self.input_size = input_size
        if num_embeddings is None:
            num_embeddings = 0
        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)

    def forward(self, prev_hidden, feats, *cur_embeddings):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size

        feats_proj = self.i2h(feats.view(-1, nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1, nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(
            -1, hidden_size)
        emition = self.score(tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT, nB).transpose(0, 1)

        alpha = softmax(emition)  # nB * nT
        context = (feats * alpha.transpose(0, 1).contiguous().view(nT, nB, 1).expand(nT, nB, nC)).sum(0).squeeze(
            0)  # nB * nC
        if cur_embeddings:
            context = torch.cat([context] + cur_embeddings, 1)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_embeddings=None):
        super().__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_embeddings)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.generator = nn.Linear(hidden_size, num_classes)
        self.num_embeddings = num_embeddings
        if num_embeddings is not None:
            self.char_embeddings = Parameter(torch.randn(num_classes + 1, num_embeddings))

    def forward(self, feats, text_length, text):
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size
        assert (input_size == nC)
        assert (nB == text_length.numel())

        num_steps = text_length.data.max()
        num_labels = text_length.data.sum()
        if self.num_embeddings is not None:
            targets = torch.zeros(nB, num_steps + 1).long().cuda()
            start_id = 0
            for i in range(nB):
                targets[i][1:1 + text_length.data[i]] = text.data[start_id:start_id + text_length.data[i]] + 1
                start_id = start_id + text_length.data[i]
            targets = targets.transpose(0, 1).contiguous()

        output_hiddens = torch.zeros(num_steps, nB, hidden_size).type_as(feats.data)
        hidden = torch.zeros(nB, hidden_size).type_as(feats.data)
        max_locs = torch.zeros(num_steps, nB)
        max_vals = torch.zeros(num_steps, nB)
        for i in range(num_steps):
            if self.num_embeddings is not None:
                cur_embeddings = self.char_embeddings.index_select(0, targets[i])
                hidden, alpha = self.attention_cell(hidden, feats, cur_embeddings)
            else:
                hidden, alpha = self.attention_cell(hidden, feats)
            output_hiddens[i] = hidden
            if self.processed_batches % 500 == 0:
                max_val, max_loc = alpha.data.max(1)
                max_locs[i] = max_loc.cpu()
                max_vals[i] = max_val.cpu()
        if self.processed_batches % 500 == 0:
            print('max_locs', list(max_locs[0:text_length.data[0], 0]))
            print('max_vals', list(max_vals[0:text_length.data[0], 0]))
        new_hiddens = torch.zeros(num_labels, hidden_size).type_as(feats.data)
        b = 0
        start = 0
        for length in text_length.data:
            new_hiddens[start:start + length] = output_hiddens[0:length, b, :]
            start = start + length
            b = b + 1
        probs = self.generator(new_hiddens)
        return probs


class CRNN(nn.Module):
    def __init__(self, image_height, num_class, hidden_size, image_channels=1, num_embeddings=None):
        super(CRNN, self).__init__()
        assert image_height % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),  # 256x8x25
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 256x4x25
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),  # 512x4x25
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),  # 512x2x25
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True))  # 512x1x25
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.attention = Attention(hidden_size, hidden_size, num_class, num_embeddings)

    def forward(self, x, length, *text):
        # conv features
        x = self.cnn(x)
        N, C, H, W = x.size()
        if H != 1:
            raise ValueError('Invalid feature map shape: excepted [N, C, 1, W], got {}'.format(x.size()))
        x = x.squeeze(2).permute(2, 0, 1)

        # rnn features
        rnn = self.rnn(x)
        output = self.attention(rnn, length, *text)

        return output
