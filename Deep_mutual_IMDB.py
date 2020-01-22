import torch
from torchtext import data
from torchtext import datasets

SEED = 1234

#torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize = 'spacy', include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)

from torchtext import datasets

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

import random

train_data, valid_data = train_data.split(random_state = random.seed(0))

MAX_VOCAB_SIZE = 25_000

TEXT.build_vocab(train_data,
                 max_size = MAX_VOCAB_SIZE,
                 vectors = "glove.6B.100d",
                 unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch=True,
    device=device,
    shuffle=True)

import torch.nn as nn

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)
        self.init_weights()
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]

        embedded = self.dropout(self.embedding(text))

        # embedded = [sent len, batch size, emb dim]

        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        # hidden = [batch size, hid dim * num directions]

        return self.fc(hidden)

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

pretrained_embeddings = TEXT.vocab.vectors

import torch.optim as optim

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum() / len(correct)
    return acc

import torch.nn.functional as F
from torch.autograd import Variable

def train(mutual_model, iterator, optimizer, criterion):
    epoch_loss = []
    epoch_acc = []
    for i in range(model_num):
        epoch_loss.append(0)
        epoch_acc.append(0)

    for batch in iterator:
        text, text_lengths = batch.text
        outputs = []
        for i in range(model_num):
            model = mutual_model[i]
            model.train()
            predictions = model(text, text_lengths).squeeze(1)
            outputs.append(predictions)

        for i in range(model_num):
            ce_loss = criterion(outputs[i], batch.label)
            kl_loss = 0
            for j in range(model_num):
                if i != j:
                    kl_loss += loss_kl(F.log_softmax(outputs[i]),
                                       F.softmax(Variable(outputs[j])))
            #print(kl_loss)
            loss = ce_loss + kl_loss / (model_num - 1)

            acc = binary_accuracy(outputs[i], batch.label)

            optimizer[i].zero_grad()
            loss.backward()
            optimizer[i].step()

            epoch_loss[i] += loss.item()
            epoch_acc[i] += acc.item()

    for i in range(model_num):
        epoch_loss[i] /= len(iterator)
        epoch_acc[i] /= len(iterator)
    return epoch_loss, epoch_acc


def evaluate(mutual_model, iterator, criterion):
    epoch_loss = []
    epoch_acc = []
    for i in range(model_num):
        epoch_loss.append(0)
        epoch_acc.append(0)

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            outputs = []
            for i in range(model_num):
                model = mutual_model[i]
                model.eval()
                predictions = model(text, text_lengths).squeeze(1)
                outputs.append(predictions)

            for i in range(model_num):
                ce_loss = criterion(outputs[i], batch.label)
                kl_loss = 0
                for j in range(model_num):
                    if i != j:
                        kl_loss += loss_kl(F.log_softmax(outputs[i]),
                                           F.softmax(Variable(outputs[j])))
                loss = ce_loss + kl_loss / (model_num - 1)

                acc = binary_accuracy(outputs[i], batch.label)

                epoch_loss[i] += loss.item()
                epoch_acc[i] += acc.item()

    for i in range(model_num):
        epoch_loss[i] /= len(iterator)
        epoch_acc[i] /= len(iterator)
    return epoch_loss, epoch_acc

def in_train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        text, text_lengths = batch.text

        predictions = model(text, text_lengths).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def in_evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text

            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 8
model_num = 2
mutual_model = []
mutual_optim = []

best_valid_loss = []
criterion = nn.BCEWithLogitsLoss()

in_model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)

in_model.embedding.weight.data.copy_(pretrained_embeddings)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
in_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
in_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
in_best_valid_loss = float('inf')
in_optimizer = torch.optim.Adam(in_model.parameters(), lr=0.001)
in_model.cuda()

for i in range(model_num):
    model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                BIDIRECTIONAL,
                DROPOUT,
                PAD_IDX)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    model.cuda()
    mutual_model.append(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mutual_optim.append(optimizer)
    best_valid_loss.append(float('inf'))

loss_kl = nn.KLDivLoss(reduction='batchmean')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss, train_acc = train(mutual_model, train_iterator, mutual_optim, criterion)
    valid_loss, valid_acc = evaluate(mutual_model, valid_iterator, criterion)

    in_train_loss, in_train_acc = in_train(in_model, train_iterator, in_optimizer, criterion)
    in_valid_loss, in_valid_acc = in_evaluate(in_model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    for i in range(model_num):
        if valid_loss[i] < best_valid_loss[i]:
            best_valid_loss[i] = valid_loss[i]
            torch.save(mutual_model[i].state_dict(), f'mutual_model{i}.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    for i in range(model_num):
        print(f"Model{i}")
        print(f'\tTrain Loss: {train_loss[i]:.3f} | Train Acc: {train_acc[i] * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss[i]:.3f} |  Val. Acc: {valid_acc[i] * 100:.2f}%')

    if in_valid_loss < in_best_valid_loss:
        in_best_valid_loss = in_best_valid_loss
        torch.save(in_model.state_dict(), 'in-model.pt')
    print("독립")
    print(f'\tTrain Loss: {in_train_loss:.3f} | Train Acc: {in_train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {in_valid_loss:.3f} |  Val. Acc: {in_valid_acc * 100:.2f}%')

for i in range(model_num):
    mutual_model[i].load_state_dict(torch.load(f'mutual_model{i}.pt'))
    test_loss, test_acc = evaluate(mutual_model, test_iterator, criterion)
    print(f'Test<m{i}> Loss: {test_loss[i]:.3f} | Test<m{i}> Acc: {test_acc[i] * 100:.2f}%')

print("독립")
in_model.load_state_dict(torch.load('in-model.pt'))
in_test_loss, in_test_acc = in_evaluate(in_model, test_iterator, criterion)
print(f'Test Loss: {in_test_loss:.3f} | Test Acc: {in_test_acc*100:.2f}%')