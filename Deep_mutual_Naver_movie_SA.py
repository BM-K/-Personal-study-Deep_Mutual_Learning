import time
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from konlpy.tag import Mecab
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator
from torchtext.vocab import Vectors
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

    # input_dim = len(TEXT.vocab)
    # embedding_dim = 160 # kr-data 벡터 길이
    # hidden_dim = 256
    # output_dim = 1 # sentiment analysis


def tokenizer1(text):
    result_text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》;]', '', text)
    a = Mecab().morphs(result_text)
    return ([a[i] for i in range(len(a))])


class RNN(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim,
                 batch_size, dropout, pad_idx):
        super().__init__()
        vocab = TEXT.vocab

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=2, bidirectional=True)
        self.init_weights()

        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, hidden):
        # text = [sent len, batch size]
        embedded = self.embedding(text)
        # embedded = [sent len, batch size, emb dim]

        output, hidden = self.gru(embedded, hidden)
        # output = [sent len, batch size, hid dim*2(numlayer)]
        # hidden = [4, batch size, hid dim]

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # batch_dim x hid_dim*2     

        return torch.sigmoid(self.fc(hidden))

    def init_hidden(self):
        result = Variable(torch.rand(4, self.batch_size, self.hidden_dim))
        if use_cuda:
            return result.cuda()
        else:
            return result

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


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(mutual_model, iterator, optimizer, criterion):
    epoch_loss = []
    epoch_acc = []
    for i in range(model_num):
        epoch_loss.append(0)
        epoch_acc.append(0)

    for batch in iterator:
        outputs = []
        for i in range(model_num):
            model = mutual_model[i]
            model.train()
            hidden = model.init_hidden()
            predictions = model(batch.text, hidden).squeeze(1)
            outputs.append(predictions)

        for i in range(model_num):
            batch.label = batch.label.float()
            ce_loss = criterion(outputs[i], batch.label)
            kl_loss = 0
            for j in range(model_num):
                if i != j:
                    kl_loss += loss_kl(F.log_softmax(outputs[i]),
                                       F.softmax(Variable(outputs[j])))
            loss = ce_loss + kl_loss / (model_num - 1)

            acc = binary_accuracy(outputs[i], batch.label)

            optimizer[i].zero_grad()
            loss.backward(retain_graph=True)
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
            outputs = []

            for i in range(model_num):
                model = mutual_model[i]
                model.eval()
                hidden = model.init_hidden()
                predictions = model(batch.text, hidden).squeeze(1)
                outputs.append(predictions)

            for i in range(model_num):
                batch.label = batch.label.float()
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

        hidden = model.init_hidden()

        predictions = model(batch.text, hidden).squeeze(1)

        batch.label = batch.label.float()
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
            hidden = model.init_hidden()

            predictions = model(batch.text, hidden).squeeze(1)

            batch.label = batch.label.float()
            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

ID = data.Field(sequential=False,
                use_vocab=False)

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=tokenizer1,
                  lower=True,
                  batch_first=False,
                  fix_length=20,
                  )

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True,
                   )
SEED = 1234
torch.manual_seed(SEED)
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data, val_data = TabularDataset.splits(
    path='.', train='naver_movie_train.txt', test='naver_movie_test.txt',
    validation='naver_movie_eval.txt', format='tsv',
    fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True
) # train과 text파일이 현재 디렉토리에 있어햐함. 여기서 tsv로 구분된 것 사용. 첫 번째 header는 무시

batch_size = 500
train_loader = BucketIterator(dataset=train_data, batch_size=batch_size, device=device, shuffle=True)
test_loader = BucketIterator(dataset=test_data, batch_size=batch_size, device=device, shuffle=True)
val_loader = BucketIterator(dataset=val_data, batch_size=batch_size, device=device, shuffle=True)

vectors = Vectors(name="kr-projected.txt")

TEXT.build_vocab(train_data, vectors=vectors, min_freq=5, max_size=15000)

if __name__ == '__main__':
    model_num = 3
    mutual_model = []
    mutual_optim = []

    input_dim = len(TEXT.vocab)
    embedding_dim = 160  # kr-data 벡터 길이
    hidden_dim = 256
    output_dim = 1
    dropout = 0.5
    N_EPOCHS = 6
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    criterion = nn.BCELoss()
    best_valid_loss = []
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    in_model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, batch_size, dropout, PAD_IDX)
    in_model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    in_model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
    in_model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)
    in_model.cuda()
    in_optimizer = torch.optim.Adam(in_model.parameters(), lr=0.001)
    in_best_valid_loss = float('inf')

    for i in range(model_num):
        model = RNN(input_dim, embedding_dim, hidden_dim, output_dim, batch_size, dropout, PAD_IDX)
        model.embedding.weight.data.copy_(TEXT.vocab.vectors)
        model.embedding.weight.data[UNK_IDX] = torch.zeros(embedding_dim)
        model.embedding.weight.data[PAD_IDX] = torch.zeros(embedding_dim)
        model.cuda()
        mutual_model.append(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        mutual_optim.append(optimizer)
        best_valid_loss.append(float('inf'))

    loss_kl = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train(mutual_model, train_loader, mutual_optim, criterion)
        valid_loss, valid_acc = evaluate(mutual_model, val_loader, criterion)
        in_train_loss, in_train_acc = in_train(in_model, train_loader, in_optimizer, criterion)
        in_valid_loss, in_valid_acc = in_evaluate(in_model, val_loader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        for i in range(model_num):
            if valid_loss[i] < best_valid_loss[i]:
                best_valid_loss[i] = valid_loss[i]
                torch.save(mutual_model[i].state_dict(), f'mutual_K_model{i}.pt')

        if in_valid_loss < in_best_valid_loss:
            in_best_valid_loss = in_valid_loss
            torch.save(in_model.state_dict(), 'one-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        for i in range(model_num):
            print(f"Model{i}")
            print(f'\tTrain Loss: {train_loss[i]:.3f} | Train Acc: {train_acc[i]* 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss[i]:.3f} |  Val. Acc: {valid_acc[i]* 100:.2f}%')
        print("독립")
        print(f'\tTrain Loss: {in_train_loss:.3f} | Train Acc: {in_train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {in_valid_loss:.3f} |  Val. Acc: {in_valid_acc * 100:.2f}%')

    for i in range(model_num):
        mutual_model[i].load_state_dict(torch.load(f'mutual_K_model{i}.pt'))
        test_loss, test_acc = evaluate(mutual_model, test_loader, criterion)
        print(f'Test<m{i}> Loss: {test_loss[i]:.3f} | Test<m{i}> Acc: {test_acc[i] * 100:.2f}%')
    print("독립")
    in_model.load_state_dict(torch.load('one-model.pt'))
    in_test_loss, in_test_acc = in_evaluate(in_model, test_loader, criterion)
    print(f'Test Loss: {in_test_loss:.3f} | Test Acc: {in_test_acc * 100:.2f}%')
