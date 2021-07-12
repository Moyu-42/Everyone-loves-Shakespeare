import torch
import torch.utils.data as Data
from torch.nn.init import xavier_uniform_
import torch.nn as nn
import torch.optim as optim
import Transformer
import numpy as np
import re
from nltk.corpus import stopwords
from gensim.models import word2vec


class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def make_data(sentences, src_len, tgt_len):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0]]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1]]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2]]]

        while len(enc_input[0]) != src_len:
            enc_input[0].append(0)
        while len(dec_input[0]) != tgt_len:
            dec_input[0].append(0)
            dec_output[0].append(0)

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

def word_vec(vocab, sentences, fg=0):
    word_vec = xavier_uniform_(torch.empty(len(vocab), 200))
    model = word2vec.Word2Vec.load('./ran')
    wv = model.wv
    for sentence in sentences:
        for word in sentence[fg]:
            word_vec[vocab[word]] = torch.tensor(wv[word])
    if fg == 1:
        word_vec[vocab['<EOS>']] = torch.tensor(wv['<EOS>'])
    return word_vec

def get_data():
    sentences = []
    with open("../shakespeare") as f:
        while 1:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                continue
            line = line.lower()
            line = re.sub(',', ' <SEP>', line)
            line = re.sub('[.?:]', ' <EOS>', line).split()
            # line = [word for word in line if word not in set(stopwords.words('english'))]
            sentences.append(line)
    enc_inputs = [enc_input for enc_input in sentences]
    dec_inputs = [['<BOS>'] + dec_input for dec_input in sentences]
    dec_outputs = [dec_output + ['<EOS>'] for dec_output in sentences]
    sentences = [val for val in zip(enc_inputs[:-1], dec_inputs[1:], dec_outputs[1:])]
    return sentences

def beam_search(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs).squeeze(0)[i]
        topk = torch.topk(projected, 8)
        prob = torch.multinomial(topk[0], 1)
        prob = topk[1][prob.item()]
        next_word = prob
        next_symbol = next_word.item()
    return dec_input[0][1:]

if __name__ == "__main__":
    sentences = get_data()
    # Padding
    src_vocab = dict()
    tgt_vocab = dict()
    src = 0
    tgt = 0
    src_len = 0
    tgt_len = 0
    for line in sentences:
        src_len = max(src_len, len(line[0]))
        tgt_len = max(tgt_len, len(line[1]))
        for token in line[0]:
            if token not in src_vocab:
                src_vocab[token] = src
                src += 1
        for token in line[1]:
            if token not in tgt_vocab:
                tgt_vocab[token] = tgt
                tgt += 1
        for token in line[2]:
            if token not in tgt_vocab:
                tgt_vocab[token] = tgt
                tgt += 1

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    wordvec_src = word_vec(src_vocab, sentences)
    wordvec_tgt = word_vec(tgt_vocab, sentences, 1)
    # print(wordvec)
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_len, tgt_len)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 64, True)

    model = torch.load('model.pt')

    # model = Transformer.Transformer(src_vocab_size, tgt_vocab_size, wordvec_src, wordvec_tgt).cuda()
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = optim.Adam(params=model.parameters(), lr=0.0003)
    # for epoch in range(60):
    #     Loss = []
    #     for enc_inputs, dec_inputs, dec_outputs in loader:
    #         enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
    #         outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
    #         loss = criterion(outputs, dec_outputs.view(-1))
    #         Loss.append(loss.item())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print("Epoch: {} loss = {:.6f}".format(epoch, np.mean(Loss)))
    # torch.save(model, 'model.pt')

    model.eval()
    raw = "And the sun shine, having no alternative, on the nothing new."
    print(raw)
    for i in range(3):
        enc_inputs = raw.lower()
        enc_inputs = enc_inputs.replace(',', ' <SEP>')
        enc_inputs = enc_inputs.replace('.', ' <EOS>')
        enc_inputs = enc_inputs.split(' ')
        enc_inputs = [enc_input for enc_input in enc_inputs if enc_input in src_vocab]
        # enc_inputs = [src_vocab[word] for word in enc_inputs if word not in set(stopwords.words('english'))]
        enc_inputs = [src_vocab[word] for word in enc_inputs]
        while len(enc_inputs) != src_len:
            enc_inputs.append(0)
        enc_inputs = torch.LongTensor(enc_inputs)
        greedy_dec_input = beam_search(model, enc_inputs.view(1, -1).cuda(), start_symbol=tgt_vocab['<BOS>'])
        new_phrase = [idx2word[n.item()] for n in greedy_dec_input.squeeze()]
        # print(new_phrase)
        sentence = ""
        for it in new_phrase:
            sentence += it + ' '
            if it == '<EOS>':
                break
        sentence = sentence.replace(' <SEP>', ',')
        sentence = sentence.replace(' <EOS>', '.')
        print(sentence)
        raw = sentence
