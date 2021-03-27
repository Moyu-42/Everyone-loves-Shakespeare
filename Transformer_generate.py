import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import Transformer


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
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        while len(enc_input[0]) != src_len:
            enc_input[0].append(0)
        while len(dec_input[0]) != tgt_len:
            dec_input[0].append(0)
            dec_output[0].append(0)

        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


def get_data():
    sentences = []
    with open("shakespeare") as f:
        while 1:
            line = f.readline()
            if not line:
                break
            if line == '\n':
                continue
            line = line.replace('.', ' .')
            line = line.replace(',', ' ,')
            line = line.replace(':', ' :')
            line = line.replace('?', ' ?')
            line = line.replace(';', ' ;')
            line = line.replace('\n', '')
            enc_input = line + ' P'
            dec_input = 'S ' + line
            dec_output = line + ' E'
            sentences.append([enc_input, dec_input, dec_output])
    return sentences


def greedy_search(model, enc_input, start_symbol):
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol
        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[i]
        next_symbol = next_word.item()
    return dec_input


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
        src_len = max(src_len, len(line[0].split()))
        tgt_len = max(tgt_len, len(line[1].split()))
        for token in line[0].split():
            if token not in src_vocab:
                src_vocab[token] = src
                src += 1
        for token in line[1].split():
            if token not in tgt_vocab:
                tgt_vocab[token] = tgt
                tgt += 1
        for token in line[2].split():
            if token not in tgt_vocab:
                tgt_vocab[token] = tgt
                tgt += 1

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_len, tgt_len)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 64, True)

    # model = torch.load('model.pt')

    model = Transformer.Transformer(src_vocab_size, tgt_vocab_size).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: {} loss = {:.6f}".format(epoch, loss))

    torch.save(model, 'model.pt')

    enc_inputs = torch.randint(0, tgt_vocab_size, (1, 1))[0]
    greedy_dec_input = greedy_search(model, enc_inputs[0].view(1, -1).cuda(), start_symbol=tgt_vocab['S'])
    predict, _, _, _ = model(enc_inputs[0].view(1, -1).cuda(), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    new_phrase = [idx2word[n.item()] for n in predict.squeeze()]
    print(new_phrase)
    sentence = ""
    for it in new_phrase:
        if it == 'E':
            continue
        sentence += ' ' + it
    print(sentence)
