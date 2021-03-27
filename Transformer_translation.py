import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import random
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
    sentences = [
        ['When I do count the clock that tells the time , P', 'S When I do count the clock that tells the time ,', 'When I do count the clock that tells the time , E'],
        ['And see the brave day sunk in hideous night ; P', 'S And see the brave day sunk in hideous night ;', 'And see the brave day sunk in hideous night ; E'],
        ['When I behold the violet past prime , P', 'S When I behold the violet past prime ,', 'When I behold the violet past prime , E'],
        ['And sable curls all silver\'d o\'er with white : P', 'S And sable curls all silver\'d o\'er with white :', 'And sable curls all silver\'d o\'er with white : E'],
        ['When lofty trees I see barren of leaves , P', 'S When lofty trees I see barren of leaves ,', 'When lofty trees I see barren of leaves , E'],
        ['Which erst from heat did canopy the herd , P', 'S Which erst from heat did canopy the herd ,', 'Which erst from heat did canopy the herd , E'],
        ['And summer\'s green, all girded up in sheaves , P', 'S And summer\'s green, all girded up in sheaves ,', 'And summer\'s green, all girded up in sheaves , E'],
        ['Born on the bier with white and bristly beard ; P', 'S Born on the bier with white and bristly beard ;', 'Born on the bier with white and bristly beard ; E'],
        ['Then of thy beauty do I question make , P', 'S Then of thy beauty do I question make ,', 'Then of thy beauty do I question make , E'],
        ['That thou among the wastes of time must go , P', 'S That thou among the wastes of time must go ,', 'That thou among the wastes of time must go , E'],
        ['Since sweets and beauties do themselves forsake , P', 'S Since sweets and beauties do themselves forsake ,', 'Since sweets and beauties do themselves forsake , E'],
        ['And die as fast as they see others grow ; P', 'S And die as fast as they see others grow ;', 'And die as fast as they see others grow ; E'],
        ['And nothing \'gainst Time\'s scythe can make defence ; P', 'S And nothing \'gainst Time\'s scythe can make defence ;', 'And nothing \'gainst Time\'s scythe can make defence ; E'],
        ['Save breed, to brave him when he takes thee hence . P', 'S Save breed, to brave him when he takes thee hence .', 'Save breed, to brave him when he takes thee hence . E'],
        ['Shall I compare thee to a summer\'s day ? P', 'S Shall I compare thee to a summer\'s day ?', 'Shall I compare thee to a summer\'s day ? E'],
        ['Thou art more lovely and more temperate : P', 'S Thou art more lovely and more temperate :', 'Thou art more lovely and more temperate : E'],
        ['Rough winds do shake the darling buds of May , P', 'S Rough winds do shake the darling buds of May ,', 'Rough winds do shake the darling buds of May , E'],
        ['And summer\'s lease hath all too short a date : P', 'S And summer\'s lease hath all too short a date :', 'And summer\'s lease hath all too short a date : E'],
        ['Sometime too hot the eye of heaven shines , P', 'S Sometime too hot the eye of heaven shines ,', 'Sometime too hot the eye of heaven shines , E'],
        ['And often is his gold complexion dimm\'d ; P', 'S And often is his gold complexion dimm\'d ;', 'And often is his gold complexion dimm\'d ; E'],
        ['And every fair from fair sometime declines , P', 'S And every fair from fair sometime declines ,', 'And every fair from fair sometime declines , E'],
        ['By chance or nature\'s changing course untrimm\'d , P', 'S By chance or nature\'s changing course untrimm\'d ,', 'By chance or nature\'s changing course untrimm\'d , E'],
        ['But thy eternal summer shall not fade , P', 'S But thy eternal summer shall not fade ,', 'But thy eternal summer shall not fade , E'],
        ['Nor lose possession of that fair thou owest ; P', 'S Nor lose possession of that fair thou owest ;', 'Nor lose possession of that fair thou owest ; E'],
        ['Nor shall death brag thou wander\'st in his shade , P', 'S Nor shall death brag thou wander\'st in his shade ,', 'Nor shall death brag thou wander\'st in his shade , E'],
        ['When in eternal lines to time thou growest : P', 'S When in eternal lines to time thou growest :', 'When in eternal lines to time thou growest : E'],
        ['So long as men can breathe or eyes can see , P', 'S So long as men can breathe or eyes can see ,', 'So long as men can breathe or eyes can see , E'],
        ['So long lives this and this gives life to thee . P', 'S So long lives this and this gives life to thee .', 'So long lives this and this gives life to thee . E'],
        ['Cupid laid by his brand , and fell asleep : P', 'S Cupid laid by his brand , and fell asleep :', 'Cupid laid by his brand , and fell asleep : E'],
        ['A maid of Dian\'s this advantage found , P', 'S A maid of Dian\'s this advantage found ,', 'A maid of Dian\'s this advantage found , E'],
        ['And his love-kindling fire did quickly steep , P', 'S And his love-kindling fire did quickly steep ,', 'And his love-kindling fire did quickly steep , E'],
        ['In a cold valley-fountain of that ground ; P', 'S In a cold valley-fountain of that ground ;', 'In a cold valley-fountain of that ground ; E'],
        ['Which borrow\'d from this holy fire of Love , P', 'S Which borrow\'d from this holy fire of Love ,', 'Which borrow\'d from this holy fire of Love , E'],
        ['A dateless lively heat , still to endure , P', 'S A dateless lively heat , still to endure ,', 'A dateless lively heat , still to endure , E'],
        ['And grew a seething bath , whichyet men prove , P', 'S And grew a seething bath , whichyet men prove ,', 'And grew a seething bath , whichyet men prove , E'],
        ['Against strange maladies a sovereign cure . P', 'S Against strange maladies a sovereign cure .', 'Against strange maladies a sovereign cure . E'],
        ['But at my mistress\' eye Love\'s brand new-fired , P', 'S But at my mistress\' eye Love\'s brand new-fired ,', 'But at my mistress\' eye Love\'s brand new-fired , E'],
        ['The boy for trial needs would touch my breast ; P', 'S The boy for trial needs would touch my breast ;', 'The boy for trial needs would touch my breast ; E'],
        ['I , sick withal , the help of bath desired , P', 'S I , sick withal , the help of bath desired ,', 'I , sick withal , the help of bath desired , E'],
        ['And thither hied , a sad distemper\'d guest , P', 'S And thither hied , a sad distemper\'d guest ,', 'And thither hied , a sad distemper\'d guest , E'],
        ['But found no cure : the bath for my help lies , P', 'S But found no cure : the bath for my help lies ,', 'But found no cure : the bath for my help lies , E'],
        ['Where Cupid got new fire--my mistress\' eyes . P', 'S Where Cupid got new fire--my mistress\' eyes .', 'Where Cupid got new fire--my mistress\' eyes . E'],
        ['When I do count the clock that tells the time , P', 'S When I do count the clock that tells the time ,', 'When I do count the clock that tells the time , E'],
        ['And see the brave day sunk in hideous night ; P', 'S And see the brave day sunk in hideous night ;', 'And see the brave day sunk in hideous night ; E'],
        ['When I behold the violet past prime , P', 'S When I behold the violet past prime ,', 'When I behold the violet past prime , E'],
        ['And sable curls all silver\'d o\'er with white : P', 'S And sable curls all silver\'d o\'er with white :', 'And sable curls all silver\'d o\'er with white : E'],
        ['When lofty trees I see barren of leaves , P', 'S When lofty trees I see barren of leaves ,', 'When lofty trees I see barren of leaves , E'],
        ['Which erst from heat did canopy the herd , P', 'S Which erst from heat did canopy the herd ,', 'Which erst from heat did canopy the herd , E'],
        ['And summer\'s green , all girded up in sheaves , P', 'S And summer\'s green , all girded up in sheaves ,', 'And summer\'s green , all girded up in sheaves , E'],
        ['Born on the bier with white and bristly beard ; P', 'S Born on the bier with white and bristly beard ;', 'Born on the bier with white and bristly beard ; E'],
        ['Then of thy beauty do I question make , P', 'S Then of thy beauty do I question make ,', 'Then of thy beauty do I question make , E'],
        ['That thou among the wastes of time must go , P', 'S That thou among the wastes of time must go ,', 'That thou among the wastes of time must go , E'],
        ['Since sweets and beauties do themselves forsake , P', 'S Since sweets and beauties do themselves forsake ,', 'Since sweets and beauties do themselves forsake , E'],
        ['And die as fast as they see others grow ; P', 'S And die as fast as they see others grow ;', 'And die as fast as they see others grow ; E'],
        ['And nothing \'gainst Time\'s scythe can make defence P', 'S And nothing \'gainst Time\'s scythe can make defence', 'And nothing \'gainst Time\'s scythe can make defence E'],
        ['Save breed , to brave him when he takes thee hence . P', 'S Save breed , to brave him when he takes thee hence .', 'Save breed , to brave him when he takes thee hence . E']
    ]

    # Padding
    src_vocab = dict()
    tgt_vocab = dict()
    src = 0
    tgt = 0
    src_len = 0
    tgt_len = 0
    for li in sentences:
        src_len = max(src_len, len(li[0].split()))
        tgt_len = max(tgt_len, len(li[1].split()))
        for token in li[0].split():
            if token not in src_vocab:
                src_vocab[token] = src
                src += 1
        for token in li[1].split():
            if token not in tgt_vocab:
                tgt_vocab[token] = tgt
                tgt += 1
        for token in li[2].split():
            if token not in tgt_vocab:
                tgt_vocab[token] = tgt
                tgt += 1

    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences, src_len, tgt_len)
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    model = Transformer.Transformer(src_vocab_size, tgt_vocab_size).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    for epoch in range(15):
        for enc_inputs, dec_inputs, dec_outputs in loader:
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))
            print("Epoch: {} loss = {:.6f}".format(epoch, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    enc_inputs = torch.randint(0, tgt_vocab_size, (1, 1))[0]
    print(idx2word[enc_inputs[0].item()])
    greedy_dec_input = greedy_search(model, enc_inputs[0].view(1, -1).cuda(), start_symbol=tgt_vocab['S'])
    predict, _, _, _ = model(enc_inputs[0].view(1, -1).cuda(), greedy_dec_input)
    predict = predict.data.max(1, keepdim=True)[1]
    # print(enc_inputs[0], '->', [idx2word[n.item()] for n in predict.squeeze()])
    new_phrase = [idx2word[n.item()] for n in predict.squeeze()]
    sentence = ""
    for it in new_phrase:
        if it == 'E':
            break
        sentence += ' ' + it
    print(sentence)
