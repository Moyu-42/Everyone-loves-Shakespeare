from gensim.models import word2vec
import re

def load_txt():
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
            line = ['<BOS>'] + line
            sentences.append(line)
    return sentences

if __name__ == '__main__':
    sentences = load_txt()
    num_features = 200    # Word vector dimensionality
    num_workers = 16       # Number of threads to run in parallel
    downsampling = 1e-3   # Downsample setting for frequent words
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
            vector_size=num_features, sg=1, sample=downsampling, min_count=1)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('ran')
