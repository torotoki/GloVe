import argparse
import codecs
import numpy as np
import matplotlib.pyplot as plt

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--vocab_file', default='vocab.txt', type=str)
  parser.add_argument('--vectors_file', default='vectors.txt', type=str)
  args = parser.parse_args()

  with open(args.vocab_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]
  with open(args.vectors_file, 'r') as f:
    vectors = {}
    for line in f:
      vals = line.rstrip().split(' ')
      vectors[vals[0]] = [float(x) for x in vals[1:]]

  vocab_size = len(words)
  vocab = {w: idx for idx, w in enumerate(words)}
  ivocab = {idx: w for idx, w in enumerate(words)}

  vector_dim = len(vectors[ivocab[0]])
  W = np.zeros((vocab_size, vector_dim))
  for word, v in vectors.items():
    if word == '<unk>':
      continue
    W[vocab[word], :] = v

  # normalize each word vector to unit length
  W_norm = np.zeros(W.shape)
  d = (np.sum(W ** 2, 1) ** (0.5))
  W_norm = (W.T / d).T
  evaluate_vectors(W_norm, vocab, ivocab)

def evaluate_vectors(W, vocab, ivocab):
  """
  Evaluate the trained word vectors on
  'Japanese Word Similarity Dataset'
  """
  filenames = [
    'score_noun.csv', 'score_verb.csv', 'score_adj.csv', 'score_adv.csv'
    #'score_noun.csv'
  ]
  prefix = './eval/data/'

  # to avoid memory overflow, could be increased/descreased
  # depending on system and vocab_size
  split_size = 100

  count_tot = 0   # count all questions
  full_count = 0  # count all questions, including those with unknown words
  error = 0.
  for i in range(len(filenames)):
    filename = filenames[i]
    with codecs.open('%s/%s' % (prefix, filename), 'r', 'shift_jis') as f:
      full_data = [line.rstrip().split(',') for line in f]
      full_count += len(full_data)
      data = [x for x in full_data if all(x[i] in vocab for i in [0,1])]

    indices = np.array([[vocab[word] for word in row[0:2]] for row in data])
    ind1, ind2 = indices.T  # first words indices, second words indices

    x,y = [], []
    predictions = np.zeros((len(indices),))
    num_iter = int(np.ceil(len(indices) / float(split_size)))
    for j in range(num_iter):
      subset = np.arange(j*split_size, min((j+1)*split_size, len(ind1)))

      for i in subset:
        # cosine similarity if input W has been normalized
        dist = np.dot(W[ind1[i], :],  W[ind2[i], :])
        gold = float(data[i][2]) / 10  # labels are on a scale of 1 to 10
        #print("word1: %s \tword2: %s \tgold: %.2f \tscore: %.2f" % (data[i][0], data[i][1], gold, dist))
        error += (gold-dist)**2
        x.append(dist)
        y.append(gold)

    print("total # of all data, including those with unknown words:", full_count)
    print("total # of data:", len(data))
    print("total error:  %f" % np.sqrt(error / len(data)))
    plt.scatter(x,y, label=filename, linewidths="0.01")
    #b,m = np.polynomial.polynomial.polyfit(x, y, 1)
    #X_line = np.linspace(-1,1,100)
    #plt.plot(X_line, b + m*X_line, '-')
  #plt.xlim([0,1])
  #plt.ylim([0,1])
  plt.xlabel("estimated value")
  plt.ylabel("gold label")
  plt.legend()
  plt.show()

if __name__ == '__main__':
  main()

