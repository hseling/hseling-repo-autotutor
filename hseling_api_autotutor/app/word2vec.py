from tqdm.auto import tqdm
import numpy as np
import os
import json
from .database import redis_db
# vectors = []

word2vec = {}
BASE = os.path.dirname(os.path.abspath(__file__))

with open (os.path.join(BASE,"recommendation_logic", "lyashevskaya_freq_dict.json") , "r", encoding="utf-8") as f:
    lyashevskaya_freq_dict = json.load(f)
sorted_lyash = [k for k, v in sorted(lyashevskaya_freq_dict.items(), key=lambda item: item[1], reverse=True)]
# print(sorted_lyash[:10])
sorted_lyash = set(sorted_lyash[:int(len(lyashevskaya_freq_dict)/2)])

word2vec_file = open(os.path.join(BASE, "recommendation_logic", "processing_data", 'cc.ru.300.vec'))

n_words, embedding_dim = word2vec_file.readline().split()
n_words, embedding_dim = int(n_words), int(embedding_dim)

# Zero vector for PAD
# vectors.append(np.zeros((1, embedding_dim)))

progress_bar = tqdm(desc='Read word2vec', total=n_words)

while True:

    line = word2vec_file.readline().strip()

    if not line:
        break

    current_parts = line.split()

    current_word = ' '.join(current_parts[:-embedding_dim])

    if current_word in sorted_lyash:

        current_vectors = current_parts[-embedding_dim:]
        # current_vectors = np.array(list(map(float, current_vectors)))
        # current_vectors = np.expand_dims(current_vectors, 0)
        word2vec[current_word] = current_vectors
        # word_key = "w2v_" + current_word
        # check_existese = redis_db.lrange(word_key, 0, 1)
        # if len(check_existese) == 0 or check_existese is None:
        #     for w2v_digit in current_vectors:
        #         redis_db.rpush(word_key, w2v_digit)
        # else:
        #     print("word2vec already initialized. Skip")
        #     return

    # vectors.append(current_vectors)
    # break

    progress_bar.update(1)

progress_bar.close()

word2vec_file.close()

