import tensorflow as tf
import numpy as np
import os

# file_name = "bee movie script.txt"
folder_path = "C:/Users/nthnt/PycharmProjects/TextGenerator/Datasets/" + "Bionicle_Stories/"
# get names of multiple text files as dataset
data_names = os.listdir(folder_path)

# reading in text dataset
raw_texts = []
for txt in data_names:
    with open(folder_path + txt, 'rb') as f:
        single_text = f.read().decode(encoding='utf-8')
    #print(f"Length of text: {len(single_text)} characters")
    raw_texts.append(single_text)

print(f"Length of all text: {sum([len(words) for words in raw_texts])} characters")

# get unique characters
vocab = sorted(set("".join(raw_texts)))
print(f'{len(vocab)} unique characters')

char2id = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=list(vocab), mask_token=None)

id2char = tf.keras.layers.experimental.preprocessing.StringLookup(
    vocabulary=char2id.get_vocabulary(), invert=True, mask_token=None)


# sequence of characters to readable words
def char2text(ids):
    return tf.strings.reduce_join(id2char(ids), axis=-1)

# turn text dataset into integers
text_id = char2id(tf.strings.unicode_split("".join(raw_texts), 'UTF-8'))
#print(text_id)
ids_dataset = tf.data.Dataset.from_tensor_slices(text_id)

#for ids in ids_dataset.take(10):
 #   print(id2char(ids).numpy().decode('utf-8'))

seq_length = 100
examples_per_epoch = len(raw_texts)//(seq_length+1)

# splits text into sequences of length 101
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

split_input_target(list("Tensorflow"))
dataset = sequences.map(split_input_target)


# Batch size
BATCH_SIZE = 256

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))



