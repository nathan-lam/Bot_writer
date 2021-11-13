import tensorflow as tf
import os

from make_dataset import char2id, id2char
from RNN_model import RNN, OneStep, embedding_dim, rnn_units

attempt = 2
checkpoint_dir = f'Attempt{attempt}/training_checkpoints/'


latest = tf.train.latest_checkpoint(checkpoint_dir)

model = RNN(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(char2id.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
model.load_weights(latest)

one_step_model = OneStep(model, id2char, char2id)

# model = tf.keras.models.load_model(checkpoint_dir + 'final_model/')

# one_step_model = tf.saved_model.load(f'Attempt{attempt}/one_step')



#####################
# Generating data
#####################

states = None
next_char = tf.constant(['Attempt1:\n','Attempt2:\n','Attempt3:\n','Attempt4:\n','Attempt5:\n']) # seed
result = [next_char]

for n in range(500):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
print(result[0].numpy().decode("utf-8"))



################
# Saving output
################

for i in range(len(result)):
    f= open(f"Attempt{attempt}/Generated_Text_{i}.txt","w+")
    f.write(result[i].numpy().decode("utf-8"))
    f.close()
