import os
import time
import tensorflow as tf

from make_dataset import char2id, id2char, dataset
from RNN_model import RNN, OneStep, loss, embedding_dim, rnn_units


attempt = 2
EPOCHS = 100

model = RNN(
    # Be sure the vocabulary size matches the `StringLookup` layers.
    vocab_size=len(char2id.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = f'Attempt{attempt}/training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

one_step_model = OneStep(model, id2char, char2id)

model.save(f'Attempt{attempt}/final_model')

tf.saved_model.save(one_step_model, f'Attempt{attempt}/one_step')

#####################
# Generating data
#####################

start = time.time()
states = None
next_char = tf.constant(['This was written by a bot\n'])
result = [next_char]

for n in range(10000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)


################
# Saving output
################
f= open(f"Attempt{attempt}/Generated_Text.txt","w+")
f.write(result[0].numpy().decode('utf-8'))
f.close()
