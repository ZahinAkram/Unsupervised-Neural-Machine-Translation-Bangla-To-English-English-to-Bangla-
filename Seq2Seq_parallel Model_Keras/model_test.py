#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import 
from __future__ import print_function
from keras.models import Model
from keras.layers import Input, LSTM, Dense ,GaussianNoise
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


#Define Sizes
batch_size = 32  # Batch size for training.
epochs = 50  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 7000


# In[4]:


#File Directory of Monolingual Datasets 
# data_dir1 ='xgyl7qzsx2/vextors_em_eng_10k.txt'
# data_dir2 ='xgyl7qzsx2/vectors_em_bn_10k.txt'
data_path='ben.txt'


# In[5]:


#Vectorize
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()


# In[6]:


# with open(data_dir1, 'r', encoding='utf-8') as f:
#     lines = f.read().split('\n')    
# for line in lines[: min(num_samples, len(lines) - 1)]:
#     input_texts.append(line)
#     for char in line:
#         if char not in input_characters:
#             input_characters.add(char)   


# In[7]:


#Data Preprocessing English Mono
# char_count=0
# with open(data_dir1, 'r', encoding='utf-8') as f:
#     while(True):
#         lines=f.readline().split('\n')
#         input_texts.append(lines)
#         for char in lines:
#             if char not in input_characters:
#                 input_characters.add(char)
#                 char_count=char_count+1
#                 if(char_count>100):
#                         break


# In[8]:
with open(data_path, 'r', encoding='utf-8',errors='ignore') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

#Data Preprocessing Bengali Mono
# with open(data_dir2, 'r', encoding='utf-8') as f2:
#     tlines = f2.read().split('\n')
# for tline in tlines[: min(num_samples, len(tlines) - 1)]:
#     target_texts.append(tline)
#     for char in tline:
#         if char not in target_characters:
#             target_characters.add(char)   


# In[9]:


# target_characters
# input_characters
#input_characters = dict(input_characters)


# In[ ]:


input_characters = sorted(list(input_characters))  
target_characters = sorted(list(target_characters))
num_encoder_tokens = 79#len(input_characters)
num_decoder_tokens = 143#len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


# In[ ]:





# In[ ]:


#Tokenize
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# In[ ]:


#initializing One HOT Encoding
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens),dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens),dtype='float32')


# In[ ]:





# In[ ]:


#One Hot Encoding
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


# In[ ]:


#Encoder # Define an input sequence and process it.

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]


# In[ ]:


decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


# In[ ]:


#model = Sequential()
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#model.add(GaussianNoise(0.05)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
          
# filepath="okam_olpo_eng_10k-{epoch:02d}-{loss:.4f}-{val_acc:.2f}.hdf5"
# checkpoint = ModelCheckpoint(filepath,monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

# #his = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2,callbacks=callbacks_list, verbose=1)

# # Save model
# model.save('10kEng_140Kben_.h5')

model.load_weights('10kEng_140Kben_.h5')
# In[ ]:


# print(his.history.keys())

# # summarize history for accuracy
# plt.plot(his.history['acc'])
# plt.plot(his.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# # In[ ]:


# # summarize history for loss
# plt.plot(his.history['loss'])
# plt.plot(his.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)


