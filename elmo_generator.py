from keras.utils import Sequence
import re
import numpy as np
from keras import backend as K
import tensorflow as tf
import tensorflow_hub as hub
import math

class ElmoBatchGenerator(Sequence):

    def __init__(self, 
            data:[],
            labels:[],
            sequence_length:int = -1, #-1 if mode is default
            output_mode = "default",
            signature = 'tokens',
            batch_size:int = 32,
            return_incomplete_batch = True,
            use_embedding_caching = True
            # caching_id:str='',
        ):
        self.data = data
        self.labels = labels
        self.m = len(data)
        assert(self.m >1)
        # self.caching_id = re.sub('[^a-zA-Z0-9]','_', caching_id)
        # assert(len(caching_id)>0)
        self.batch_size = batch_size
        assert(self.batch_size>1)
        if output_mode !='default':
            assert(sequence_length >1)
        #define cache
        self.emb_dim = [512 if output_mode == 'word_emb' else 1024][0]
        self.output_mode = output_mode
        self.signature = signature
        self.sequence_length = sequence_length
        self.use_embedding_caching = use_embedding_caching
        self.sess = K.get_session()
        assert(self.signature in ['default', 'tokens'])
        self.return_incomplete_batch = return_incomplete_batch
        self.define_model()
        if self.output_mode=='default' and self.signature =='default':
            print('Caching is not possible with current config')
            self.use_embedding_caching = False
        if self.use_embedding_caching:
            if output_mode =='default':
                self.cache = np.zeros((self.m, self.emb_dim), dtype='float32')
            else:
                self.cache = np.zeros((self.m, self.sequence_length ,self.emb_dim), dtype='float32')

    def define_model(self):
        print('loading/ download elmo model..')
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
        print('model ready!')
        if self.signature =='tokens':
            self.x_tensor = tf.placeholder(dtype = tf.string, shape=(None,None))
            self.token_lengths = tf.cast(tf.count_nonzero( self.x_tensor, axis=1), dtype=tf.int32)
            self.embeddings = self.elmo(
                inputs={
                    "tokens": self.x_tensor,
                    "sequence_len": self.token_lengths
                },
                signature="tokens",
                as_dict=True)[ self.output_mode]
        else:
            self.x_tensor = tf.placeholder(dtype = tf.string, shape=(None,))
            self.embeddings = self.elmo(
                self.x_tensor,
                signature="default",
                as_dict=True)[ self.output_mode]
        self.sess.run(tf.global_variables_initializer())

    def get_embeddings_for_texts(self, texts):
        emb = self.sess.run(fetches = self.embeddings, feed_dict= { self.x_tensor: texts })
        if self.output_mode == 'deafult':
            return emb
        return emb[:, :self.sequence_length]

    def get_embeddings_for_tokens(self, tokens):
        for i, _tokens in enumerate(tokens):
            try:
                _tokens = _tokens[: self.sequence_length]
                _tokens.extend( [""]*( self.sequence_length - len(_tokens) ) )
                tokens[i] = _tokens
            except Exception as e:
                print(e)
                print('TOkens:', _tokens)
                print('Original:', tokens[i])
        emb = self.sess.run(fetches = self.embeddings, feed_dict= { self.x_tensor: tokens })
        return emb
        
    def get_batch(self, index):
        batch_start = (index * self.batch_size)% self.m
        batch_end = ((index + 1) * self.batch_size) % self.m
        if batch_end <= batch_start:
            batch_end = self.m
        batch_data = self.data[ batch_start: batch_end]
        labels_batch = self.labels[ batch_start: batch_end]
        if self.use_embedding_caching:
            cached_emb = self.cache[ batch_start:batch_end]
            counts = np.count_nonzero(cached_emb, axis= -1)
            if self.output_mode!='default':
                counts = np.count_nonzero(counts, axis= -1)
            counts = np.where(counts>1, 1, 0).sum()
            if counts == cached_emb.shape[0]:
                print('using cache')
                return ( cached_emb, labels_batch)
        if self.signature =='default':
            embeddings = self.get_embeddings_for_texts( batch_data)
        else:
            embeddings = self.get_embeddings_for_tokens( batch_data)
        if self.use_embedding_caching:
            self.cache[batch_start: batch_end] = embeddings
        return (embeddings, labels_batch)

    def __len__(self):
        if self.return_incomplete_batch:
            return int( math.ceil(self.m/ self.batch_size))
        return int( self.m/ self.batch_size)

    #for keras compatibility
    def __getitem__(self, idx):
        return self.get_batch(idx)

    


if __name__ == "__main__":
    
    
    tokens_input = [
        ["the", "cat", "is", "on", "the", "mat"],
        ["dogs", "are", "in", "the", "fog"],
        ["the", "cat", "is", "on", "the", "mat", "cat"],
        ["the", "cat", "is", "on", "the", "mat"],
        ["dogs", "are", "in", "the", "fog"],
        ["the", "cat", "is", "on", "the", "mat", "cat"],
        ["the", "cat", "is", "on", "the", "mat"],
        ["dogs", "are", "in", "the", "fog"],
        ["the", "cat", "is", "on", "the", "mat", "cat"],
        ["the", "cat", "is", "on", "the", "mat"],
        ["dogs", "are", "in", "the", "fog"],
        ["the", "cat", "is", "on", "the", "mat", "cat"]
    ]
    
    labels = [1,]* len(tokens_input)

    gen = ElmoBatchGenerator(
            tokens_input, \
            labels,\
            sequence_length= 5,\
            output_mode= 'elmo',\
            signature= 'tokens',\
            batch_size= 5,\
            return_incomplete_batch= True,\
            use_embedding_caching= True
            # caching_id='niraj'
        )
    # a = gen.get_embeddings_for_tokens(tokens_input)
    # print(a.shape)
    # print(a[:2])

    # print( len(gen))
    for i in range(10):
        x, y_true = gen[i]
        print(x.shape, ' | ', len(y_true))


    
