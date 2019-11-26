import tensorflow as tf
import tensorflow_hub as hub

tokens_input = [["the", "cat", "is", "on", "the", "mat"],
                ["dogs", "are", "in", "the", "fog"],
                ["the", "cat", "is", "on", "the", "mat", "cat"]
            ]

sequence_len= 6

for i, _tokens in enumerate(tokens_input):
    _tokens = _tokens[:sequence_len]
    _tokens.extend( [""]*( sequence_len- len(_tokens) ) )
    tokens_input[i] = _tokens


print('loading model')
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
print('model loaded')

# embeddings = elmo(
#     ["the cat is on the mat", "dogs are in the fog and are playing"],
#     signature="default",
#     as_dict=True)["elmo"]

# tokens_length = [6, 5]
x_tensor = tf.placeholder(dtype = tf.string, shape=(None,None))
sequence_lengths = tf.cast(tf.count_nonzero(x_tensor, axis=1), dtype=tf.int32)

embeddings = elmo(
    inputs={
        "tokens": x_tensor,
        "sequence_len": sequence_lengths
    },
    signature="tokens",
    as_dict=True)["default"]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(tokens_length))
    # a = sess.run(embeddings)
    l, a = sess.run(fetches = [sequence_lengths, embeddings], feed_dict= { x_tensor: tokens_input })

print(l)
print(a.shape)
print(a[:5])

