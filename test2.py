import tensorflow as tf
import tensorflow_hub as hub

print('loading model')
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
print('model loaded')

x = ["the cat is on the mat", "dogs are in the fog and are playing"]

x_tensor = tf.placeholder(dtype = tf.string, shape=(None,))

embeddings = elmo(
    x_tensor,
    signature="default",
    as_dict=True)["elmo"]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(tokens_length))
    # a = sess.run(embeddings)
    a = sess.run(fetches = [embeddings], feed_dict= { x_tensor: x })

print(a.shape)