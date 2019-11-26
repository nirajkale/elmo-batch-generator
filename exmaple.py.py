import pandas as pd
from elmo_generator import ElmoBatchGenerator
from keras import layers, models
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from tqdm import tqdm

if __name__ == "__main__":
    
    # data, labels = load_dataset()
    df= pd.read_excel('sentiments.xlsx')
    df.dropna()

    data= list(df['data'])
    labels = list(df['labels'])

    tokens = []
    targets = []
    for i, text in tqdm(enumerate(data), desc='tokenizing tests', total= len(data)):
        try:
            if not (text is np.nan):
                tokens.append(text_to_word_sequence(text))
                if labels[i]=='pos':
                    targets.append(1)
                else:
                    targets.append(0)
        except Exception as e:
            print(e)
    
    targets = np.array(targets)
    assert(len(data)==len(labels))

    train_gen = ElmoBatchGenerator(
        data = tokens[:20000],
        labels = targets[:20000],
        sequence_length= 100,
        output_mode= 'elmo',
        signature= 'tokens',
        batch_size= 32,
        return_incomplete_batch= True,
        use_embedding_caching= True
    )

    val_gen = ElmoBatchGenerator(
        data = tokens[20000:],
        labels = targets[20000:],
        sequence_length= 100,
        output_mode= 'elmo',
        signature= 'tokens',
        batch_size= 32,
        return_incomplete_batch= True,
        use_embedding_caching= True
    )

    input_layer = layers.Input(shape=(100,train_gen.emb_dim,), name= 'input_layer')
    a = layers.LSTM(128 )(input_layer)
    a = layers.Dense(128, activation= 'relu')(a)
    output_layer = layers.Dense(1, activation= 'sigmoid')(a)

    model = models.Model( input_layer, output_layer)
    model.compile( optimizer='adam', metrics= ['acc'], loss= 'binary_crossentropy')
    print(model.summary())

    train_gen.warmup(3)

    model.fit_generator(
        generator = train_gen,
        epochs = 8,
        verbose = 1,
        validation_data: val_gen
    )

    tokens = [text_to_word_sequence(text) for text in texts]

    x = train_gen.get_embeddings_for_tokens(tokens)

    print(model.predict(x))

    