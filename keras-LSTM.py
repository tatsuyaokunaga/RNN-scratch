from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Input, Dense, Embedding
from keras.models import Model
from keras.layers import SimpleRNN,LSTM
from keras.datasets import imdb

max_features = 10000
maxlen = 40
batch_size = 32

print('load dataset')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print('padding')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

print('build model')
inp = Input(shape=(maxlen,), dtype='int32', name='main_input')
x = Embedding(max_features, 128)(inp) # max_featuresを128次元に成形
simple_rnn_out = LSTM(32)(x)
predictions = Dense(1, activation='sigmoid')(simple_rnn_out)
model = Model(inputs=inp, outputs=predictions)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('train')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=3,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
