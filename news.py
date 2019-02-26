import numpy as np
from keras import models
from keras.datasets import reuters
from keras import layers
from keras.utils.np_utils import to_categorical

NUM_WORDS = 10000

SPORT = '''
Brendan Rodgers can do better than Leicester and may only be treating the job as a stepping stone on the way to a top-six Premier League side, believes Kris Commons.

Celtic manager Rodgers has verbally agreed to become the new Leicester manager, according to Sky sources, following the sacking of Claude Puel on Sunday.

However, Commons believes the former Liverpool boss is capable of managing better sides than 12th-placed Leicester, and when asked if it was a bigger job than Celtic, replied: "Absolutely not.

"Leicester have got a top coach - a proven manager - but I think Brendan's better than Leicester. I think he proved that at Liverpool and he proved that at Celtic.

"It wasn't so long ago he was linked with the Arsenal job, and I think that's the sort of thing that Brendan wants. He wants to be winning titles.

"Whether this is a stepping stone, a couple-year job, [before] he looks at the likes of Arsenal, or maybe a Chelsea - I think that's more in Rodgers' mind right now."

Commons, who played for Celtic between 2011 and 2017, admitted he understood why Rodgers wanted to return to the Premier League, but questioned the timing of his expected departure.

The 46-year-old is capable of leading Celtic to an unprecedented 'treble treble', and Commons said: "It's a strange one. Obviously the Premier League is a real big pull for him, but it's the timing. Leicester have not got a great deal to play for - they're lingering in mid-table.

"I understand why he's gone there, but being on the cusp of winning eight Premiership titles [in a row], and with the opportunity to win the treble, the timing is a little bit off.

"I would have thought he would have waited until the end of the season."

Rodgers is expected to be replaced by former Celtic boss Neil Lennon on an interim basis, Sky Sports News understands, and when asked whether returning to Parkhead was a risk for him, Commons said: "No, I think Celtic runs through his blood.

"I don't think Lennon will be in a position to turn this job down. He's a great manager who's done wonderfully well in Scotland, and he's the sort of manager who thrives on the atmosphere of Celtic.

"He thrived as a player, captain and manager. But he's got big boots to fill - seven trophies on the spin for Brendan Rodgers."
'''


def vectorize_sequences(sequences, dimensions=NUM_WORDS):
    shape = (len(sequences), dimensions)
    results = np.zeros(shape)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def predict(model, news, index):
    unknown = 2
    words = news.split(' ')
    data = list([index.get(word, unknown) for word in words])
    data = list(filter(lambda i: i <= NUM_WORDS, data))
    data = vectorize_sequences([data, ])
    p = model.predict(data)
    return np.argmax(p[0])


def main():
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=NUM_WORDS)

    word_index = reuters.get_word_index()

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = to_categorical(train_labels)
    y_test = to_categorical(test_labels)

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=20, batch_size=512, validation_data=(x_test, y_test))

    print(predict(model, SPORT, word_index))


if __name__ == '__main__':
    main()
