import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras.datasets import imdb

NEGATIVE_REVIEW = '''
You have to worry about how some producers market a movie . ACE VENTURA PET DETECTIVE is a case in point . It's about a detective who loves animals , he keeps all sorts of furry and feathered friends in his house and there's nothing more family friendly than a movie featuring lovely animals , you can just see the little ones begging mom and dad to take them to the cinema to see this movie . Just to hammer the point home the main plot centres around a dolphin mascot being kidnapped and there's nothing more heartbreaking to a family audience than a dolphin in peril 

So with got a set up and plot that Walt Disney would have been proud of and what do the producers do ? They decide to inject lots of crude humour into it . I was going to write " Crude adult humour " but it's not even sophisticated enough to reach the standards of adult . Are there any adults who think references to veneral disease are funny ? Will children get the in jokes to THE CRYING GAME ? I should hope not . Bizarrely the BBC decided to show this earlier today at a traditional family slot in the early evening which proves the BBC know as little about movies as the average Hollywood producer 

That's bad enough but what really disgusted me is that ACE VENTURA PET DETECTIVE turned Jim Carrey into a film star . Now that is unforgivable
'''

MIDDLE_REVIEW = '''
That is the problem with people watching movies like this. They like it, and call Jim Carrey the best. Any one can make a fool out of themselves. It isn't really that hard at all. Really, a movie that an actor makes a fool out of himself to make people laugh. Most any comedy now is just this. I really got out of this in my 20's. I do not see it really funny any more or Jim Carrey at all. Its just dumb, like the title Dumb and Dumber.

Being dumb is actually not funny. You will never really grow up by being dumb, and people learn to be dumb, by watching these kind of movies, thinking it is the coolest thing to try what they do to make people laugh. OK, that really makes no sense at all, but people will do it. Might as well, go get plastered from drinking, and make people laugh too. That is exactly the same thing people do as Jim Carrey does. He is probably drunk when he does his movies.

I am hesitant on seeing Kick Ass 2, because he is in it.
'''

POSITIVE_REVIEW = '''
In 1994, a year that gave us Forrest Gump, Pulp Fiction and The Shawshank Redemption, Morgan Creek Productions and Warner Bros. Pictures came out with the funniest film of the decade, err, 1994, with the Jim Carrey super-vehicle, ACE VENTURA: PET DETECTIVE.

Right off the bat, it is a Jim Carrey showcase: chasing missing albino pigeons, rescuing a pampered shiatsu, and making high speed getaways, this all works because it is Ace Ventura, a hilarious, live-action cartoon character who spouts one-liners like an M-16.

The premise of the film is that the Miami Dolphin's team mascot, Snowflake the dolphin, has been captured and needs to be found in time for the Super Bowl. Ace is hired and the sexy Courtney Cox is made his partner until the dolphin is recovered. With an assortment of funny situations like Ace falling into a great white shark tank (!), a montage of Ace searching for a missing jewel, Ace head banging in a CANNIBAL CORPSE concert, it's all gold, and as you have read, Ace is what drives this film into comedy genius! 

It was a busy and successful year for Jim Carrey, as he came out with another pair of comedy blockbusters, the LOOSELY translated Dark Horse comic THE MASK, and his hilarious buddy comedy with Jeff Daniels, DUMB and DUMBER.

Thanks to Tom Shadyac for letting Carrey out of the bag and letting this damn funny Canadian strut his lanky stuff.

It has some offensive, adult-oriented material, but if caught on television, this is nearly perfect family entertainment: as long as you don't mind your kids talking from their a$$es for the next couple months.
'''


def vectorize_sequences(sequences, dimensions=1000):
    shape = (len(sequences), dimensions)
    results = np.zeros(shape)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def show_acc(history):
    history_dict = history.history

    acc_values = history_dict['acc']
    val_acc_values = history_dict['val_acc']
    epochs = range(1, len(acc_values) + 1)

    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def predict(model, review, index):
    unknown = 2
    words = review.split(' ')
    data = list([index.get(word, unknown) for word in words])
    data = list(filter(lambda i: i <= 1000, data))
    data = vectorize_sequences([data, ])
    p = model.predict(data)
    return p


def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
    word_index = imdb.get_word_index()
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(1000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=4, batch_size=512, validation_data=(x_test, y_test))
    show_acc(history)
    print(predict(model, NEGATIVE_REVIEW, word_index))
    print(predict(model, MIDDLE_REVIEW, word_index))
    print(predict(model, POSITIVE_REVIEW, word_index))


if __name__ == '__main__':
    main()
