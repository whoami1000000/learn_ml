import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras.datasets import imdb

NEGATIVE_REVIEW = '''
OK, I'll keep this brief, but I'll also nail my credentials to the mast first. I know science, cosmology, sci-fi and film-making pretty well. I have, in the past taught film studies at degree level, and worked on the production of more than one major Hollywood sci-fi film. I read Stephen Baxter and I like space stuff a lot. More than a lot. And why am I telling you this? Because I should have loved "Interstellar", but I didn't for one simple reason: It is a truly dreadful piece of film-making.

I won't even bother with dissecting the "science" in the film, because that isn't really the issue (it is wrong on pretty much every level). I'm not even going to bother complaining about the absurd audio balance in the sound mix. And I'll ignore the sub-standard special effects - because FX don't a great movie make.

The issues I have with Interstellar are essentially these: The film is an arse-numbing three hours long, and feels far longer. The pace is stunningly poorly judged - the first hour in particular could have been cut to ten minutes and far more would have been gained than lost.

The characters are paper-thin, and I didn't care about any of them. The plot is entirely derivative (mostly of the vastly superior "Contact"). The special effects aren't special at all and the editing (or lack of) is so self-indulgent it is a text book example of a director so enamoured with his project that he loses objectivity. The result is a plodding, flabby, and desperately dull movie devoid of any real excitement or emotional impact.

I won't go on, but special mention must be made of the planets - Paddling World and Coldworld. You see virtually nothing of either, and so utterly uninteresting are they that what should have been a moment of genuine cinematic wonderment was squandered with a bit of poor CGI painfully inferior to "A Perfect Storm" and a location less dramatic than your own back garden.

I've probably not been a brief as I intended, but as I write this I feel the disappointment and actual anger I felt on leaving the cinema bubbling to the surface again. It was a total let-down and a waste of more than three hours of my life. The gushing reviews on here are ridiculous and absurd, and I am forced to conclude that reviewers either watched a different film to me, or saw something so brilliant it completely passed me by. I am fairly confident I didn't doze off, although I desperately wanted to.

So in conclusion, "Contact" did all of this far, far better, fifteen years ago. In Contact the characters are human, believable, beautifully realised and you care what happens to them. The relationship between Father and Daughter is deeply moving and inspiring. But then the plot is far more sophisticated anyway, dealing with the social tensions and impact of the discovery of extra-terrestrial life, the science is accurate and entirely plausible (it was written by Carl Sagan after all), the movie is genuinely thrilling and full of spectacle, and it has something very profound to say.

Contact is everything Interstellar is not, and it has a considerably shorter running time. Contact brought tears to my eyes, Interstellar bored me to tears.
'''

POSITIVE_REVIEW = '''
Wowser! This Christopher Nolan film was presaged with such marketing hype that I went in with pretty low and cynical expectations. But I was frankly blown away with it.

Just about everyone raves about Christopher Nolan's work, and you look back at his Filmography and it makes for a pretty impressive resume: from Memento via the (rather over-hyped imho) Dark Knight Batman series-reboot through to Inception, one of my favourite films of all time. For me, Interstellar is right up there with Inception for thought-provoking, visually spectacular and truly epic cinema.

We start in familiar 'Day after Tomorrow" territory, with mankind having in some way – not entirely explained – messed up the planet. As I understood it (and the film probably does require multiple watches with – see comments below – subtitles=on) the rather clever premise is that the world's food supplies are being progressively destroyed by a vindictive 'blight'. This delivers the double whammy of destroying mankind's provisions but also, by massive reproduction of the organism, progressively depleting the Earth's oxygen. For some reason – again, which I didn't get on first viewing – this is accompanied by massive dust storms. It is a morbid bet as to what is going to get the mid-West population first: starvation, lung disease or suffocation. Matthew McConnaughey plays the widowed Cooper, an ex-NASA drop out turned farmer given the opportunity by mission-leader Professor Brand (an excellent Michael Caine) to pilot a NASA mission. The goal is to punch through a mysterious wormhole in space where they suspect, through previous work, that a new home for mankind could be found.

The first part of the film is set on and around Cooper's farm, setting in place one of the emotional wrenches at the heart of the film: that Cooper in volunteering for the mission and having to leave behind his elderly father (John Lithgow, again superb) and young children Murph (aged 10) and Tom (aged 15) whilst recognising that danger for him comes not just from the inherent risks involved but from the theory of relativity that could change everything, time-wise, for when he returns.

Cooper is supported on the mission by a team of scientists including Brand's daughter played by a love-struck Anne Hathaway, who again shows she can act.

To say any more would spoil what is a voyage of visual and mental discovery. (However, I would add that it is good to see that the character that plays my namesake Dr Mann (in a surprise cameo) is equally good looking! LOL).

In terms of plus points, where do I start? The visuals are utterly stunning. Whilst reminiscent in places of Kubrick's "stargate" from 2001, the similarity is only passing. The film adds a majesty and scale to space that surpasses wonder. Elsewhere there are some interesting visual effects: this might have just been me of course, but after the dramatic launch there was something about the camera moves during the first scenes of weightlessness that made me feel genuinely nauseous.

Equally stunning is Hans Zimmer's score which is epic and (in places) very VERY loud. The film certainly doesn't "go quietly into the night"! When matching the noise of the score/choir to the sound effects in the launch sequence the combination is ear-bleedingly effective. This must be a strong contender for the soundtrack Oscar for 2014. One quibble, again 2001 related, is that Zimmer uses the last chord of Also Sprach Zarathustra in the score sufficiently often that one hopes Richard Strauss's estate receives some royalties! The acting is top notch: I've already mentioned Caine and Lithgow, but McConnaughey, Hathaway and Jessica Chastain are all great. A particular shout-out should go to Mackenzie Foy as the young Murph, who is magnetically charismatic and just brilliant in the role.

Above all, Nolan's direction is exquisite. The film has a slow build on earth (which adds to the lengthy running time) but defines the characters and primes the plot perfectly. And some of the editing cuts – again, Cooper's farm departure/launch sequence overlay is a great example – are superb in building the mood and the tension.

I've decided that I am an extremely tough reviewer and for me a 10 star film is a rarity indeed. Where I could have knocked off a star was in some of the dialogue on the soundtrack, which was pretty inaudible in places: McConnaughey in particular with his general mumbling and strong southern accent is indecipherable in places. I look forward to the DVD subtitles. And one of the character's dying words – delivering a key plot point in the film – was completely lost to me (but thankfully later restated). Whilst the expansive plot is highly ambitious, the end of the film, playing fast and loose with physics I fear, requires a gravity-defying suspension of belief (although I guess the same could equally be said of 2001: A Space Odyssey).

However, the film has stayed so firmly lodged in my mind for 24 hours I will make a rare exception to my rating 'rule'. Overall, this is a top-notch Sci-Fi film. And a final word: PEOPLE THIS IS A MUST SEE ON THE BIG SCREEN! (If you enjoyed this review, please see my archive of previous reviews at bob-the-movie-man.com and sign up for future notifications. Thanks).
'''


def vectorize_sequences(sequences, dimensions=10000):
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
    data = list(filter(lambda i: i <= 10000, data))
    data = vectorize_sequences([data, ])
    p = model.predict(data)
    return p


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    word_index = imdb.get_word_index()

    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=4, batch_size=512, validation_data=(x_test, y_test))

    show_acc(history)

    print(predict(model, NEGATIVE_REVIEW, word_index))
    print(predict(model, POSITIVE_REVIEW, word_index))
