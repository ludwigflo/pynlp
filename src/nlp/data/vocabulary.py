from typing import Union


class Vocabulary:

    def __init__(self, name: str, pad_token=0, sos_token=1, eos_token=2) -> None:
        """
        Class for representing the vocabulary of a corpus.

        Parameters
        ----------
            name: name of the class.
        """

        # store the vocabulary name as well as the pad, sos and eos tokens
        self.name = name
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        # initialize the dictionaries containing the vocabularies and initialize the word count variable
        self.word2count = dict()
        self.word2index = {"<PAD>": pad_token, '<START>': sos_token, "<EOS>": eos_token}
        self.index2word = {pad_token: "<PAD>", sos_token: '<START>', eos_token: "<EOS>"}
        self.num_words = len(self.index2word)

    def __len__(self) -> int:
        """Returns the length of the vocabulary."""
        return self.num_words

    def add_word(self, word: str) -> None:
        """
        Adds a single word to the word dictionaries.

        Parameters
        ----------
            word: Word, which should included into the word dictionaries
        """

        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1
        if word not in self.word2count:
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence: list) -> None:
        """
        Adds all words of a sentence into the word dictionaries.

        Parameters
        ----------
            sentence: sentence, which should included into the word dictionaries.
        """

        for word in sentence:
            self.add_word(word)

    def add_document(self, document: list) -> None:
        """
        Adds all words in a document into a vocabulary dictionary.

        Parameters
        ----------
            document: Document, which words should included into the vocabulary.
        """

        for sentence in document:
            self.add_sentence(sentence)

    def add_corpus(self, corpus: list) -> None:
        """
        Adds all words in a corpus into the vocabulary dictionary.

        Parameters
        ----------
            corpus: Corpus, from which the words should be read into a dictionary.
        """

        for document in corpus:
            self.add_document(document)

    def trim(self, min_count: Union[int, None]=None, max_count: Union[int, None]=None) -> None:
        """
        Stores only words, which are above a special threshold in the word count.

        Parameters
        ----------
            min_count: threshold, above which the count of a word must be in order to be considered in the dictionary.
            max_count: threshold, under which the count of a word must be in order to be considered in the dictionary.
        """

        # store words, which have a count over a certain threshold
        keep_words = []
        count_list = []
        for word, word_count in self.word2count.items():
            consider = True
            if min_count is not None:
                if word_count < min_count:
                    consider = False
            if max_count is not None:
                if word_count > max_count:
                    consider = False
            if consider:
                keep_words.append(word)
                count_list.append(word_count)
        print("Keep {} / {} words ({:.4f}%)".format(len(keep_words),
                                                    len(self.word2index),
                                                    float(len(keep_words))/len(self.word2index)))

        # reinitialize dictionaries
        self.word2index = {"<PAD>": self.pad_token, '<START>': self.sos_token, "<EOS>": self.eos_token}
        self.word2count, self.num_words = {}, 3
        self.index2word = {self.pad_token: "<PAD>", self.sos_token: '<START>', self.eos_token: "<EOS>"}

        self.num_words = 3
        # crate new dictionaries with only words above the specified threshold
        for i, word in enumerate(keep_words):
            self.add_word(word)

        # recompute the word counts
        self.word2count = {}
        for i, word in enumerate(keep_words):
            self.word2count[word] = count_list[i]
