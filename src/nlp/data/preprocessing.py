from nltk.tokenize import  sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nlp.utils import read_parameter_file
from nltk.corpus import stopwords
import string
import sys
import ast
import re


def list_to_list_string(list_var: list) -> str:
    """
    Converts a list representation into the string representation of a list.

    Parameters
    ----------
        list_var:

    Returns
    -------
        string_var:
    """

    string_var = str(list_var)
    return string_var


def list_string_to_list(string_var: str) -> list:
    """
    Converts a string representation of a list into the list itself.

    Parameters
    ----------
        string_var:

    Returns
    -------
        list_var:
    """

    string_var = ast.literal_eval(string_var)
    list_var = [x.strip() for x in string_var]
    return list_var


class Preprocessing:
    def __init__(self, parameters_file: str) -> None:
        """
        Parameters
        ----------
            parameters_file: yaml file, in which the preprocessing parameters are specified. It needs to contain the key
                             preprocessing, under which the preprocessing steps are specified.
        """

        self.preprocessing_params = read_parameter_file(parameters_file)['preprocessing']

    @staticmethod
    def sentence_word_split(sentence: str) -> list:
        """
        Splits a sentence, provided as a single string, into a list of its words.

        Parameters
        ----------
            sentence:

        Returns
        -------
            token_list:
        """

        token_list = word_tokenize(sentence)
        return token_list

    @staticmethod
    def document_sentence_split(document: str) -> list:
        """
        Splits a document, provided as a single string, into a list of its sentences.

        Parameters
        ----------
            document: String representation of a document.

        Returns
        -------
            document: List representation of a document.
        """

        document = sent_tokenize(document)
        return document

    @staticmethod
    def add_start_and_end_tokens(sentence: list, sos_token: str='<START>', eos_token: str= "<EOS>") -> list:
        """
        Adds a start token and an end token to the provided sentence (which is represented as a list of words).

        Parameters
        ----------
            sentence: Sentence, to which start and end tokens should be added.
            sos_token:
            eos_token:

        Returns
        -------
            sentence_out: Sentence, to which start and end tokens are added.
        """

        sentence_out = [sos_token]
        sentence_out.extend(sentence)
        sentence_out.append(eos_token)
        return sentence_out

    def compute_tokenized_corpus(self, corpus: list) -> list:
        """
        Computes the standard form of a corpus, which is provided as a list of documents. Each of these documents can
        either be a single string containing all sentences, or a list of sentences. In the latter case, each sentence
        can either be a single string or a list of its words. The tokenized corpus is a list of documents, which are
        represented as a list of sentences. Each of these sentences is represented as a list of tokens (words).

        Parameters
        ----------
            corpus: A list of documents.

        Returns
        -------
            corpus: Tokenized corpus.
        """

        for i, document in enumerate(corpus):
            i = float(i)
            out_string = "\rDocuments: {0}%".format(int((i+1) / len(corpus) * 100))
            sys.stdout.write(out_string)
            sys.stdout.flush()
            i = int(i)

            # convert documents into lists of sentences
            if type(document) is str:
                document = self.document_sentence_split(document)

            # convert sentences into lists of words
            for j, sentence in enumerate(document):
                if type(sentence) is str:
                    sentence = self.sentence_word_split(sentence)

                # store the processed sentence in the document
                document[j] = sentence

            # store the processed document in the corpus
            corpus[i] = document

        out_string = "\rDocuments: {0}%".format(100)
        sys.stdout.write(out_string)
        sys.stdout.flush()
        print('\n')
        return corpus

    @staticmethod
    def stem_sentence(sentence: list) -> list:
        """
        Stemming of all words in a provided sentence.

        Parameters
        ----------
            sentence: List representation of a sentence.

        Returns
        -------
            sentence: List representation of a sentence, in which the words are stemmed.
        """

        stemmer = PorterStemmer()
        sentence = [stemmer.stem(word) for word in sentence]
        return sentence

    @staticmethod
    def lemmatize_sentence(sentence: list) -> list:
        """
        Lemmatizes a sentence, which is provided as a list of words

        Parameters
        ----------
            sentence: List representation of a sentence.

        Returns
        -------
            sentence: List representation of a sentence, in which the words are lemmatized.
        """

        lemmatizer = WordNetLemmatizer()
        sentence = [lemmatizer.lemmatize(word) for word in sentence]
        return sentence

    @staticmethod
    def lower_case_sentence(sentence: list) -> list:
        """
        Converts all words in a sentence into their lower cased version.

        Parameters
        ----------
            sentence: List representation of a sentence.

        Returns
        -------
            sentence: List representation of a sentence, in which all words are lower cased.
        """

        sentence = [x.lower() for x in sentence]
        return sentence

    @staticmethod
    def remove_punctuation(sentence: list) -> list:
        """
        Removes all of the common punctuations in a sentence, which is provided as a list of words.

        Parameters
        ----------
            sentence: Representation of a sentence as a list of words.

        Returns
        -------
            sentence: Representation of a sentence as a list of words, in which punctuations are removed.
        """

        punctuation = list(string.punctuation)
        sentence = [x for x in sentence if x not in punctuation]
        return sentence

    @staticmethod
    def remove_non_alpha_numeric_characters(sentence: list) -> list:
        """
        Removes all non alpha-numeric characters from an input sentence, which is represented as a list of words.

        Parameters
        ----------
            sentence: List representation of a sentence.

        Returns
        -------
            sentence: List representation of a sentence, in which all non-alpha numeric characters are removed.
        """

        sentence = [re.sub(r'\W+', '', x) for x in sentence]
        sentence = [x for x in sentence if len(x) > 0]
        return sentence

    @staticmethod
    def remove_stop_words(sentence: list, language: str='english') -> list:
        """
        Removes all stopwords from a sentence, which is provided as a list of words. This function uses stopwords, which
        are defined in the nltk library. A language can be chosen by providing an appropriate language string.

        Parameters
        ----------
            sentence: List representation of a sentence.
            language: Language, from which the stopwords, defined by the nltk library, are imported.

        Returns
        -------
            sentence: List representation of a sentence, in which the stopwords have been removed.
        """

        stop_words = set(stopwords.words(language))
        sentence = [x for x in sentence if x not in stop_words]
        return sentence

    def preprocess_corpus(self, corpus: list) -> list:
        """
        Complete preprocessing routing for the whole provided corpus. The preprocessing steps are specified in the
        provided parameter file at the time of instance creation.

        Parameters
        ----------
            corpus: List of documents, which should be preprocessed.

        Returns
        -------
            corpus: Preprocessed corpus, represented as a list of documents.
        """

        # currently available preprocessing methods
        excluded_methods = ['compute_tokenized_corpus', 'document_sentence_split',
                            'preprocess_corpus', 'sentence_word_split', 'add_start_and_end_tokens']

        method_list = [method_name for method_name in dir(self)
                       if callable(getattr(self, method_name))
                       and method_name not in excluded_methods
                       and method_name[0:2]!='__']

        if self.preprocessing_params['add_start_and_end_tokens']:
            method_list.append('add_start_and_end_tokens')

        # convert corpus into its list representation (tokenized form)
        print('Converting the provided corpus to its list representation...')
        corpus = self.compute_tokenized_corpus(corpus)

        # get the current language for removing the stop words, if necessary
        language = 'english'
        if 'language' in self.preprocessing_params:
            if self.preprocessing_params['language'] is not None:
                language = self.preprocessing_params['language']

        # store all method names from the preprocessing file, which are available in this module, in a list
        preprocessing_list = []
        for key in method_list:
            if key in self.preprocessing_params:
                if self.preprocessing_params[key]:
                    preprocessing_list.append(key)

        print('Preprocess corpus...')
        method_length = len(method_list)
        num_documents = len(corpus)
        # iterate through the method names
        for j, preprocessing_method_name in enumerate(preprocessing_list):

            # load the corresponding function from its name string
            preprocessing_method = getattr(self, preprocessing_method_name)

            # iterate through the corpus and preprocess the documents
            for d, document in enumerate(corpus):

                # print the current preprocessing status
                j, d = float(j), float(d)
                out_string = "\rPreprocessing methods: {0}%  Documents: {1}%".format(int((j+1)/method_length*100),
                                                                                     int((d+1)/num_documents*100))
                sys.stdout.write(out_string)
                sys.stdout.flush()
                j, d = int(j), int(d)

                # preprocess each sentence with the currently loaded method
                for i, sentence in enumerate(document):
                    if preprocessing_method_name == 'remove_stop_words':
                        sentence = preprocessing_method(sentence, language=language)
                    else:
                        sentence = preprocessing_method(sentence)
                    sentence = [x for x in sentence if len(x)>0]
                    document[i] = sentence
                corpus[d] = document
        out_string = "\rPreprocessing methods: {0}%  Documents: {1}%".format(100, 100)
        sys.stdout.write(out_string)
        sys.stdout.flush()
        print('\n')
        return corpus
