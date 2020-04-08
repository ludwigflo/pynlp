from nlp.utils import read_parameter_file
from .preprocessing import Preprocessing
from .vocabulary import Vocabulary
from copy import deepcopy
from typing import Union
import os

# import the best available version of the pickle module
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle


class Corpus:
    def __init__(self, corpus_name: str='', param_file: str='', corpus_list: Union[list, None]= None) -> None:
        """
        Parameters
        ----------
            corpus_name: Name of the corpus.
            param_file: File, in which all parameters of the current corpus are stored.
        """

        # initialize preprocessing routine
        self.corpus = []
        self.corpus_name = corpus_name
        self.voc = Vocabulary(name=corpus_name)
        self.params = read_parameter_file(param_file)
        self.preprocessor = Preprocessing(param_file)

        # load the corpus, as specified in the parameter file
        self.load_corpus(corpus_list)

    def __len__(self) -> int:
        """
        Computes the length of the corpus (number of documents).

        Returns
        -------
            corpus_length, which is basically the number of documents, which the corpus contains.
        """
        return len(self.corpus)

    def _load_from_corpus_list(self, corpus_list_raw: list) -> list:
        """
        Loads a raw corpus from a list representation of a corpus. After that, it preprocesses the corpus and extracts
        its vocabulary. Finally, it stores the preprocessed corpus as well as the vocabulary as class attributes.

        Parameters
        ----------
            corpus_list_raw: Raw corpus in its list representation.

        Returns
        -------
            corpus: Preprocessed corpus in list representation.
        """

        corpus = self.preprocessor.preprocess_corpus(corpus_list_raw)
        return corpus

    def _load_from_document_folder(self, corpus_path: str) -> list:
        """
        Loads a corpus based on a path, in which its documents in form of text files are stored.

        Parameters
        ----------
            corpus_path: Path, in which documents in form of text files are stored.

        Returns
        -------
            corpus: Preprocessed corpus in list representation.
        """

        document_names = [x for x in os.listdir(corpus_path) if '.txt' in x]
        raw_corpus = []
        for document_name in document_names:
            with open(corpus_path + document_name, 'r') as f:
                lines = f.readlines()
                document = ''
                for line in lines:
                    document += line
                raw_corpus.append(document)
        corpus = self.preprocessor.preprocess_corpus(raw_corpus)
        return corpus

    def _load_corpus_object(self, corpus_name) -> None:
        """
        Loads a complete corpus object.

        Parameters
        ----------
            corpus_name: Name, under which the corpus is saved (file_name of the corpus).
        """

        with open(corpus_name, 'rb') as f:
            # load the provided pickle file
            c_temp = pickle.load(f)

            # copy the class attributes of the loaded file to the class attributes of the current file
            self.preprocessor = deepcopy(c_temp.preprocessor)
            self.corpus_name = deepcopy(c_temp.corpus_name)
            self.params = deepcopy(c_temp.params)
            self.corpus = deepcopy(c_temp.corpus)
            self.voc = deepcopy(c_temp.voc)

            # delete the loaded class instance
            del c_temp

    def load_corpus(self, corpus_list: Union[None, list]) -> None:
        """
        Parameters
        ----------
            corpus_list: Corpus in its list presentation (Optional, only if loading type == 'list')
        """

        # if the corpus key exists in the parameter file
        if 'corpus' in self.params:
            # if the loading_type key exists in the parameter file
            if 'loading_type' in self.params['corpus']:

                # extract the required parameters
                loading_type = self.params['corpus']['loading_type']

                # check, if the provided parameters are correct
                types = ['folder', 'object', 'list', 'None']
                assert loading_type in types, "Loading type needs to be 'folder', 'pickel_file', list or None"

                # if no loading at initialization time is required
                if loading_type == 'None':
                    print('No corpus should be loaded at initializing time - loading type is None')

                # if we should load the corpus from a list
                elif (loading_type=='list') and (corpus_list is not None):
                    self.corpus = self._load_from_corpus_list(corpus_list)

                # if we load the single documents from a root folder (documents provided in text format)
                elif loading_type=='folder':
                    # check if a path has been provided
                    if 'path' in self.params['corpus']:
                        path = self.params['corpus']['path']
                        self.corpus = self._load_from_document_folder(path)
                    else:
                        print('You need to provide a directory containing the documents in text file format')

                # if we load a complete corpus object
                elif loading_type=='object':
                    object_dir = self.params['corpus']['path']
                    self._load_corpus_object(object_dir)

                # add the corpus to the vocabulary
                if loading_type != 'None' and loading_type != 'object':
                    self.voc.add_corpus(self.corpus)
            else:
                print('No loading_type key specified in the parameter file')
        else:
            print('No corpus key specified in the parameter file')

    def save_corpus_object(self, output_path: str) -> None:
        """
        Saves the current corpus object to a pickle file.

        Parameters
        ----------
            output_path: Directory, in which the corpus should be saved.
        """

        with open(output_path + self.corpus_name, 'wb') as f:
            pickle.dump(self, f)
