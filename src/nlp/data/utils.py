import numpy as np
import h5py
import os


def save_corpus(data_path: str, data_file_name: str, corpus: list) -> None:
    """
    Parameters
    ----------
        data_path: path, in which the output file is located.
        data_file_name: name of the hdf5 file, in which the corpus is stored.
        corpus: Corpus, which should be stored.
    """

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # create the hdf5 file and initialize the data type, which is used to store the corpus elements
    h5_file = h5py.File(data_path + data_file_name,'w')
    h5_dtype = h5py.special_dtype(vlen=str)

    for i, document in enumerate(corpus):
        data = np.array(document, dtype=object)
        group = h5_file.create_group(name='document_'+str(i))
        group.create_dataset(name='sentences', data=data, dtype=h5_dtype)


def load_corpus(data_file_name: str) -> list:
    """
    Parameters
    ----------
        data_file_name: name of the hdf5 file, in which the corpus is stored

    Returns
    -------
        corpus: list representation of a corpus (list of documents, which are lists of sentences, which are either lists
                strings or strings)
    """

    # load a hdf5 file
    h5_file = h5py.File(data_file_name, 'r')

    # load each document in the file, convert it to the correct data type and store it in the corpus list
    corpus = []
    for document in h5_file:
        doc = h5_file[document]['sentences'][:].tolist()
        corpus.append(doc)
    return corpus


if __name__ == '__main__':
    file_path = '/media/data1/flo/NLP/src/nlp/data/Ted/contents/ted.hdf5'
    corpus = load_corpus(file_path)
    print(type(corpus), len(corpus))
    print(corpus)
