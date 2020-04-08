from nlp.text_embeddings.document_embeddings.bag_of_words import BagOfWords
from nlp.data.corpus import Corpus
import csv
import os


def load_tsv_file(file_path: str) -> list:
    """
    Loads a tsv file, which is specified by its path.

    Parameters
    ----------
    file_path: Path to the tsv file.

    Returns
    -------
    lines: A list containing the lines of the tsv file.
    """

    with open(file_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter='\t')
        lines = []
        for line in csv_reader:
            lines.append(line)
    return lines


def compute_corpus(tsv_file_path: str, yaml_file_path: str, output_file_name: str) -> Corpus:
    """
    Reads a tsv file of the lmdb dataset, extracts its corresponding text, prprocesses the text accoring to some
    parameters, which are specified in a yaml file. after that, it converts the text and stores the results as a Corpus
    object.

    Parameters
    ----------
    tsv_file_path: file to the csv file which should be read.
    yaml_file_path: File to the yaml file, which  defines the preprocessing parameters.
    output_file_name: Name of the corpus, which is used to store the Corpus object.

    Returns
    -------
    corpus: Corpus object, created from the review texts in the tsv file.

    """

    # read the tsv file and return the lines, in which the reviews are stored.
    lines = load_tsv_file(tsv_file_path)
    index = lines[0].index('review')
    corpus_list = [x[index] for x in lines[1:]]

    # compute and return the corpus
    corpus = Corpus(output_file_name, yaml_file_path, corpus_list)
    return corpus


def create_dataset(tsv_file_path: str, param_file: str, output_path: str):
    """
    Creates a dataset by loading the data of the lmdb dataset, preprocessing it and storing it in the provided
    directory. The corpora are split into labeled train, unlabeled train and test corpora. For the labeled train
    corpora, a text file containing the labels is created.

    Parameters
    ----------
    tsv_file_path: base path, in which the tsv files, which contain the raw data, are stored.
    param_file: Parameter file, which specifies the preprocessing routines.
    output_path: path, in which the dataset should be stored.
    """

    # create the output directory, if it doesn't already exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # get the names of the tsv files
    tsv_names = [x for x in os.listdir(tsv_file_path) if '.tsv' in x]

    # for each tsv name, create a corpus object and save it into the output path
    for tsv_name in tsv_names:
        # store the corpus
        output_name = tsv_name.replace('.tsv', '.pickl')
        corpus = compute_corpus(tsv_file_path + tsv_name, param_file, output_name)
        corpus.save_corpus_object(output_path)

        # load the tsv file
        lines = load_tsv_file(tsv_file_path + tsv_name)
        id_index = lines[0].index('id')
        sent_index = None
        if 'sentiment' in lines[0]:
            sent_index = lines[0].index('sentiment')

        # store the entries in a csv file
        with open(output_path + output_name.replace('.pickl', '.csv'), 'w') as f:
            csv_writer = csv.writer(f, delimiter=',')
            for line in lines[1:]:
                if sent_index is not None:
                    out_line =  [line[id_index], line[sent_index]]
                else:
                    out_line = [line[id_index]]
                csv_writer.writerow(out_line)


def compute_bag_of_words_embeddings(output_path: str, param_file: str) -> None:
    """

    Parameters
    ----------
    output_path: Path, in which the bag of words embeddings should be computed.
    param_file: Parameter file, which is needed to load the corpus.

    Returns
    -------

    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # load a corpus
    print('Loading the Corpus...\n')
    corpus = Corpus('Loaded Corpus', param_file=param_file)
    corpus_name = corpus.corpus_name

    # compute the bag of words embeddings for the corpus
    bow = BagOfWords(corpus)

    print('Saving the corpus...')
    embedding_name = corpus_name.replace('.pickl', '.embd')
    bow.save_object(output_path + embedding_name)


if __name__ == '__main__':
    tsv_file_path = 'testData.tsv'
    param_file = 'params.yaml'
    output_path = 'test/'

    # compute_corpus(tsv_file_path, param_file, 'test')
    # create_dataset(tsv_file_path, param_file, output_path)
    # corpus = Corpus('Loaded Corpus', param_file=param_file)
    compute_bag_of_words_embeddings(output_path, param_file)

