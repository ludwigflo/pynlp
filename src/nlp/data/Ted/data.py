import urllib.request
import numpy as np
import lxml.etree
import zipfile
import urllib
import h5py
import os
import re


def extract_tags(base_dir: str) -> None:
    """
    """

    # create a list with all required tag names
    tag_list = ['content', 'keywords', 'title', 'description']

    # get all talk ids
    id_list = os.listdir(base_dir)

    # iterate through all talk ids
    for file_id in id_list:

        # get the current directory
        dir_act = base_dir + file_id + '/'

        # get all language directories
        languages = os.listdir(dir_act)

        # iterate through all languages in the current talk id
        for language in languages:

            # get the current xml file
            xml_file = dir_act + language + '/file.xml'

            # create the tag files
            for tag in tag_list:
                save_tag(dir_act+language+'/', xml_file, tag_name=tag)


def save_tag(output_path: str, ted_file: str, tag_name) -> None:
    """
    """

    tree = lxml.etree.parse(ted_file)
    for elem in tree.iter():
        if elem.tag == tag_name:
            tag_text = elem.text
            with open(output_path + tag_name +'.txt', 'w') as f:
                f.write(tag_text)


def split_ted_files(output_path: str, ted_in_file: str) -> None:
    """
    """

    tree = lxml.etree.parse(ted_in_file)


    language = 'no_info'
    # get the language of the current xml ted file
    for elem in tree.iter():
        if elem.tag == 'xml':
            language = elem.attrib['language']

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for elem in tree.iter():

        # find a new file
        if elem.tag == 'file':

            # extract a subtree from with the file as root
            sub_tree = lxml.etree.ElementTree(elem)

            # iterate through the sub tree
            for child in sub_tree.iter():

                # find the talk id of the subtree
                if child.tag == 'talkid':
                    talk_id = child.text

                    # create a directory based on the talk id
                    if not os.path.exists(output_path+talk_id+'/'+language+'/'):
                        os.makedirs(output_path+talk_id+'/'+language+'/')

                    # save the subtree in the created directory
                    sub_tree.write(output_path+talk_id+'/'+language+'/file.xml', pretty_print=True)


def download_ted_dataset(link=None, file_name=None) -> None:
    """
    Downloads a dataset, if it doesn't already exist.

    Parameters
    ----------
        link: Link of the file, which should be downloaded.
        file_name: Name of the downloaded file.
    """

    if link is None:
        link = "https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip"
    if file_name is None:
        file_name = 'ted_en-20160408.zip'

    # Download the dataset if it's not already there: this may take a minute as it is 75MB
    if not os.path.isfile(file_name):
        urllib.request.urlretrieve(link, filename=file_name)


def get_raw_text(file_name: str) -> str:
    """
    Returns the raw input text (in english) of the Ted talk series.

    Returns
    -------
        ted_text: Text of the Ted talk series in english.
    """

    # extract the subtitle text of the ted talks from the XML:
    with zipfile.ZipFile(file_name, 'r') as z:
        doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))
    ted_text = '\n'.join(doc.xpath('//content/text()'))
    return ted_text


def tokenize_ted_file(file_name: str) -> list:
    """
    """

    with open(file_name, 'r') as f:
        lines = f.readlines()
        sentences = []
        for line in lines:
            line = line.replace('TED Talk Subtitles and Transcript:', '')
            line = re.split('(?<!\d)[.;](?!\d)', line)
            line = [re.sub(r'\(.*?\)', '', x).rstrip() for x in line]
            line = [x for x in line if len(x)>0]
            sentences.extend(line)
    return sentences


def split_sentences_ted_dataset(file_path: str, file_name: str, base_path: str) -> None:
    """
    """

    # create the output path, if it doesn't exist
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    # get all Ted Talks
    talk_folder_names = os.listdir(base_path)
    talk_folder_names = sorted([int(x) for x in talk_folder_names])
    talk_folder_names = [str(x) for x in talk_folder_names]

    # create the hdf5 data file
    hdf5_file = h5py.File(file_path + file_name, 'w')

    # iterate through all folders
    for i, folder_name in enumerate(talk_folder_names):

        print(i)

        # create a document group in the hdf5 file
        document_group = hdf5_file.create_group('document: ' + str(i))

        # get all languages of the current talk
        path_act = base_path + folder_name + '/'
        languages = os.listdir(path_act)

        # iterate through all languages
        for language in languages:

            # create a language group in the current document group
            language_group = document_group.create_group('language: ' + language)

            # get all text_files in the current language group
            language_path_act = path_act + language + '/'
            text_files = [x for x in os.listdir(language_path_act) if x.endswith('.txt')]

            # iterate through all text files
            for text_file in text_files:

                # extract a list of sentences of the current talk, language and text file
                sentences = tokenize_ted_file(language_path_act + text_file)
                sentences = np.array(sentences, dtype=object)

                # create a dataset for the current talk, language and text_file
                string_dt = h5py.special_dtype(vlen=str)
                text_file = text_file.replace('.txt', '')
                language_group.create_dataset(text_file, data=sentences, dtype=string_dt)


base_path = '/media/data1/flo/NLP/src/nlp/data/Ted/raw/'
file_path = '/media/data1/flo/NLP/src/nlp/data/Ted/sentence_split/'
file_name = 'ted.hdf5'
split_sentences_ted_dataset(file_path, file_name, base_path)
