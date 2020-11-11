from collections import defaultdict
from PIL import Image
import numpy as np
import os

def images_preprocess_generator(images_list,images_path):
    '''
    images_path: path to folder containing images
    images_list: list of names of images
    '''
    for image_name in images_list:
        ############################################################################
        # TODO: Yield image with normalized values (1/255) as a numpy array
        #       of shape (1,299,299,3) for InceptionV3.
        # Hint: You may meed to use PIL.Image to open image as array
        # Note: If you use any other network, you have to chose its compatable shape.
        ############################################################################
        
        ############################################################################
        #END TODO
        ############################################################################
        

def load_images_list(filepath):
    '''
    Load names of train/test/val images into a list from a text file
    '''
    images_txt = open(filepath,'r')
    images_list = []
    for line in images_txt:
        images_list.append(line.strip())
    return images_list

def load_descriptions(filepath):
    '''
    Loads captions of each image into a dictionary along with startseq and endseq words appended.
    You may look into token.txt file to see the captions corresponding to each image.
    '''
    descriptions_dict = defaultdict(list)
    descriptions_txt = open(filepath,'r')
    for line in descriptions_txt:
        image, description = line.strip().split('#')[0], line.strip().split('\t')[1]
        descriptions_dict[image].append('seqstart ' + description + ' seqend')
    return descriptions_dict

def clean_descriptions(descriptions_dict):
    '''
    Remove punctuations and convert all words to lower case for consistancy. Also, calculates max length across all captions.
    '''
    max_len = 0
    for image, sentences_list in descriptions_dict.items():
        for i in range(len(sentences_list)):
            sentence = sentences_list[i]
            # tokenize
            sentence = sentence.split()
            # convert to lower case
            sentence = [word.lower() for word in sentence]
            # remove punctuation and numbers
            sentence = [w for w in sentence if w.isalpha()]
            max_len = max(max_len,len(sentence))
            # store as string
            sentences_list[i] =  ' '.join(sentence)
    return descriptions_dict, max_len

def generate_vocabulary(descriptions_dict):
    '''
    Generates vocabualary from all the captions. Threshold is used to filter out uncommon words.
    '''
    vocab_count = defaultdict(int)
    for image,sentences in descriptions_dict.items():
        for sentence in sentences:
            for word in sentence.split():
                if word.isalpha():
                    vocab_count[word.lower()] += 1
    return vocab_count

def word_indexing(vocab_list):
    '''
    Create dictionaries for words and ids inter conversion
    '''
    word_to_id = {}
    id_to_word = {}
    for i,word in enumerate(vocab_list,1):
        word_to_id[word] = i
        id_to_word[i] = word
    return word_to_id, id_to_word