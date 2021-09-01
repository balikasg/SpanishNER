"""Extracts data in a training friendly format for NER training."""
import os
import json
import argparse
from collections import defaultdict
import spacy


def get_raw_data(sourcepath):
    """Loads data as released and creates a dictionary containing
     both the text and the annotation. The dictionary structure is
      data[some_document_id]["text"]
      data[some_document_id]["ann"]

    sourcepath: directory containing the .txt and .ann files
    """
    data = defaultdict(dict)
    for filename in os.listdir(sourcepath):
        root = filename[:-4]  # Given cc_onco972.txt, keeps cc_onco972
        if filename.endswith('txt'):
            with open(f'{sourcepath}/' +filename, encoding='utf') as infile:
                data[root]['text'] = infile.read()
        if filename.endswith('ann'):
            with open(f'{sourcepath}/' +filename, encoding='utf') as infile:
                labels = infile.read()
            document_entities = []
            for label in labels.split("\n"):  # For each IOB tag
                if label:
                    _, tag_and_indexes, passage = label.split("\t")
                    tag, start_idx, end_idx = tag_and_indexes.split()
                    this_entity = {'label': tag,
                                   'begin_idx':int(start_idx),
                                   'end_idx': int(end_idx),
                                   'passage': passage
                                   }
                    document_entities.append(this_entity)
            data[root]['ann'] = document_entities
    return data


def get_tokens_features(text, nlp, ner_tags=None):
    """Uses spacy to extract features"""
    tokens, pos, ner, sid, dep, lemma = [], [], [], [], [], []
    doc = nlp(text)
    for token in doc:
        tokens.append(token.text)  # tokenizes with spacy
        pos.append(token.pos_)  # part-of-speech
        sid.append(token.idx)  # start index of the token
        dep.append(token.dep_)  # dep parsing
        lemma.append(token.lemma_)  # lemma of the token
    ner = ["O"]*len(tokens)
    if ner_tags:  # Evaluates to true for training data
        for id_, token in enumerate(tokens):
            # For each NER tag we have, validate if the current word is an entity
            # using the start and the end indexes of the data
            for label in ner_tags:
                if sid[id_] >= label['begin_idx'] and sid[id_] < label['end_idx'] :
                    # If the current token in between the tag limits, get IOB tag it!
                    ner[id_] = 'B-' + label['label'] if sid[id_] == label['begin_idx']\
                        else 'I-' + label['label']
    return tokens, pos, ner, dep, lemma, sid


def get_training_format(data):
    """Training format generator. The format looks like:
        data = {
            "doc_id": {
            "tokens": [...], # actual token
            "ner": [...], # IOB tag of token
            "pos": [...], # part of speech of token
            "dep": [...],  # dep. parsing tag of token
            "lemma": [...], # lemma of token
            "sid": [...] # token start index in document
            }
    }
    """
    new_data = defaultdict(dict)
    nlp = spacy.load("es_core_news_sm")
    for _, root in enumerate(data.keys()):
        if 'ann' in data[root]:  # True for training, test do not have .ann files
            tokens, pos, ner, dep, lemma, sid = get_tokens_features(data[root]['text'],
                                                                    nlp,
                                                                    data[root]['ann'])
        else:
            tokens, pos, ner, dep, lemma, sid = get_tokens_features(data[root]['text'], nlp)
        new_data[root]['tokens'] = tokens
        new_data[root]['ner'] = ner
        new_data[root]['pos'] = pos
        new_data[root]['dep'] = dep
        new_data[root]['lemma'] = lemma
        new_data[root]['sid'] = sid
    return new_data


def define_arguments():
    """Defines argparser arguments"""
    first_doc_line = __doc__.strip().split("\n", maxsplit=1)
    parser = argparse.ArgumentParser(description=first_doc_line)
    parser.add_argument("--input-dir",
                        type=str,
                        default="task1/",
                        help="Directory holding the meddoprof data.")
    parser.add_argument("--out-file",
                        type=str,
                        default="meddoprof_data_processed.json",
                        help="File to write the processed data in training-friendly format.")
    return parser

def main():
    """CLI input"""
    parser = define_arguments()
    args = parser.parse_args()
    data = get_raw_data(args.input_dir)
    data = get_training_format(data)
    with open(args.out_file, 'w', encoding='utf8') as out:
        json.dump(data, out, indent=4)


if __name__ == '__main__':
    main()
