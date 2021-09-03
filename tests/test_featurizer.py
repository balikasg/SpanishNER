import unittest
from unittest import mock
import argparse
from pathlib import Path
from tagger.feature_extraction import transform_entity_file,\
    get_raw_data, get_training_format, define_arguments


class TestFeaturizer(unittest.TestCase):
    def setUp(self):
        self.test_data_dir = Path.cwd() / "tests" / "data"

    def test_tranform_entity_file(self):
        content = (self.test_data_dir / "test.ann").read_text()
        actual = transform_entity_file(content)
        expected = [{'label': 'PROFESION', 'begin_idx': 31, 'end_idx': 40, 'passage': 'enfermero'}]
        self.assertEqual(actual, expected)

    def test_get_raw_data(self):
        actual = get_raw_data(self.test_data_dir)
        expected = {'test': {'text': 'Anamnesis\nPaciente de 33 años, enfermero de profesión.',
                             'ann': [{'label': 'PROFESION', 'begin_idx': 31,
                                      'end_idx': 40, 'passage': 'enfermero'}]}}
        self.assertDictEqual(actual, expected)

    def test_get_training_format(self):
        raw = get_raw_data(self.test_data_dir)
        actual = get_training_format(raw)
        expected = {'test': {'tokens': ['Anamnesis', '\n', 'Paciente', 'de', '33', 'años', ',', 'enfermero', 'de', 'profesión', '.'],
                             'ner': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PROFESION', 'O', 'O', 'O'],
                             'pos': ['PROPN', 'SPACE', 'PROPN', 'ADP', 'NUM', 'NOUN', 'PUNCT', 'NOUN', 'ADP', 'NOUN', 'PUNCT'],
                             'dep': ['ROOT', 'punct', 'appos', 'case', 'nummod', 'nmod', 'punct', 'appos', 'case', 'nmod', 'punct'],
                             'lemma': ['Anamnesis', '\n', 'Paciente', 'de', '33', 'año', ',', 'enfermero', 'de', 'profesión', '.'],
                             'sid': [0, 9, 10, 19, 22, 25, 29, 31, 41, 44, 53]}}
        self.assertDictEqual(actual, expected)

    @mock.patch('argparse.ArgumentParser.parse_args',
                         return_value=argparse.Namespace(input_dir='test_dir'))
    def test_argparser(self, mock_args):
        args = define_arguments()
        assert args.input_dir == "test_dir"
