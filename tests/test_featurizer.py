import unittest
from pathlib import Path
from tagger.feature_extraction import transform_entity_file, get_raw_data


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
