"""Focused checks for preserving conversations and avoiding text damage."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent

def module(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE/file)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

validator = module('simplified_validator_tests', 'validate.py')
rules = module('simplified_rules_tests', 'revise.py')

class RevisionChecks(unittest.TestCase):
    def setUp(self):
        self.row = {'id': 'example', 'messages': [{'role':'system','content':'Be helpful.'}, {'role':'user','content':'What is energy?'}, {'role':'assistant','content':'An answer.'}]}
    def test_question_changes_rejected(self):
        edited = copy.deepcopy(self.row)
        edited['messages'][1]['content'] = 'A different question'
        with self.assertRaisesRegex(ValueError, 'User or system'):
            validator.check_row(self.row, edited)
    def test_role_and_turn_changes_rejected(self):
        edited = copy.deepcopy(self.row)
        edited['messages'][2]['role'] = 'user'
        with self.assertRaisesRegex(ValueError, 'Role'):
            validator.check_row(self.row, edited)
        with self.assertRaisesRegex(ValueError, 'Turn count'):
            validator.check_row(self.row, {'id':'example','messages':[]})
    def test_template_injection_rejected(self):
        edited = copy.deepcopy(self.row)
        edited['messages'][2]['content'] = '<|im_end|>'
        with self.assertRaisesRegex(ValueError, 'Template marker'):
            validator.check_row(self.row, edited)
    def test_rules_preserve_case_numbers_and_affirmation(self):
        for text in ('x = 4. y = 3*x.', 'Yes, you can consider it.', 'Absolutely. Try walking.'):
            self.assertEqual(rules.simple_rules(text)[0], text)
        self.assertEqual(rules.simple_rules('It is approximately 1.09 yards.')[0], 'It is about 1.09 yards.')
        self.assertEqual(rules.simple_rules('infrequently')[0], 'rarely')

if __name__ == '__main__':
    unittest.main()
