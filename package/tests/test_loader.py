# test_loader.py

import unittest
from src.core.loader import load_data
import pandas as pd

class TestLoader(unittest.TestCase):

    def setUp(self):
        self.test_file = 'data/workflow.csv'
        self.expected_columns = [
            'jobID', 'taskID', 'CPU', 'RAM', 'disk', 
            'parent_task', 'Runtime_C1', 'Runtime_C2', 
            'Runtime_C3', 'deadline', 'task_type'
        ]

    def test_load_data(self):
        df = load_data(self.test_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), self.expected_columns)
        self.assertFalse(df.empty)

    def test_load_data_invalid_file(self):
        with self.assertRaises(FileNotFoundError):
            load_data('invalid_file.csv')

if __name__ == '__main__':
    unittest.main()