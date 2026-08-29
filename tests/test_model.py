"""
Unit tests for customer churn model and preprocessing.
"""
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import load_data, clean_total_charges, create_tenure_bins
from src.model import load_model


class TestPreprocessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        cls.data_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'data', 
            'raw', 
            'Customer-Churn.csv'
        )
        cls.df = load_data(cls.data_path)
    
    def test_load_data(self):
        """Test data loading."""
        self.assertIsNotNone(self.df)
        self.assertGreater(len(self.df), 0)
    
    def test_columns_exist(self):
        """Test required columns exist."""
        required_cols = ['customerID', 'Churn', 'tenure', 'TotalCharges']
        for col in required_cols:
            self.assertIn(col, self.df.columns)
    
    def test_clean_total_charges(self):
        """Test TotalCharges cleaning."""
        cleaned = clean_total_charges(self.df)
        self.assertEqual(cleaned['TotalCharges'].dtype, 'float64')
        self.assertEqual(cleaned['TotalCharges'].isnull().sum(), 0)
    
    def test_create_tenure_bins(self):
        """Test tenure binning."""
        binned = create_tenure_bins(self.df)
        self.assertIn('tenure_bin', binned.columns)
        self.assertEqual(binned['tenure_bin'].nunique(), 6)


class TestModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        cls.model_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'models',
            'ada_boost_churn_model.pkl'
        )
    
    def test_model_exists(self):
        """Test model file exists."""
        self.assertTrue(os.path.exists(self.model_path))
    
    def test_model_loads(self):
        """Test model can be loaded."""
        try:
            model = load_model(self.model_path)
            self.assertIsNotNone(model)
        except FileNotFoundError:
            self.skipTest("Model file not found")


if __name__ == '__main__':
    unittest.main()