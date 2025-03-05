import unittest
import numpy as np
from src.SimplifiedThreePL import SimplifiedThreePL
from src.Experiment import Experiment

class TestSimplifiedThreePL(unittest.TestCase):
   def setUp(self):
      #Creating Experiment for testing
      class Experiment:
          def __init__(self):
              self.n_correct = np.array([55, 60, 75, 90, 95])
              self.n_incorrect = 100 - self.n_correct
              self.theta_i = np.array([0, 1, 2, 3, 4])
              self.b_i = np.array([1, 2, 3, 4, 5])
    
      self.experiment = Experiment()
      self.model = SimplifiedThreePL(self.experiment)

   #Initialization Tests
   def test_constructor_valid_input(self):
      self.assertIsInstance(self.model, SimplifiedThreePL)

   def test_constructor_invalid_input(self):
      with self.assertRaises(TypeError):
         SimplifiedThreePL(None)

   #Prediction Tests
   def test_predict_probability_range(self):
      parameters = [1.0, 0.0]
      probabilities = self.model.predict(parameters)
      self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

   def test_predict_higher_base_rate_increases_probability(self):
      prob_high = self.model.predict([1.0, 1.0])
      prob_low = self.model.predict([1.0, -1.0])
      self.assertTrue(np.all(prob_high > prob_low))

   def test_predict_higher_difficulty_lowers_probability(self):
      prob_positive_a = self.model.predict([1.0, 0.0])
      prob_negative_a = self.model.predict([-1.0, 0.0])
      self.assertTrue(np.all(prob_negative_a > prob_positive_a))

   def test_predict_higher_ability_increases_probability(self):
      prob_high_ability = self.model.predict([2.0, 0.0])
      prob_low_ability = self.model.predict([0.5, 0.0])
      self.assertTrue(np.all(prob_high_ability > prob_low_ability))

   def test_predict_matches_expected_output(self):
      expected_probabilities = np.array([0.55, 0.60, 0.75, 0.90, 0.95])
      predicted = self.model.predict([1.0, 0.0])
      np.testing.assert_almost_equal(predicted, expected_probabilities, decimal=2)

   #Parameter Estimation Tests
   def test_negative_log_likelihood(self):
      initial_nll = self.model.negative_log_likelihood([0.5, 0.0])
      self.model.fit()
      final_nll = self.model.negative_log_likelihood([self.model.get_discrimination(), self.model._logit_base_rate])
      self.assertLess(final_nll, initial_nll)

   def test_larger_estimate_for_steeper_curve(self):
      self.model.fit()
      est_a = self.model.get_discrimination()
      self.assertGreater(est_a, 0.0)

   def test_accessing_parameters_before_fit(self):
      with self.assertRaises(ValueError):
          self.model.get_discrimination()
      with self.assertRaises(ValueError):
          self.model.get_base_rate()

   #Integration Tests
   def test_stable_parameter_estimates(self):
      self.model.fit()
      initial_discrimination = self.model.get_discrimination()
      initial_base_rate = self.model.get_base_rate()
      self.model.fit()
      final_discrimination = self.model.get_discrimination()
      final_base_rate = self.model.get_base_rate()
      self.assertAlmostEqual(initial_discrimination, final_discrimination, delta=0.01)
      self.assertAlmostEqual(initial_base_rate, final_base_rate, delta=0.01)

   def test_integration_with_known_data(self):
      conditions_accuracy = np.array([0.55, 0.60, 0.75, 0.90, 0.95])
      self.experiment.n_correct = np.array([int(accuracy * 100) for accuracy in conditions_accuracy])
      self.experiment.n_incorrect = 100 - self.experiment.n_correct
      self.model.fit()
      predictions = self.model.predict([self.model.get_discrimination(), self.model._logit_base_rate])
      np.testing.assert_almost_equal(predictions, conditions_accuracy, decimal=2)

   #Corruption Tests
   def test_private_attribute_access(self):
      with self.assertRaises(AttributeError):
          _ = self.model._discrimination


if __name__ == "__main__":
  unittest.main()







