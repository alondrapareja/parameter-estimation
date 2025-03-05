import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize #Found this function on ChatGBT to help with minimizing the negative log likelihood


class SimplifiedThreePL:
   def __init__(self, experiment):
       #Checks if the input is valid
       if not hasattr(experiment, 'n_correct') or not hasattr(experiment, 'n_incorrect') or not hasattr(experiment, 'theta_i') or not hasattr(experiment, 'b_i'):
           raise TypeError("There is invalid experiment data. Required attributes are missing") #Used ChatGBT to check for validity function
      
       self.experiment = experiment
       self.difficulty_params = np.array([2, 1, 0, -1, -2])
       self._is_fitted = False
       self._discrimination = None
       self._logit_base_rate = None

   #Calculates correct, incorrect, and total number of trails. Includes number of conditons
   def summary(self):
       n_total = np.sum(self.experiment.n_correct + self.experiment.n_incorrect)
       n_correct = np.sum(self.experiment.n_correct)
       n_incorrect = np.sum(self.experiment.n_incorrect)
       n_conditions = len(self.experiment.n_correct)


       return {
           'n_total': n_total,
           'n_correct': n_correct,
           'n_incorrect': n_incorrect,
           'n_conditions': n_conditions
       }

   def predict(self, parameters):
       a_i = parameters[0]  #Discrimination
       logit_c_i = parameters[1]  #Logit transformed base rate
      
       #Computes inverse logit to get base rate
       c_i = 1 / (1 + np.exp(-logit_c_i))
      
       #Calculates probability of correct response for every condition
       probabilities = c_i + (1 - c_i) * (1 / (1 + np.exp(-a_i * (self.experiment.theta_i - self.experiment.b_i))))

       return probabilities


   def negative_log_likelihood(self, parameters):
       a_i = parameters[0]  #Discrimination
       logit_c_i = parameters[1] #Logit transformed base rate
      
       #Computes inverse logit to get base rate
       c_i = 1 / (1 + np.exp(-logit_c_i))
      
       #Calculates probability of correct and incorrect response for every condition
       P_correct = c_i + (1 - c_i) * (1 / (1 + np.exp(-a_i * (self.experiment.theta_i - self.experiment.b_i))))
       P_incorrect = 1 - P_correct
      
       #Calculates negative log likelihood
       nll = -np.sum(self.experiment.n_correct * np.log(P_correct) + self.experiment.n_incorrect * np.log(P_incorrect))
      
       return nll


   def fit(self):
       initial_params = np.array([0.5, 0.0])  #Starts guesses for a_i and logit_c_i
      
       #Found this method on ChatGBT along with the minimize function to help with minimizing the negative log likelihood)
       result = minimize(self.negative_log_likelihood, initial_params, method='L-BFGS-B', bounds=[(0, None), (None, None)])
      
       if not result.success:
           raise RuntimeError("The optimization has failed")


       #Extracts estimated parameters
       est_a = result.x[0]  #Estimated discrimination
       est_logit_c = result.x[1]  #Estimated logit of base rate
      
       #Converts logit to base rate
       self._discrimination = est_a
       self._logit_base_rate = est_logit_c
       self._base_rate = 1 / (1 + np.exp(-est_logit_c))
      
       self._is_fitted = True  #Marks model as fitted


   def get_discrimination(self):
       if not self._is_fitted:
           raise ValueError("The model is not fitted")
       return self._discrimination


   def get_base_rate(self):
       if not self._is_fitted:
           raise ValueError("The model is not fitted")
       return self._base_rate


   #Makes sure you can't access private attributes
   def __getattr__(self, name):
       if name.startswith('_'):
           raise AttributeError(f"The private attribute '{name}' cannot be accessed directly")
       raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")