import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize #Found this function on ChatGBT to help with minimizing the negative log likelihood

class SimplifiedThreePL:
    def __init__(self, experiment):
        self.experiment = experiment
        self.difficulty_params = np.array([2, 1, 0, -1, -2]) 
        self._is_fitted = False
        self._discrimination = None
        self._logit_base_rate = None

    #Calculates correct, incorrect, and total number of trails. Includes number of conditons.
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
    #Calculates the probability of correct responses for each condition depending on the parameters 
    def predict(self, parameters):

        a_i = parameters[0]  #Discrimination 
        logit_c_i = parameters[1]  #Logit transformed base rate 
        
        #Computes inverse logit to get base rate
        c_i = 1 / (1 + np.exp(-logit_c_i))
        
        #Calculates probability of correct response for every condition
        probabilities = c_i + (1 - c_i) * (1 / (1 + np.exp(-a_i * (self.experiment.theta_i - self.experiment.b_i))))

        return probabilities

    #Computes the negative log-likelihood of the data given the parameters    
    def negative_log_likelihood(self, parameters):
            a_i = parameters[0]  #Discrimination 
            logit_c_i = parameters[1]  #Logit transformed base rate 
            
            #Computes inverse logit to get base rate
            c_i = 1 / (1 + np.exp(-logit_c_i))
            
            #Calculates probability of correct and incorrect response for every condition
            P_correct = c_i + (1 - c_i) * (1 / (1 + np.exp(-a_i * (self.experiment.theta_i - self.experiment.b_i))))
            P_incorrect = 1 - P_correct
            
            #Calculates negative log likelihood 
            nll = -np.sum(self.experiment.n_correct * np.log(P_correct) + self.experiment.n_incorrect * np.log(P_incorrect))
            
            return nll

    #Implements maximum likelihood estimation to find the best fitting discrimination parameter and base rate parameter        
    def fit(self):
            #Uses an initial guess for the parameters
            initial_params = np.zeros(2)
            
            #Minimizes negative log likelihood function to find the best fitting parameters
            result = minimize(self.negative_log_likelihood, initial_params, method='BFGS') #Found this method on ChatGBT along with the minimize function to help with minimizing the negative log likelihood
            
            #Extracts the estimated parameters from the result
            est_a = result.x[0]  #Estimates discrimination
            est_logit_c = result.x[1]  #Estimates logit of base rate
            
            #Stores the estimated parameters
            self._discrimination = est_a
            self._logit_base_rate = est_logit_c
            self._base_rate = 1 / (1 + np.exp(-est_logit_c))  #Converts logit back to base rate
            
            self._is_fitted = True  #Marks the model as fitted
        
    #Returns estimate of discrimination paramater a and raises error if it hasn't been fitted yet
    def get_discrimination(self):
            if not self._is_fitted:
                raise ValueError("The model is not fitted yet.")
            return self._discrimination

    #Retuns estimate of base rate parameter c and raises errror if it hasn't been fitted yet.    
    def get_base_rate(self):
            if not self._is_fitted:
                raise ValueError("The model is not fitted yet.")
            return self._base_rate

