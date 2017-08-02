import numpy as np
import pandas as pd
import os
import sys
from sklearn import linear_model

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_MAIN_DIRECTORY = '/Your/path/here/'

DEFAULT_NUM_CROSS_FOLDS = 5
Z_SCORE_FILL_WITH_0 = True

import data_funcs
import helper_funcs as helper
from generic_wrapper import ClassificationWrapper

def reload_dependencies():
    reload(data_funcs)
    reload(helper)

class LRWrapper(ClassificationWrapper):
    """Wrapper class for performing a grid search over hyperparameter settings for a Logistic 
    Regression classifier. Inherits from the generic ClassificationWrapper"""
    def __init__(self, filename, penalties=['l1','l2'], c_vals=[.0001,.001,.01,.1,1.0,10.0,100.0],
                 wanted_label='tomorrow_Group_Happiness_Evening_Label',
                 classifier_name='LR', num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, cont=False,
                 dropbox_path=DEFAULT_MAIN_DIRECTORY, datasets_path='Data/Cleaned/', results_path=None, 
                 check_test=True, normalize_and_fill=False, normalization='between_0_and_1', 
                 optimize_for='val_acc', min_or_max='max', save_results_every_nth=1, 
                 check_noisy_data=True, cross_validation=True):
        """Initializes both the LRWrapper and its parent.

        Args: almost entirely the same as the parent class, except:
            penalties: The type of regularization penalties that will be tested. Can be either
                'l1' or 'l2'
            c_vals: Settings for the C hyperparameter that will be tested. 
        """
        # Hyperparameters to test
        self.penalties = penalties
        self.c_vals = c_vals

        ClassificationWrapper.__init__(self, filename=filename, wanted_label=wanted_label, cont=cont, 
            classifier_name=classifier_name, num_cross_folds=num_cross_folds, dropbox_path=dropbox_path, 
            datasets_path=datasets_path, results_path=results_path, check_test=check_test, 
            normalize_and_fill=normalize_and_fill, normalization=normalization, optimize_for=optimize_for, 
            min_or_max=min_or_max, save_results_every_nth=save_results_every_nth, 
            check_noisy_data=check_noisy_data, cross_validation=cross_validation)

        self.model = None

    def define_params(self):
        """Defines the list of hyperparameters that will be tested."""
        self.params = {}
        self.params['penalty'] = self.penalties
        self.params['C'] = self.c_vals

    def predict_on_data(self, X):
        """Gets the classification predictions on some data X. 

        Args:
            X: a matrix of data

        Returns: the predicted Y labels.
        """
        try:
            preds = self.model.predict(X)
        except:
            print "Error! Could not predict using the model. Predicting majority class instead."
            # predict majority class
            preds = np.sign(np.mean(self.data_loader.train_Y))*np.ones(len(X))
        return preds
    
    def train_and_predict(self, param_dict, predict_on='val'):
        """Initializes a LR classifier according to the desired parameter settings, 
        trains it, and returns the predictions on the appropriate evaluation dataset.

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
            predict_on: The dataset used for evaluating the model. Can set to 
                'Test' to get final results.
        
        Returns: The predicted Y labels.
        """
        if predict_on == 'test':
            predict_X = self.data_loader.test_X
        else:
            predict_X = self.data_loader.val_X
        
        self.model = linear_model.LogisticRegression(penalty=param_dict['penalty'], 
                                                C=param_dict['C'])
        self.model.fit(self.data_loader.train_X, self.data_loader.train_Y)
        preds = self.predict_on_data(predict_X)
        
        return preds

    def test_on_test(self, param_dict):
        """Trains the model and tests it on the tests data. 

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.

        Returns: The predicted Y labels.
        """
        return self.train_and_predict(param_dict, predict_on='test')

if __name__ == "__main__":
    print "LR MODEL SELECTION"
    print "\tThis code will sweep a set of parameters to find the ideal settings for a LR model on a single dataset"

    if Z_SCORE_FILL_WITH_0:
        normalize_and_fill = True
        datasets_path = 'Data/'
        normalization = 'z_score'
    else:
        normalize_and_fill = False
        datasets_path = 'Data/Cleaned/'
        normalization = 'between_0_and_1'

    if len(sys.argv) < 3:
        print "Error: usage is python logistic_regression.py <filename> <label> <continue>"
        print "\t<filename>: e.g. all_modalities_present.csv - program will look in the following directory for this file", DEFAULT_MAIN_DIRECTORY + datasets_path
        print "\t<label>: e.g. 'happiness' - the classification label you are trying to predict"
        print "\t<continue>: optional. If 'True', the wrapper will pick up from where it left off by loading a previous validation results file"
        sys.exit()
    filename = sys.argv[1] #get data file from command line argument
    print "\nLoading dataset", DEFAULT_MAIN_DIRECTORY + datasets_path + filename
    print ""

    label = sys.argv[2]

    if len(sys.argv) >= 4 and sys.argv[3] == 'True':
        cont = True
        print "Okay, will continue from a previously saved validation results file for this problem"
    else:
        cont = False
    print ""

    wrapper = LRWrapper(filename, dropbox_path=DEFAULT_MAIN_DIRECTORY, datasets_path=datasets_path,
                        cont=cont, wanted_label=label, normalize_and_fill=normalize_and_fill,
                        normalization=normalization)

    print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'

    wrapper.run()