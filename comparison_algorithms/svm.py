import numpy as np
import pandas as pd
import os
import sys
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

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

class SVMWrapper(ClassificationWrapper):
    """Wrapper class for performing a grid search over hyperparameter settings for a Support
    Vector Machine classifier. Inherits from the generic ClassificationWrapper"""
    def __init__(self, filename, c_vals=[0.1, 1.0, 10.0, 100.0], beta_vals=[.0001, .001, .01, .1, 1.0], 
                 kernels=['linear', 'rbf'], wanted_label='tomorrow_Group_Happiness_Evening_Label', 
                 cont=False, classifier_name='SVM', num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, 
                 dropbox_path=DEFAULT_MAIN_DIRECTORY, datasets_path='Data/', results_path=None, 
                 check_test=True, normalize_and_fill=False, normalization='between_0_and_1', 
                 optimize_for='val_acc', min_or_max='max', 
                 save_results_every_nth=1, check_noisy_data=True, cross_validation=True):
        """Initializes both the SVMWrapper and its parent.

        Args: almost entirely the same as the parent class, except:
            c_vals: Settings for the C hyperparameter that will be tested. 
            beta_vals: Settings for the kernel width parameter (beta) that will be tested.
            kernels: String names of kernels to test. Can be 'linear', 'rbf', or 'poly'.
        """
        # Hyperparameters to test
        self.c_vals = c_vals
        self.beta_vals=beta_vals
        self.kernels = kernels

        ClassificationWrapper.__init__(self, filename=filename, wanted_label=wanted_label, cont=cont, 
            classifier_name=classifier_name, num_cross_folds=num_cross_folds, dropbox_path=dropbox_path, 
            datasets_path=datasets_path, results_path=results_path, check_test=check_test, 
            normalize_and_fill=normalize_and_fill, normalization=normalization, 
            optimize_for=optimize_for, min_or_max=min_or_max, cross_validation=cross_validation,
            save_results_every_nth=save_results_every_nth, check_noisy_data=check_noisy_data)

        self.trim_extra_linear_params()
        self.model = None

    def load_data(self):
        """Loads data from csv files using the DataLoader class. Must change labels to be {-1,1}."""
        self.data_loader = data_funcs.DataLoader(self.datasets_path + self.filename, 
                                                 normalize_and_fill=self.normalize_and_fill,
                                                 cross_validation=True,
                                                 supervised=True,
                                                 wanted_label=self.wanted_label,
                                                 normalization=self.normalization,
                                                 labels_to_sign=True,
                                                 separate_noisy_data=self.check_noisy_data)

    def trim_extra_linear_params(self):
        """Linear kernels cannot use the beta parameter, so removes any redundant settings 
        involving a linear kernel and different settings of beta."""
        single_beta = None
        i = 0
        while i < len(self.list_of_param_settings):
            setting = self.list_of_param_settings[i]
            if setting['kernel'] == 'linear':
                if single_beta is None:
                    single_beta = setting['beta']
                elif setting['beta'] != single_beta:
                    self.list_of_param_settings.remove(setting)
                    continue
            i += 1
        self.num_settings = len(self.list_of_param_settings)

    def define_params(self):
        """Defines the list of hyperparameters that will be tested."""
        self.params = {}
        self.params['C'] = self.c_vals
        self.params['beta'] = self.beta_vals
        self.params['kernel'] = self.kernels

    def predict_on_data(self, X):
        """Gets the classification predictions on some data X. 

        Args:
            X: a matrix of data

        Returns: the predicted Y labels.
        """
        try:
            preds = self.model.predict(X)
        except:
            # predict majority class
            preds = np.sign(np.mean(self.data_loader.train_Y))*np.ones(len(predict_X))
        return preds
    
    def train_and_predict(self, param_dict, predict_on='val'):
        """Initializes an SVM classifier according to the desired parameter settings, 
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
        
        self.model = SVC(C=param_dict['C'], kernel=param_dict['kernel'], gamma=param_dict['beta'])
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
    print "SVM MODEL SELECTION"
    print "\tThis code will sweep a set of parameters to find the ideal settings for an SVM on a single dataset"

    if Z_SCORE_FILL_WITH_0:
        normalize_and_fill = True
        datasets_path = 'Data/'
        normalization = 'z_score'
    else:
        normalize_and_fill = False
        datasets_path = 'Data/Cleaned/'
        normalization = 'between_0_and_1'

    if len(sys.argv) < 3:
        print "Error: usage is python svm.py <filename> <label> <continue>"
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

    wrapper = SVMWrapper(filename, dropbox_path=DEFAULT_MAIN_DIRECTORY, datasets_path=datasets_path,
                          cont=cont, wanted_label=label, normalize_and_fill=normalize_and_fill,
                          normalization=normalization)

    print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'

    wrapper.run()