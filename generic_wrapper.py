""" Generic wrapper class meant to help search the space of parameters
    for different types of models, save the results, and determine the 
    best parameter settings."""

import numpy as np
import pandas as pd
import os
import sys
import copy
import ast
from time import time
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_MAIN_DIRECTORY = '/Your/path/here/'

DEFAULT_NUM_CROSS_FOLDS = 5
DEFAULT_CLEAN_FILE = 'all_modalities_present.csv'
DEFAULT_NOISY_FILE = 'extra_noisy_data.csv'

import data_funcs
import helper_funcs as helper

def reload_dependencies():
    reload(data_funcs)
    reload(helper)

class Wrapper:
    """This generic parent class defines functions common to any wrapper that must test
    different hyperparameter settings for a model in order to find the best ones. It 
    can be inherited to build wrappers for many types of models.
    
    Flexibly allows the child class to define the names and values of all of the 
    hyperparameters it needs to test in a dictionary.
    """
    def __init__(self, filename, cont=False, classifier_name='MMAE', 
                 num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, dropbox_path=DEFAULT_MAIN_DIRECTORY, 
                 datasets_path='Data/', results_path=None, check_test=False, 
                 normalize_and_fill=False, normalization='between_0_and_1', 
                 optimize_for='val_score', min_or_max='max', 
                 save_results_every_nth=1, cross_validation=True):
        """ 
        Initializes the parent class.

        Args:
            filename: The name of a .csv file containing the data.
            cont: A boolean. If true, will try to load a saved results .csv and continue 
                training on the next unfinished result.
            classifier_name: String name of the classifier trained. Used to know where to save
                results.
            num_cross_folds: An integer number of folds to use in cross validation.
            dropbox_path: The path to the main dropbox directory which contains the results and
                data directories.
            datasets_path: The path from the main dropbox to the datasets directory.
            results_path: The path from the main dropbox to the directory where results should 
                be saved.
            check_test: A boolean. If true, will evaluate final results on held-out test set 
                after running.
            normalize_and_fill: If True, will ask DataLoader to normalize the data and fill 
                missing values.
            normalization: How to normalize the input data. Can be 'z_score', 'between_0_and_1', 
                or None.
            optimize_for: The name of the criteria the wrapper is trying to optimize. 
            min_or_max: A string that can be either 'min' if the wrapper is trying to minimize the
                score on the validation data, or 'max' if it should be maximized.
            save_results_every_nth: An integer representing the number of settings to test before
                writing the results df to a csv file.
            cross_validation: set to False to not use cross validation.
        """
        # memorize arguments and construct paths
        self.filename = filename
        self.cont = cont
        self.classifier_name = classifier_name
        self.num_cross_folds = num_cross_folds
        self.dropbox_path = dropbox_path
        self.datasets_path = dropbox_path + datasets_path
        if results_path is None:
            self.results_path = dropbox_path + 'Results/' + self.classifier_name + '/'
        else:
            self.results_path = dropbox_path + results_path
        self.check_test = check_test
        self.save_results_every_nth = save_results_every_nth
        self.optimize_for = optimize_for
        self.normalize_and_fill = normalize_and_fill
        self.normalization = normalization
        self.min_or_max = min_or_max
        self.cross_validation = cross_validation

        self.save_prefix = self.get_save_prefix(filename, replace=cont)

        self.params = {}
        self.define_params()

        self.load_data()

        self.construct_list_of_params_to_test()
        self.num_settings = len(self.list_of_param_settings)

        #storing the results
        self.time_sum = 0
        if cont:
            self.val_results_df = pd.DataFrame.from_csv(self.results_path + self.save_prefix + '.csv')
            print '\nPrevious validation results df loaded. It has', len(self.val_results_df), "rows"
            self.started_from = len(self.val_results_df)
        else:
            self.val_results_df = pd.DataFrame()
            self.started_from = 0

    # These functions need to be overwritten by the child class
    def define_params(self):
        """ This function should set self.params to a dict where they keys represent names of parameters
            to test (e.g. for SVM, 'C') as they should be saved to the val_results_df, and the values of 
            self.params should be a list of values for the parameter that need to be tested. An example 
            dict:
                self.params['C'] = [1,10,100]
                self.params['beta'] = [.001, .01, .1]
        """
        print "Error! define_params should be overwritten in child class"
        raise NotImplementedError

    def train_and_predict(self, param_dict):
        print "Error! train_model_for_task should be overwritten in child class"
        raise NotImplementedError

    def test_on_test(self, param_dict):
        print "Error! train_model_for_task should be overwritten in child class"
        raise NotImplementedError
    
    # The following functions do not need to be overwritten in the child class.
    def load_data(self):
        """Initialize's the classes data_loader object, which takes care of loading 
        data from a file."""
        self.data_loader = data_funcs.DataLoader(self.datasets_path + self.filename, 
                                                 normalize_and_fill=self.normalize_and_fill,
                                                 cross_validation=self.cross_validation,
                                                 normalization=self.normalization)

    def construct_list_of_params_to_test(self):
        """Will make a class level variable that is a list of parameter dicts.
        Each entry in the list is a dict of parameter settings, 
        eg. {'C'=1.0, 'beta'=.01, ...}. This list represents all of the combinations 
        of hyperparameter settings that need to be tested. 
        """
        self.list_of_param_settings = []
        self.recurse_and_append_params(copy.deepcopy(self.params), {})

    def recurse_and_append_params(self, param_settings_left, this_param_dict, debug=False):
        """Given a dictionary listing all the settings needed for each parameter (key) in the 
        dict, recursively performs a breadth-first-search over all of the possible combinations
        of hyperparameter settings.

        For each setting still left in the dict, creates a node of the breadth-first search tree
        where that hyperparameter is set to that specific setting.

        Saves all of the combinations into the class's list_of_param_settings field.
        
        Args:
            param_settings_left: A dictionary of lists. The keys are parameters
                (like 'C'), the values are the list of settings for those parameters that 
                need to be tested (like [1.0, 10.0, 100.0]). 
            this_param_dict: A dictionary containing a single setting for each parameter. If 
                a parameter is not in this_param_dict's keys, a setting for it has not been chosen yet.
            debug: A Boolean. If True, will print debugging statements.
        """
        if debug: print "Working on a parameter dict containing", this_param_dict
        for key in self.params.keys():
            if key in this_param_dict:
                continue
            else:
                try:
                    this_setting = param_settings_left[key].pop()
                except:
                    print "ERROR! could not pop from param_setting", key, "which is", param_settings_left[key]
                if debug: print "Popped", key, "=", this_setting, "off the params left"
                if len(param_settings_left[key]) > 0:
                    if debug: print "Recursing on remaining parameters", param_settings_left
                    self.recurse_and_append_params(copy.deepcopy(param_settings_left), 
                                                   copy.deepcopy(this_param_dict))
                if debug: print "Placing the popped setting", key, "=", this_setting, "into the parameter dict"
                this_param_dict[key] = this_setting
                
        self.list_of_param_settings.append(this_param_dict)
        if debug: print "Appending parameter dict to list:", this_param_dict, "\n"
    
    def get_save_prefix(self, filename, replace=False):
        """Computes a prefix to use when saving results files based on the classifier
        name and the data file name.

        Args: 
            filename: String name of data file.
            replace: A Boolean. If True, and the code detects an existing results file
                with the same name, it will replace it.
        Returns: The string save prefix. 
        """
        end_loc = filename.find('.')
        prefix = self.classifier_name + '-' + filename[0:end_loc]

        if not replace:
            while os.path.exists(self.results_path + prefix + '.csv'):
                prefix = prefix + '2'
        return prefix

    def setting_already_done(self, param_dict):
        """Returns True if a particular setting of the hyperparameters has already been tested.
        
        Args:
            param_dict: A dictionary representing a setting for all of the hyperparameters.
        Returns: Boolean.
        """
        mini_df = self.val_results_df
        for key in param_dict.keys():
            setting = param_dict[key]
            if type(setting) == list:
                setting = str(setting)
            mini_df = mini_df[mini_df[key] == setting]
            if len(mini_df) == 0:
                return False
        print "Setting already tested"
        return True

    def convert_param_dict_for_use(self, setting_dict):
        """When loading rows from a saved results df in csv format, some 
        of the settings may end up being converted to a string representation
        and need to be converted back to actual numbers and objects.
        
        May need to be overwritten in child class.""" 
        if 'architecture' in setting_dict.keys():
            if type(setting_dict['architecture']) == str:
                setting_dict['architecture'] = ast.literal_eval(setting_dict['architecture'])

        if 'optimizer' in setting_dict.keys():
            if 'GradientDescent' in setting_dict['optimizer']:
                setting_dict['optimizer'] = tf.train.GradientDescentOptimizer
            elif 'Adagrad' in setting_dict['optimizer']:
                setting_dict['optimizer'] = tf.train.AdagradOptimizer
            else:
                setting_dict['optimizer'] = tf.train.AdamOptimizer

        if 'batch_size' in setting_dict.keys():
            setting_dict['batch_size'] = int(setting_dict['batch_size'])
            print "batch size just got changed in convert_param_dict. It's an", type(setting_dict['batch_size'])
        return setting_dict

    def sweep_all_parameters(self):
        """Runs through all of the computed combinations of hyperparameter settings, 
        storing the results of testing with each."""
        print "\nYou have chosen to test a total of", self.num_settings, "settings"
        sys.stdout.flush()

        #sweep all possible combinations of parameters
        for param_dict in self.list_of_param_settings:
            self.test_one_setting(param_dict)
            
        self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

        print "\n--------------PARAMETER SWEEP IS COMPLETE--------------"

    def test_one_setting(self, param_dict):
        """Tests a single setting of the hyperparameters.

        Args:
            param_dict: A dictionary with the hyperparameter names as keys and the values
                they should be set to as values.
        """
        if self.cont and self.setting_already_done(param_dict):
            return

        # Times the computation for each setting
        t0 = time()
        
        results_dict = self.get_cross_validation_results(param_dict)
        self.val_results_df = self.val_results_df.append(results_dict,ignore_index=True)
        
        t1 = time()
        this_time = t1 - t0
        self.time_sum = self.time_sum + this_time
        
        print "\n", self.val_results_df.tail(n=1)
        print "It took", this_time, "seconds to obtain this result"
        self.print_time_estimate()
        
        sys.stdout.flush()

        # Output the results file every few iterations for safekeeping 
        if len(self.val_results_df) % self.save_results_every_nth == 0:
            self.val_results_df.to_csv(self.results_path + self.save_prefix + '.csv')

    def get_cross_validation_results(self, param_dict):
        """Gets the score from testing on each cross validation fold and saves the average.

        Args:
            param_dict: A dictionary with the hyperparameter names as keys and the values
                they should be set to as values.
        """
        scores = []
        for f in range(self.num_cross_folds):
            self.data_loader.set_to_cross_validation_fold(f)
            scores.append(self.train_and_predict(param_dict))
        print "Scores for each fold:", scores
        param_dict[self.optimize_for] = np.mean(scores)
        return param_dict

    def print_time_estimate(self):
        """Prints an estimate of the total time remaining to finish testing all of the
        hyperparameter settings."""
        num_done = len(self.val_results_df)-self.started_from
        num_remaining = self.num_settings - num_done - self.started_from
        avg_time = self.time_sum / num_done
        total_secs_remaining = int(avg_time * num_remaining)
        hours, mins, secs = helper.get_secs_mins_hours_from_secs(total_secs_remaining)

        print "\n", num_done, "settings processed so far,", num_remaining, "left to go"
        print "Estimated time remaining:", hours, "hours", mins, "mins", secs, "secs"

    def find_best_setting(self, optimize_for=None, min_or_max=None):
        """After all testing is finished, locates the row in the results file that 
        contains the best possible hyperparameter settings.

        Args:
            optimize_for: String name of the result column that the wrapper should 
                optimize. Defaults to class value.
            min_or_max: Whether the value of optimize_for should be minimized or 
                maximized. Defaults to class value.
        Returns:
            A dictionary containing the best setting
        """
        if optimize_for is None:
            optimize_for = self.optimize_for
        if min_or_max is None:
            min_or_max = self.min_or_max

        scores = self.val_results_df[optimize_for].tolist()
        if min_or_max == 'min':
            best_score = min(scores)
        else:
            best_score = max(scores)
        best_idx = scores.index(best_score)
        best_setting = self.val_results_df.iloc[best_idx]

        print "\nThe best", optimize_for, "was", best_setting[optimize_for]
        print "It was found with the following settings:"
        print best_setting
        print "\n"

        return best_setting

    def get_final_results(self):
        """Finds the best setting of the hyperparameters, then gets the final results on 
        the test set if the field check_test is set to True."""
        best_setting = self.find_best_setting()
        
        if not self.check_test:
            print "check_test is set to false, Will not evaluate performance on held-out test set."
            return
        print "\nAbout to evaluate results on held-out test set!!"
        print "Will use the settings that produced the best", optimize_for
        
        best_setting = self.convert_param_dict_for_use(best_setting)
        test_score = self.test_on_test(best_setting)

        print "\nFINAL TEST RESULTS:", test_score
        
    def run(self):
        """Tests all of the settings, then finds the best one and possibly tests on the 
        test set."""
        self.sweep_all_parameters()
        self.get_final_results()

class ClassificationWrapper(Wrapper):
    """A class that inherits from the generic wrapper, and provides another abstract
    parent class that can be used to easily build wrappers that test classification
    models.
    """
    def __init__(self, filename, wanted_label=None, cont=False, classifier_name='SVM', 
                num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, dropbox_path=DEFAULT_MAIN_DIRECTORY, 
                datasets_path='Data/', results_path=None, check_test=False, 
                normalize_and_fill=False, normalization='z_score', optimize_for='val_acc', 
                min_or_max='max', save_results_every_nth=1, check_noisy_data=False,
                cross_validation=True):
        """Initializes both the parent ClassificationWrapper and its parent Wrapper.

        Args: almost entirely the same as the parent class, except:
            wanted_label: The name of the column containing the labels that we are trying 
                to classify.
            check_noisy_data: If True, will tell the data loader to separate out the noisy
                data in the data file, and will compute results on this data separately.
        """
        self.wanted_label = wanted_label
        self.check_noisy_data = check_noisy_data

        Wrapper.__init__(self, filename=filename, cont=cont, classifier_name=classifier_name, 
                         num_cross_folds=num_cross_folds, dropbox_path=dropbox_path, 
                         datasets_path=datasets_path, results_path=results_path, 
                         check_test=check_test, normalize_and_fill=normalize_and_fill,
                         normalization=normalization, optimize_for=optimize_for, 
                         min_or_max=min_or_max, save_results_every_nth=save_results_every_nth,
                         cross_validation=cross_validation)
    
    def predict_on_data(self, X):
        print "Error! predict_on_data should be overwritten in child class"
        raise NotImplementedError
    
    def load_data(self):
        """Initializes the data loader object of the class. Specific to classification 
        because the data loader must load supervised data, based on the wanted class label,
        and possibly separate noisy data."""
        self.data_loader = data_funcs.DataLoader(self.datasets_path + self.filename, 
                                                 normalize_and_fill=self.normalize_and_fill,
                                                 cross_validation=self.cross_validation,
                                                 supervised=True,
                                                 wanted_label=self.wanted_label,
                                                 normalization=self.normalization,
                                                 separate_noisy_data=self.check_noisy_data)

    def get_save_prefix(self, filename, replace=False):
        """Overloads the parent function for computing a save prefix by including the class label
        in the name.
        
        Args: same as parent class"""
        end_loc = filename.find('.')
        prefix = self.classifier_name + '-' + filename[0:end_loc]

        if self.wanted_label is not None:
            prefix += '-' + helper.get_friendly_label_name(self.wanted_label)

        if not replace:
            while os.path.exists(self.results_path + prefix + '.csv'):
                prefix = prefix + '2'
        return prefix

    def get_cross_validation_results(self, param_dict):
        """Gets cross validation results specific to classification, by computing a number
        of classification metrics including accuracy, AUC, F1, precision, and recall, and 
        computing scores on both noisy and clean data.
        
        Args:
            param_dict: A dictionary with the hyperparameter names as keys and the values
                they should be set to as values.
        Returns: The same param_dict now containing the numerical results for each metric. 
        """
        all_acc = []
        all_auc = []
        all_f1 = []
        all_precision = []
        all_recall = []

        if self.check_noisy_data:
            noisy_acc = []
            noisy_auc = []

            clean_acc = []
            clean_auc = []

        for f in range(self.num_cross_folds):
            self.data_loader.set_to_cross_validation_fold(f)
            
            preds = self.train_and_predict(param_dict)
            true_y = self.data_loader.val_Y
            if preds is None or true_y is None:
                continue

            acc, auc, f1, precision, recall = compute_all_classification_metrics(preds, true_y)
            all_acc.append(acc)
            all_auc.append(auc)
            all_f1.append(f1)
            all_precision.append(precision)
            all_recall.append(recall)

            if self.check_noisy_data:
                noisy_preds = self.predict_on_data(self.data_loader.noisy_val_X)
                acc, auc, f1, precision, recall = compute_all_classification_metrics(noisy_preds, self.data_loader.noisy_val_Y)
                noisy_acc.append(acc)
                noisy_auc.append(auc)

                clean_preds = self.predict_on_data(self.data_loader.clean_val_X)
                acc, auc, f1, precision, recall = compute_all_classification_metrics(clean_preds, self.data_loader.clean_val_Y)
                clean_acc.append(acc)
                clean_auc.append(auc)

        print "Accuracy for each fold:", all_auc
        param_dict['val_acc'] = np.nanmean(all_acc)
        param_dict['val_auc'] = np.nanmean(all_auc)
        param_dict['val_f1'] = np.nanmean(all_f1)
        param_dict['val_precision'] = np.nanmean(all_precision)
        param_dict['val_recall'] = np.nanmean(all_recall)

        if self.check_noisy_data:
            param_dict['noisy_val_acc'] = np.nanmean(noisy_acc)
            param_dict['noisy_val_auc'] = np.nanmean(noisy_auc)
            print "Perf on noisy data:", np.nanmean(noisy_acc), "acc", np.nanmean(noisy_auc), "auc"
            param_dict['clean_val_acc'] = np.nanmean(clean_acc)
            param_dict['clean_val_auc'] = np.nanmean(clean_auc)
            print "Perf on clean data:", np.nanmean(clean_acc), "acc", np.nanmean(clean_auc), "auc"

        return param_dict
    
    def get_classification_predictions_from_df(self):
        """Will predict the class labels for the data contained in the wrapper's data loader
        object. 

        Returns: A pandas dataframe containing the data loader's data with class label 
            predictions added as an extra column.
        """
        df = copy.deepcopy(self.data_loader.df)
        X = df[self.data_loader.wanted_feats].as_matrix()
        preds = self.predict_on_data(X)
        assert(len(X) == len(preds))
        for i,label in enumerate(self.data_loader.wanted_labels):
            df['predictions_'+label] = preds[:,i]
        return df

    def get_final_results(self):
        """Finds the best setting for a number of different metrics, may test on the held-out
        test set if the check_test field is True."""
        for metric in ['val_acc', 'noisy_val_acc', 'clean_val_acc']:
            if metric in self.val_results_df.columns.values:
                best_setting = self.find_best_setting(optimize_for=metric, min_or_max='max')
                print "\nThe best", metric, "was", best_setting[metric]
                print "It was found with the following settings:"
                print best_setting

        if not self.check_test:
            print "check_test is set to false, Will not evaluate performance on held-out test set."
            return
        print "\nAbout to evaluate results on held-out test set!!"
        print "Will use the settings that produced the best", self.optimize_for
        
        best_setting = self.convert_param_dict_for_use(best_setting)
        preds = self.test_on_test(best_setting)
        true_y = true_y = self.data_loader.test_Y
        acc, auc, f1, precision, recall = compute_all_classification_metrics(preds, true_y)

        print "\nFINAL TEST RESULTS ON ALL DATA:"
        print 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall

        if self.check_noisy_data:
            noisy_preds = self.predict_on_data(self.data_loader.noisy_test_X)
            acc, auc, f1, precision, recall = compute_all_classification_metrics(noisy_preds, self.data_loader.noisy_test_Y)
            print "\nFINAL TEST RESULTS ON NOISY DATA:"
            print 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall

            clean_preds = self.predict_on_data(self.data_loader.clean_test_X)
            acc, auc, f1, precision, recall = compute_all_classification_metrics(clean_preds, self.data_loader.clean_test_Y)
            print "\nFINAL TEST RESULTS ON CLEAN DATA:"
            print 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall

def get_baseline(Y):
    """Gets the proportion of the class label that is most frequent in the data.
    
    Args:
        Y: an array of class labels
    Returns: The percent labels that belong to the most frequence class.
    """
    Y = Y.tolist()
    percent_true = float(Y.count(1.0)) / float(len(Y))
    if percent_true < 0.5:
        return 1.0 - percent_true
    else:
        return percent_true

def compute_classification_metric(metric, true_y, preds):
    """Computes a classification metric such as F1 score given the true and predicted labels.

    Args:
        metric: A function that computes a classfiication metric.
        true_y: The ground truth labels.
        preds: The model's predicted labels.
    Returns: The metric score.
    """
    try:
        result = metric(true_y, preds)
    except Exception, e:
        print "Error in computing metric:", e
        return np.nan
    return result

def binary_accuracy(true_y, preds):
    """Computes the percentage of labels that were correctly predicted by the model. 

    Args: 
        true_y: The ground truth labels.
        preds: The model's predicted labels.
    Returns: Float accuracy.
    """
    assert len(preds)==len(true_y)
    correct_labels = [1 for i in range(len(preds)) if preds[i]==true_y[i]]
    return len(correct_labels)/float(len(preds))
    
def compute_all_classification_metrics(preds, true_y):
    """Computes the accuracy, AUC, F1, precision, and recall for the model's predictions. 

    Args:
        true_y: The ground truth labels.
        preds: The model's predicted labels.
    Returns: float accuracy, AUC, F1, precision, and recall
    """
    acc = compute_classification_metric(binary_accuracy, true_y, preds)
    auc = compute_classification_metric(roc_auc_score, true_y, preds)
    f1 = compute_classification_metric(f1_score, true_y, preds)
    precision = compute_classification_metric(precision_score, true_y, preds)
    recall = compute_classification_metric(recall_score, true_y, preds)
    return acc, auc, f1, precision, recall