import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
from sklearn.svm import SVC

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_MAIN_DIRECTORY = '/Your/path/here/'

import multimodal_autoencoder as mmae
import data_funcs
from generic_wrapper import Wrapper
import generic_wrapper as gen_wrap

def reload_files():
    """Reloads imported dependencies for use with Jupyter notebooks"""
    reload(mmae)
    reload(data_funcs)
    reload(gen_wrap)

DEFAULT_NUM_CROSS_FOLDS = 5
LABELS_TO_PREDICT = ['happiness', 'health', 'calmness']

class MMAEWrapper(Wrapper):
    """A class that inherits from the generic wrapper, enabling the testing and evaluation
    of different hyperparameters settings for use with a Multimodal Autoencoder (MMAE). 
    Performs a grid search over every combination of settings.
    """
    def __init__(self, filename, classification_filename='modalities_missing.csv', 
                 layer_sizes=[[1000,100],[500,100],[300,100]], 
                 tie_weights=[True,False], dropout_probs=[1.0,0.5], weight_penalties=[0.0,.01,.001],
                 weight_initializers=['normal'], activation_funcs=['softsign','relu'], 
                 test_variational=True, cont=False, classifier_name='MMAE', 
                 num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, dropbox_path=DEFAULT_MAIN_DIRECTORY, 
                 datasets_path='Data/Cleaned/', results_path=None, 
                 temp_model_path='Results/temp_saved_models', check_test=False, 
                 optimize_for=None, min_or_max='min', save_results_every_nth=1):
        """Initializes both the MMAEWrapper and its parent Wrapper with some settings. 
        Other class settings like the loss function used to train the models are built
        into the class and have to be changed elsewhere.

        Args: almost entirely the same as the parent class, except:
            classification_filename: String name of a .csv file containing data that can 
                be used to test the classification quality of the MMAE's embeddings. 
            layer_sizes: A list of sizes of the layers used in the encoder to test. 
            tie_weights: A list of settings to test for the 'tie_weights' parameter of
                the MMAE.
            dropout_probs: A list of dropout keep probabilities to test.
            weight_penalities: A list of L2 weight regularization penalties to test.
            weight_initializers: A list of strings describing different ways to initialize
                the weights of the MMAE. 
            activation_funcs: A list of strings describing different activation functions 
                to test within the MMAE. Can contain 'softsign', 'relu', 'tanh', 'softplus', 
                or 'linear'. 
            test_variational: If True, will also construct Variational autoencoders and 
                test those with as many of the settings as are appropriate.
            temp_model_path: A place to save checkpoints of the models as they are being
                trained.
        """
        # Logistics
        self.temp_model_path = dropbox_path + temp_model_path
        if classification_filename is None:
            self.classification_filename = filename
        else:
            self.classification_filename = classification_filename
        
        # Hyperparameters to test
        self.layer_sizes = layer_sizes
        self.tie_weights = tie_weights
        self.dropout_probs = dropout_probs
        self.weight_penalties = weight_penalties
        self.weight_initializers = weight_initializers
        self.activation_funcs = activation_funcs
        self.test_variational = test_variational
        
        # Fixed hyperparameter settings
        self.loss_func = 'sigmoid_cross_entropy'
        self.learning_rate = .001
        self.clip_gradients = True
        self.normalization = 'between_0_and_1'
        self.mask_with = -1.0
        self.fill_missing = 0.0
        self.decay = True
        self.decay_steps = 1000
        self.decay_rate = 0.95
        self.batch_size = 20
        self.optimizer = tf.train.AdamOptimizer
        self.num_steps = 15000
        self.learning_rate = .001
        
        if optimize_for is None:
            optimize_for = 'val_' + self.loss_func

        # Initializes the parent class.
        Wrapper.__init__(self, filename=filename, cont=cont, classifier_name=classifier_name, 
                         num_cross_folds=num_cross_folds, dropbox_path=dropbox_path, 
                         datasets_path=datasets_path, results_path=results_path, 
                         check_test=check_test, optimize_for=optimize_for, min_or_max=min_or_max,
                         normalization=self.normalization, 
                         save_results_every_nth=save_results_every_nth)

        
        if self.test_variational:
            self.add_extra_vae_params()
    
    def load_data(self):
        """Loads data from csv files using the DataLoader class."""
        self.data_loader = data_funcs.DataLoader(self.datasets_path + self.filename, 
                                                 normalize_and_fill=False,
                                                 supervised=False,
                                                 cross_validation=True,
                                                 normalization=self.normalization, 
                                                 fill_missing_with=self.fill_missing)
        
        # Loads additional classification data
        self.classification_data_loader = data_funcs.DataLoader(self.datasets_path + self.classification_filename, 
                                                                normalize_and_fill=False,
                                                                supervised=True,
                                                                cross_validation=True,
                                                                normalization=self.normalization, 
                                                                fill_missing_with=self.fill_missing,
                                                                separate_noisy_data=True)

    def define_params(self):
        """Defines the list of hyperparameters that will be tested."""
        self.params = {}
        self.params['architecture'] = self.layer_sizes
        self.params['tie_weights'] = self.tie_weights
        self.params['dropout_prob'] = self.dropout_probs
        self.params['weight_penalty'] = self.weight_penalties
        self.params['weight_initialization'] = self.weight_initializers
        self.params['activation_function'] = self.activation_funcs
        self.params['variational'] = [False]

    def add_extra_vae_params(self):
        """Since Variational Autoencoders cannot be used with certain settings, 
        adds an additional list of hyperparameter settings that work with VAEs
        """
        for arctect in self.layer_sizes:
            for act_func in self.activation_funcs:
                for dprob in self.dropout_probs:
                    for wpen in self.weight_penalties:
                        for winit in self.weight_initializers:
                            setting_dict = {'activation_function': act_func,
                                            'architecture': arctect,
                                            'dropout_prob': dprob,
                                            'tie_weights': False,
                                            'variational': True,
                                            'weight_initialization': winit,
                                            'weight_penalty': wpen}
                            self.list_of_param_settings.append(setting_dict)
        self.num_settings = len(self.list_of_param_settings)

    def initialize_model(self, param_dict):
        """Initializes an internal instance of an MMAE with the hyperparameter
        settings in param_dict.

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
        """
        self.model = mmae.MultimodalAutoencoder(
            # constant factors that don't change
            batch_size=self.batch_size, learning_rate=self.learning_rate, 
            decay=self.decay, decay_steps=self.decay_steps, decay_rate=self.decay_rate, 
            clip_gradients=self.clip_gradients, normalization=self.normalization,
            subdivide_physiology=True, fill_missing_with=self.fill_missing, mask_with=self.mask_with,
            checkpoint_dir=self.temp_model_path, model_name='MMAE', loss_func=self.loss_func,
            verbose=False,

            # factors that change with param dict
            layer_sizes=param_dict['architecture'], variational=param_dict['variational'],
            tie_weights=param_dict['tie_weights'], dropout_prob=param_dict['dropout_prob'],
            weight_penalty=param_dict['weight_penalty'], 
            activation_func=param_dict['activation_function'],
            weight_initialization=param_dict['weight_initialization'],
            
            # feed in the data
            data_loader=self.data_loader)

    def train_and_predict(self, param_dict):
        """Initializes an MMAE according to the desired parameter settings, trains
        it, and returns the loss obtained on the validation data.

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
        
        Returns: the reconstruction loss obtained on the validation data after
            training.
        """
        self.initialize_model(param_dict)
        self.model.train(self.num_steps, record_every_nth=self.num_steps/10,
                         save_every_nth=self.num_steps+1)
        loss = self.model.get_performance_on_data_with_noise(self.data_loader.val_X)
        print "\tLoss on fold", self.model.data_loader.fold, "was", loss
        return loss

    def test_embedding_classification_quality(self):
        """Using the classification data loader, embeds the training and 
        validation data using the MMAE, then tests how well an SVM can learn
        to classify from the embeddings. 

        Returns 6 floats: the accuracy and AUC on all the data, the noisy data, 
            and the clean data.
        """
        assert len(self.model.val_loss) > 0, "Model needs to be trained before embeddings can be tested"

        feed_dict = {self.model.noisy_X: self.classification_data_loader.train_X,
                     self.model.tf_dropout_prob: 1.0}
        embed_train_X = self.model.session.run(self.model.embedding, feed_dict)

        feed_dict = {self.model.noisy_X: self.classification_data_loader.val_X,
                     self.model.tf_dropout_prob: 1.0}
        embed_X_val = self.model.session.run(self.model.embedding, feed_dict)

        feed_dict = {self.model.noisy_X: self.classification_data_loader.clean_val_X,
                     self.model.tf_dropout_prob: 1.0}
        embed_X_clean = self.model.session.run(self.model.embedding, feed_dict)

        feed_dict = {self.model.noisy_X: self.classification_data_loader.noisy_val_X,
                     self.model.tf_dropout_prob: 1.0}
        embed_X_noisy = self.model.session.run(self.model.embedding, feed_dict)

        label_accs = [np.nan] * len(LABELS_TO_PREDICT)
        label_aucs = [np.nan] * len(LABELS_TO_PREDICT)
        noisy_accs = [np.nan] * len(LABELS_TO_PREDICT)
        noisy_aucs = [np.nan] * len(LABELS_TO_PREDICT)
        clean_accs = [np.nan] * len(LABELS_TO_PREDICT)
        clean_aucs = [np.nan] * len(LABELS_TO_PREDICT)
    
        for l in range(len(LABELS_TO_PREDICT)):
            best_acc = 0.0
            best_auc = 0.0
            best_noisy_acc = 0.0
            best_noisy_auc = 0.0
            best_clean_acc = 0.0
            best_clean_auc = 0.0

            for C in [1.0, 10.0, 100.0]:
                for b in [.01, .001]:
                    svm_model = SVC(C=C, kernel='rbf', gamma=b)
                    try:
                        svm_model.fit(embed_train_X, self.classification_data_loader.train_Y[:,l])
                        
                        best_acc, best_auc = self.svm_pred_best_result(svm_model, embed_X_val,
                                                                    self.classification_data_loader.val_Y, l, 
                                                                    best_acc, best_auc)
                        best_noisy_acc, best_noisy_auc = self.svm_pred_best_result(svm_model, embed_X_noisy,
                                                                    self.classification_data_loader.noisy_val_Y, l, 
                                                                    best_noisy_acc, best_noisy_auc)
                        best_clean_acc, best_clean_auc = self.svm_pred_best_result(svm_model, embed_X_clean,
                                                                    self.classification_data_loader.clean_val_Y, l, 
                                                                    best_clean_acc, best_clean_auc)
                    except:
                        print "Error! Could not fit SVM model for some reason!"
                        
                    
            label_accs[l] = best_acc
            label_aucs[l] = best_auc
            noisy_accs[l] = best_noisy_acc
            noisy_aucs[l] = best_noisy_auc
            clean_accs[l] = best_clean_acc
            clean_aucs[l] = best_clean_auc

        return (np.atleast_2d(label_accs), np.atleast_2d(label_aucs), np.atleast_2d(noisy_accs), 
                np.atleast_2d(noisy_aucs), np.atleast_2d(clean_accs), np.atleast_2d(clean_aucs))
    
    def svm_pred_best_result(self, svm_model, X, Y, label, best_acc, best_auc):
        """Given an SVM model and some data, tests whether the SVM's predictions
        are more accurate than the existing best accuracy. Returns the highest 
        of the two."""
        preds = svm_model.predict(X)
        acc, auc, f1, precision, recall = gen_wrap.compute_all_classification_metrics(
            preds, Y[:,label])

        if acc > best_acc and auc > best_auc:
            best_acc = acc
            best_auc = auc
        
        return best_acc, best_auc

    def get_cross_validation_results(self, param_dict):
        """Goes through every cross-validation fold in the class's DataLoader, 
        assesses all necessary metrics for each fold, and saves them into the 
        param_dict.

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
        
        Returns: The param_dict augmented with keys for the names of metrics and
            values representing the score on those metrics.
        """
        losses = []
        aucs = None
        accs = None

        noisy_accs = None
        noisy_aucs = None
        clean_accs = None
        clean_aucs = None

        for f in range(self.num_cross_folds):
            self.data_loader.set_to_cross_validation_fold(f)
            self.classification_data_loader.set_to_cross_validation_fold(f)
            losses.append(self.train_and_predict(param_dict))
        
            (fold_accs, fold_aucs, f_noisy_accs, 
             f_noisy_aucs, f_clean_accs, f_clean_aucs) = self.test_embedding_classification_quality()
            accs = self.append_fold_results(accs, fold_accs)
            aucs = self.append_fold_results(aucs, fold_aucs)
            noisy_accs = self.append_fold_results(noisy_accs, f_noisy_accs)
            noisy_aucs = self.append_fold_results(noisy_aucs, f_noisy_aucs)
            clean_accs = self.append_fold_results(clean_accs, f_clean_accs)
            clean_aucs = self.append_fold_results(clean_aucs, f_clean_aucs)
            
        print "Losses for each fold:", losses
        param_dict[self.optimize_for] = np.mean(losses)

        for i, label in enumerate(LABELS_TO_PREDICT):
            param_dict['svm_val_acc_'+label] = np.nanmean(accs[:,i])
            param_dict['svm_val_auc_'+label] = np.nanmean(aucs[:,i])
            print "Average accuracy for label", label, "=", np.nanmean(accs[:,i])

            param_dict['svm_noisy_val_acc_'+label] = np.nanmean(noisy_accs[:,i])
            param_dict['svm_noisy_val_auc_'+label] = np.nanmean(noisy_aucs[:,i])
            param_dict['svm_clean_val_acc_'+label] = np.nanmean(clean_accs[:,i])
            param_dict['svm_clean_val_auc_'+label] = np.nanmean(clean_aucs[:,i])
            
        param_dict['svm_val_acc'] = np.nanmean(accs)
        param_dict['svm_val_auc'] = np.nanmean(aucs)
        param_dict['svm_noisy_val_acc'] = np.nanmean(noisy_accs)
        param_dict['svm_noisy_val_auc'] = np.nanmean(noisy_aucs)
        param_dict['svm_clean_val_acc'] = np.nanmean(clean_accs)
        param_dict['svm_clean_val_auc'] = np.nanmean(clean_aucs)
        print "Average accuracy on noisy data", np.nanmean(noisy_accs)
        print "Average accuracy on clean data", np.nanmean(clean_accs)

        return param_dict

    def append_fold_results(self, all_results, fold_results):
        """Helper function that appends an array of results for a given 
        cross-validation fold (one score for each classification label) 
        to existing results from the other folds.

        Args:
            all_results: The existing results from all previous folds. May be
                None if this is the first fold.
            fold_results: An array of results from this fold.  
        
        Returns:
            A new array containing results from all folds so far.
        """
        if all_results is None:
            all_results = fold_results
        else:
            all_results = np.concatenate([all_results, fold_results], axis=0)
        return all_results

    def test_on_test(self, param_dict):
        """Get the final loss of the model on the test data.
        
        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
        
        Returns: The final reconstruction loss.
        """
        val_loss = self.train_and_predict(param_dict)
        loss = self.model.get_performance_on_data(self.data_loader.test_X)
        print "\nFINAL TEST RESULTS:"
        print self.loss_func, loss

    def run(self):
        """Runs the wrapper by checking all combinations of parameter settings, finding
        the best one, and possibly testing on the test set.""" 
        self.sweep_all_parameters()
        self.get_final_results()

        for metric in ['svm_val_acc', 'svm_val_auc']:
            best_setting = self.find_best_setting(optimize_for=metric)

if __name__ == "__main__":
    print "MMAE MODEL SELECTION"
    print "\tThis code will sweep a set of parameters to find the ideal settings for an MMAE on a single dataset"

    datasets_path = 'Data/Cleaned/'
    if len(sys.argv) < 2:
        print "Error: usage is python autoencoder_wrapper.py <filename> <continue>"
        print "\t<filename>: e.g. all_modalities_present.csv - program will look in the following directory for this file", DEFAULT_MAIN_DIRECTORY + datasets_path
        print "\t<continue>: optional. If 'True', the wrapper will pick up from where it left off by loading a previous validation results file"
        sys.exit()
    filename = sys.argv[1] #get data file from command line argument
    print "\nLoading dataset", DEFAULT_MAIN_DIRECTORY + datasets_path + filename
    print ""

    if len(sys.argv) >= 3 and sys.argv[2] == 'True':
        cont = True
        print "Okay, will continue from a previously saved validation results file for this problem"
    else:
        cont = False
    print ""

    wrapper = MMAEWrapper(filename, dropbox_path=PATH_TO_DROPBOX, datasets_path=datasets_path,
                          cont=cont)

    print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'

    wrapper.run()

