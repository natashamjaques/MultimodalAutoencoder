import numpy as np
import pandas as pd
import os
import sys

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_MAIN_DIRECTORY = '/Your/path/here/'

DEFAULT_NUM_CROSS_FOLDS = 5
LABELS_TO_PREDICT = ['happiness', 'health', 'calmness']


import data_funcs
import helper_funcs as helper
from generic_wrapper import ClassificationWrapper
import generic_wrapper as gen_wrap
import multimodal_autoencoder as mmae

def reload_dependencies():
    reload(data_funcs)
    reload(helper)
    reload(mmae)

class MMAEClassificationWrapper(ClassificationWrapper):
    """A class that inherits from the generic wrapper, enabling the testing and evaluation
    of different hyperparameters settings for use with a Multimodal Autoencoder (MMAE) - 
    with additional classification layers. Performs a grid search over every combination of 
    settings.

    Note: this class is different from the MMAEWrapper because it trains in two steps. First,
    it trains the MMAE to reconstruct data with missing modalities. Then, in a second step,
    it trains the MMAE to classify supervised labels using a different part of the network
    connected to the encoding layers. 
    """
    def __init__(self, mmae_filename, classification_filename,
                 mmae_layer_sizes=[[1000,100],[200,100],[500,100]],
                 classification_layer_sizes=[[50,20],[25,10],[100,50],[100]], 
                 tie_weights=[True,False], mmae_dropout_probs=[1.0,0.5], mmae_weight_penalties=[.01,.001],
                 weight_initializers=['normal'], mmae_activation_funcs=['relu'], 
                 mmae_test_variational=[True,False], 
                 weight_penalties=[0.0,.001], dropout_probs=[0.5,1.0], activation_funcs=['relu'],
                 classification_learning_rate=.0001, classification_batch_size=100, classification_num_steps=15000,
                 cont=False, classifier_name='MMAE_NN_classifier', num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, 
                 dropbox_path=DEFAULT_MAIN_DIRECTORY, datasets_path='Data/Cleaned/', results_path=None, 
                 check_test=False, normalization='between_0_and_1', optimize_for='val_acc', min_or_max='max', 
                 save_results_every_nth=1, check_noisy_data=True, wanted_label=None):
        """Initializes both the MMAEClassificationWrapper and its parent Wrapper with some 
        settings. Other class settings like the loss function used to train the models are built
        into the class and have to be changed elsewhere.

        Args: almost entirely the same as the parent class and the MMAEWrapper (where the 
            mmae prefix indicates its an MMAE hyperparameter), except:
            classification_layer_sizes: A list of sizes of the classification layers 
                connected to the encoder to test. 
            weight_penalities: A list of L2 weight regularization penalties to test in the
                classification layers.
            dropout_probs: A list of dropout keep probabilities to test in the classification
                layers.
            activation_funcs: A list of strings describing different activation functions 
                to test within the classification layers. Can contain 'softsign', 'relu', 
                'tanh', 'softplus', or 'linear'. 
            classification_learning_rate: The initial learning rate to use in the second 
                step of training the classification portion of the network.
            classification_batch_size: The batch size used in the second classification
                training step.
            classficication_num_steps: The number of training steps to use when training
                the classification portion of the network.
            wanted_label: The name of the column containing the labels that we are trying 
                to classify.
        """
        # Memorize arguments
        self.classification_filename = classification_filename
        self.mmae_filename = mmae_filename
        
        # Hyperparameters to test
        self.mmae_layer_sizes = mmae_layer_sizes
        self.classification_layer_sizes = classification_layer_sizes
        self.tie_weights = tie_weights
        self.mmae_dropout_probs = mmae_dropout_probs
        self.mmae_weight_penalties = mmae_weight_penalties
        self.weight_initializers = weight_initializers
        self.mmae_activation_funcs = mmae_activation_funcs
        self.mmae_test_variational = mmae_test_variational
        self.weight_penalties = weight_penalties
        self.dropout_probs = dropout_probs
        self.activation_funcs = activation_funcs

        # fixed hyperparameters
        self.classification_learning_rate = classification_learning_rate
        self.classification_num_steps = classification_num_steps
        self.classification_batch_size = classification_batch_size
        self.mmae_loss_func = 'sigmoid_cross_entropy'
        self.mmae_learning_rate = .001
        self.mmae_num_steps = 15000
        self.mmae_batch_size = 20

        ClassificationWrapper.__init__(self, filename=classification_filename, wanted_label=wanted_label, 
            cont=cont, classifier_name=classifier_name, num_cross_folds=num_cross_folds, 
            dropbox_path=dropbox_path, datasets_path=datasets_path, results_path=results_path, 
            check_test=check_test, normalization=normalization, optimize_for=optimize_for, 
            min_or_max=min_or_max, save_results_every_nth=save_results_every_nth, 
            check_noisy_data=check_noisy_data)

        self.trim_extra_vae_params()
        self.model = None

    def load_data(self):
        """Use the DataLoader class to load unsupervised and supervised data from 
        files for the MMAE and classification portions of the network, respectively.
        """
        self.data_loader = data_funcs.DataLoader(
            self.datasets_path + self.mmae_filename, 
            normalize_and_fill=False,
            supervised=False,
            cross_validation=True,
            separate_noisy_data=self.check_noisy_data)
        self.classification_data_loader = data_funcs.DataLoader(
            self.datasets_path + self.classification_filename, 
            normalize_and_fill=False,
            cross_validation=True,
            supervised=True,
            separate_noisy_data=self.check_noisy_data,
            wanted_label=self.wanted_label)

    def define_params(self):
        """Defines the list of hyperparameters that will be tested."""
        self.params = {}
        self.params['mmae_architecture'] = self.mmae_layer_sizes
        self.params['classification_layers'] = self.classification_layer_sizes
        self.params['tie_weights'] = self.tie_weights
        self.params['mmae_dropout_prob'] = self.mmae_dropout_probs
        self.params['mmae_weight_penalty'] = self.mmae_weight_penalties
        self.params['weight_initialization'] = self.weight_initializers
        self.params['mmae_activation_function'] = self.mmae_activation_funcs
        self.params['variational'] = self.mmae_test_variational
        self.params['weight_penalty'] = self.weight_penalties
        self.params['dropout_prob'] = self.dropout_probs
        self.params['activation_func'] = self.activation_funcs

    def initialize_model(self, param_dict):
        """Initializes an internal instance of an MMAE with the hyperparameter
        settings in param_dict. Also sets the classification parameters as a 
        second step.

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
        """
        self.model = mmae.MultimodalAutoencoder(
            # constant factors that don't change
            batch_size=self.mmae_batch_size, learning_rate=self.mmae_learning_rate, 
            model_name=self.classifier_name, verbose=False, loss_func=self.mmae_loss_func,
            checkpoint_dir=self.dropbox_path + 'temp_saved_models/',

            # factors that change with param dict
            layer_sizes=param_dict['mmae_architecture'], 
            classification_layer_sizes=param_dict['classification_layers'],
            variational=param_dict['variational'], tie_weights=param_dict['tie_weights'], 
            dropout_prob=param_dict['mmae_dropout_prob'],
            weight_penalty=param_dict['mmae_weight_penalty'], 
            activation_func=param_dict['mmae_activation_function'],
            weight_initialization=param_dict['weight_initialization'],
            
            # feed in the data
            data_loader=self.data_loader, classification_data_loader=self.classification_data_loader)

        if self.wanted_label is not None:
            classification_loss = 'cross_entropy'
        else:
            classification_loss = 'sigmoid_cross_entropy'
        self.model.set_classification_params(weight_penalty=param_dict['weight_penalty'], 
                                             learning_rate=self.classification_learning_rate,
                                             dropout_prob=param_dict['dropout_prob'], 
                                             activation_func=param_dict['activation_func'], 
                                             batch_size=self.classification_batch_size,
                                             loss_func=classification_loss,
                                             suppress_warning=True)

    def trim_extra_vae_params(self):
        """Using a variational loss is already such a strong regularizer, that its 
        inadvisable to use it while also tying the weights of the encoder and decoder.
        Therefore, removes these settings from the settings the wrapper will test.
        """
        i = 0
        while i < len(self.list_of_param_settings):
            setting = self.list_of_param_settings[i]
            if setting['variational'] == True and setting['tie_weights'] == True:
                self.list_of_param_settings.remove(setting)
                continue
            i += 1
        self.num_settings = len(self.list_of_param_settings)
    
    def train_and_predict(self, param_dict, predict_on='Val'):
        """Initializes an MMAE according to the desired parameter settings, trains
        it using reconstruction loss as a first step, then trains it to perform
        classification as a second step.

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
            predict_on: The dataset used for evaluating the model. Can set to 
                'Test' to get final results.

        Returns: the reconstruction loss obtained, and the classification 
            predictions.
        """
        if predict_on == 'Test':
            unsup_predict_X = self.data_loader.test_X
            sup_predict_X = self.classification_data_loader.test_X
        else:
            unsup_predict_X = self.data_loader.val_X
            sup_predict_X = self.classification_data_loader.val_X

        self.initialize_model(param_dict)
        self.model.train(self.mmae_num_steps, record_every_nth=self.mmae_num_steps/10,
                         save_every_nth=self.mmae_num_steps*2)
        loss = self.model.get_performance_on_data_with_noise(unsup_predict_X)
        print "\tFinished training MMAE, loss was", loss
        
        self.model.train_classification(num_steps=self.classification_num_steps,
                                        record_every_nth=self.classification_num_steps/10, 
                                        save_every_nth=self.classification_num_steps*2)
        preds = self.predict_on_data(sup_predict_X)

        return loss, preds

    def predict_on_data(self, X):
        """Gets the classification predictions on some data X. 

        Args:
            X: a matrix of data

        Returns: the predicted Y labels.
        """
        return self.model.get_classification_predictions(X)

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
        num_labels = len(self.classification_data_loader.wanted_labels)

        all_acc = np.empty((self.num_cross_folds, num_labels))
        all_auc = np.empty((self.num_cross_folds, num_labels))
        all_f1 = np.empty((self.num_cross_folds, num_labels))
        all_precision = np.empty((self.num_cross_folds, num_labels))
        all_recall = np.empty((self.num_cross_folds, num_labels))
        all_loss = [np.nan] * self.num_cross_folds

        if self.check_noisy_data:
            noisy_acc = np.empty((self.num_cross_folds, num_labels))
            noisy_auc = np.empty((self.num_cross_folds, num_labels))

            clean_acc = np.empty((self.num_cross_folds, num_labels))
            clean_auc = np.empty((self.num_cross_folds, num_labels))

        for f in range(self.num_cross_folds):
            self.data_loader.set_to_cross_validation_fold(f)
            self.classification_data_loader.set_to_cross_validation_fold(f)
            
            all_loss[f], preds = self.train_and_predict(param_dict)
            true_y = self.classification_data_loader.val_Y

            if self.wanted_label is None:
                for l in range(num_labels):
                    (all_acc[f,l], all_auc[f,l], all_f1[f,l], 
                    all_precision[f,l], all_recall[f,l]) = gen_wrap.compute_all_classification_metrics(preds[:,l], true_y[:,l])
                print "\tFinished training classifier, average acc was", np.mean(all_auc[f,:])
            else:
                (all_acc[f], all_auc[f], all_f1[f], 
                 all_precision[f], all_recall[f]) = gen_wrap.compute_all_classification_metrics(preds, true_y)
                print "\tFinished training classifier, acc was", all_acc[f]

            if self.check_noisy_data:
                noisy_preds = self.predict_on_data(self.classification_data_loader.noisy_val_X)
                clean_preds = self.predict_on_data(self.classification_data_loader.clean_val_X)
                
                if self.wanted_label is None:
                    for l in range(num_labels):
                        noisy_acc[f,l], noisy_auc[f,l], _, _, _ = gen_wrap.compute_all_classification_metrics(
                            noisy_preds[:,l], self.classification_data_loader.noisy_val_Y[:,l])
                        clean_acc[f,l], clean_auc[f,l], _, _, _ = gen_wrap.compute_all_classification_metrics(
                            clean_preds[:,l], self.classification_data_loader.clean_val_Y[:,l])
                else:
                    noisy_acc[f], noisy_auc[f], _, _, _ = gen_wrap.compute_all_classification_metrics(
                            noisy_preds, self.classification_data_loader.noisy_val_Y)
                    clean_acc[f], clean_auc[f], _, _, _ = gen_wrap.compute_all_classification_metrics(
                            clean_preds, self.classification_data_loader.clean_val_Y)

        param_dict['val_loss'] = np.nanmean(all_loss)
        param_dict['val_acc'] = np.nanmean(all_acc)
        param_dict['val_auc'] = np.nanmean(all_auc)
        param_dict['val_f1'] = np.nanmean(all_f1)
        param_dict['val_precision'] = np.nanmean(all_precision)
        param_dict['val_recall'] = np.nanmean(all_recall)
        print "Finished training all folds, average acc was", np.nanmean(all_acc)
        if self.wanted_label is None:
            for i,label in enumerate(LABELS_TO_PREDICT):
                param_dict['val_acc_'+label] = np.nanmean(all_acc[:,i])
                param_dict['val_auc_'+label] = np.nanmean(all_auc[:,i])
                print "Average accuracy for label", label, "=", np.nanmean(all_acc[:,i])

        if self.check_noisy_data:
            param_dict['noisy_val_acc'] = np.nanmean(noisy_acc)
            param_dict['noisy_val_auc'] = np.nanmean(noisy_auc)
            print "Perf on noisy data:", np.nanmean(noisy_acc), "acc", np.nanmean(noisy_auc), "auc"
            param_dict['clean_val_acc'] = np.nanmean(clean_acc)
            param_dict['clean_val_auc'] = np.nanmean(clean_auc)
            print "Perf on clean data:", np.nanmean(clean_acc), "acc", np.nanmean(clean_auc), "auc"

            if self.wanted_label is None:
                for i, label in enumerate(LABELS_TO_PREDICT):
                    param_dict['svm_noisy_val_acc_'+label] = np.nanmean(noisy_acc[:,i])
                    param_dict['svm_noisy_val_auc_'+label] = np.nanmean(noisy_auc[:,i])
                    param_dict['svm_clean_val_acc_'+label] = np.nanmean(clean_acc[:,i])
                    param_dict['svm_clean_val_auc_'+label] = np.nanmean(clean_auc[:,i])

        return param_dict
    
    def get_final_results(self):
        """Find the best setting and use it to test on the test data and
        print the results."""
        best_setting = self.find_best_setting()
        print "\nThe best", self.optimize_for, "was", best_setting[self.optimize_for]
        print "It was found with the following settings:"
        print best_setting

        if not self.check_test:
            print "check_test is set to false, Will not evaluate performance on held-out test set."
            return
        print "\nAbout to evaluate results on held-out test set!!"
        print "Will use the settings that produced the best", optimize_for
        
        best_setting = self.convert_param_dict_for_use(best_setting)
        
        print "\nFINAL TEST RESULTS:"
        loss, preds = self.test_on_test(best_setting)
        true_y = self.classification_data_loader.test_Y
        accs = []
        aucs = []
        for i,label in enumerate(LABELS_TO_PREDICT):
            acc, auc, f1, precision, recall = gen_wrap.compute_all_classification_metrics(preds[:,i], true_y[:,i])
            print label, 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall
            accs.append(acc)
            aucs.append(auc)
        
        print "Overall:", 'Acc:', np.mean(accs), 'AUC:', np.mean(aucs)

    def test_on_test(self, param_dict):
        """Trains the model and tests it on the tests data. 

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.

        Returns: the reconstruction loss obtained, and the classification 
            predictions.
        """
        return train_and_predict(self, param_dict, predict_on='Test')

if __name__ == "__main__":
    print "MMAE NN MODEL SELECTION"
    print "\tThis code will sweep a set of parameters to find the ideal settings for an MMAE+NN on a single dataset"

    datasets_path = 'Data/Cleaned/'
    if len(sys.argv) < 3:
        print "Error: usage is python svm.py <filename> <label> <continue>"
        print "\t<MMAE filename>: e.g. all_modalities_present.csv - program will look in the following directory for this file", DEFAULT_MAIN_DIRECTORY + datasets_path
        print "\t<classification filename>: e.g. modalities_missing.csv - program will look in the same directory for this file and use it for classification data"
        print "\t<extra arguments>: optional. If 'True', the wrapper will continue from where it left off by loading a previous validation results file",
        print "if the name of a label, the wrapper will only train to classify on that label. Can have multiple extra arguments, e.g. happiness True"
        sys.exit()
    filename = sys.argv[1] #get data file from command line argument
    classification_filename = sys.argv[2]
    print "\nLoading dataset", DEFAULT_MAIN_DIRECTORY + datasets_path + filename
    print ""

    if len(sys.argv) >=4:
        extra_args = sys.argv[3]
        label = None
        if 'true' in extra_args.lower() or 'cont' in extra_args.lower():
            cont = True
        else:
            label = extra_args

    if len(sys.argv) >= 5 and sys.argv[4] == 'True':
        cont = True
        print "Okay, will continue from a previously saved validation results file for this problem"
    else:
        cont = False
    print ""

    wrapper = MMAEClassificationWrapper(filename, classification_filename, dropbox_path=PATH_TO_DROPBOX, 
                                        datasets_path=datasets_path, cont=cont, wanted_label=label)

    print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'

    wrapper.run()