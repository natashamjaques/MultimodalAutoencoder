import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import sys
import math

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_MAIN_DIRECTORY = '/Your/path/here/'

DEFAULT_NUM_CROSS_FOLDS = 5
Z_SCORE_FILL_WITH_0 = True
LABELS_TO_PREDICT = ['happiness', 'health', 'calmness']

import data_funcs
import helper_funcs as helper
from generic_wrapper import ClassificationWrapper
import generic_wrapper as gen_wrap

def reload_dependencies():
    reload(data_funcs)
    reload(helper)

class NeuralNetwork:
    """A basic neural net class that performs simple classification."""
    def __init__(self, filename=None, layer_sizes=[128,64], batch_size=20, 
                 learning_rate=.001, dropout_prob=1.0, weight_penalty=0.0, 
                 model_name='NN', clip_gradients=True, data_loader=None,
                 checkpoint_dir=DEFAULT_MAIN_DIRECTORY + 'temp_saved_models/', 
                 verbose=True):
        '''Initialize the class by loading the required datasets 
        and building the graph.

        Args:
            filename: a file containing the data.
            layer_sizes: a list of sizes of the neural network layers.
            batch_size: number of training examples in each training batch. 
            learning_rate: the initial learning rate used in stochastic 
                gradient descent.
            dropout_prob: the probability that a node in the network will not
                be dropped out during training. Set to < 1.0 to apply dropout, 
                1.0 to remove dropout.
            weight_penalty: the coefficient of the L2 weight regularization
                applied to the loss function. Set to > 0.0 to apply weight 
                regularization, 0.0 to remove.
            model_name: name of the model being trained. Used in saving
                model checkpoints.
            clip_gradients: a bool indicating whether or not to clip gradients. 
                This is effective in preventing very large gradients from skewing 
                training, and preventing your loss from going to inf or nan. 
            data_loader: A DataLoader class object which already has pre-loaded
                data.
            checkpoint_dir: the directly where the model will save checkpoints,
                saved files containing trained network weights.
            verbose: if True, will print many informative output statements.
            '''
        # Hyperparameters that should be tuned
        self.layer_sizes = layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob 
        self.weight_penalty = weight_penalty 

        # Hyperparameters that could be tuned 
        # (but are probably the best to use)
        self.clip_gradients = clip_gradients
        self.activation_func = 'relu'
        self.optimizer = tf.train.AdamOptimizer

        # Logistics
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.model_name = model_name
        self.output_every_nth = 100
        self.verbose = verbose

        # Extract the data from the filename
        if data_loader is None:
            self.data_loader = data_funcs.DataLoader(filename)
        else:
            self.data_loader = data_loader
        self.input_size = self.data_loader.get_feature_size()
        self.output_size = self.data_loader.num_labels
        if self.verbose:
            print "Input dimensions (number of features):", self.input_size
            print "Number of classes/outputs:", self.output_size
        
        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)

        # Use for plotting evaluation.
        self.train_acc = []
        self.val_acc = []

    def initialize_network_weights(self):
        """Constructs Tensorflow variables for the weights and biases
        in each layer of the graph. These variables will be updated
        as the network learns.

        The number of layers and the sizes of each layer are defined
        in the class's layer_sizes field.
        """
        sizes = []
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes)+1):
            if i==0:
                input_len = self.input_size # X second dimension
            else:
                input_len = self.layer_sizes[i-1]
            
            if i==len(self.layer_sizes):
                output_len = self.output_size
            else:
                output_len = self.layer_sizes[i]
                
            layer_weights = weight_variable([input_len, output_len],name='weights' + str(i))
            layer_biases = bias_variable([output_len], name='biases' + str(i))
            
            self.weights.append(layer_weights)
            self.biases.append(layer_biases)
            sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))
        
        if self.verbose:
            print("Okay, making a neural net with the following structure:")
            print(sizes)

    def build_graph(self):
        """Constructs the tensorflow computation graph containing all variables
        that will be trained."""

        with self.graph.as_default():
            # Placeholders can be used to feed in different data during training time.
            self.tf_X = tf.placeholder(tf.float32, name="X") # features
            self.tf_Y = tf.placeholder(tf.float32, name="Y") # labels
            self.tf_dropout_prob = tf.placeholder(tf.float32) # Implements dropout

            # Place the network weights/parameters that will be learned into the 
            # computation graph.
            self.initialize_network_weights()

            # Defines the actual network computations using the weights. 
            def run_network(input_X):
                hidden = input_X
                for i in range(len(self.weights)):
                    with tf.name_scope('layer' + str(i)) as scope:
                        # tf.matmul is a simple fully connected layer. 
                        hidden = tf.matmul(hidden, self.weights[i]) + self.biases[i]
                        
                        if i < len(self.weights)-1:
                            # Apply activation function
                            if self.activation_func == 'relu':
                                hidden = tf.nn.relu(hidden) 
                            # Could add more activation functions like sigmoid here
                            # If no activation is specified, none will be applied

                            # Apply dropout
                            hidden = tf.nn.dropout(hidden, self.tf_dropout_prob) 
                return hidden
            self.run_network = run_network

            # Compute the loss function
            self.logits = run_network(self.tf_X)

            # Apply a softmax function to get probabilities, train this dist against targets with
            # cross entropy loss.
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, 
                                                                               labels=self.tf_Y))

                # Add weight decay regularization term to loss
            self.loss += self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.weights])

            # Code for making predictions and evaluating them.
            self.class_probabilities = tf.nn.sigmoid(self.logits)
            self.predictions = tf.cast(tf.round(self.class_probabilities), dtype=tf.int32)
            self.correct_prediction = tf.equal(self.predictions, tf.cast(self.tf_Y, dtype=tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            # Set up backpropagation computation!
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.params = tf.trainable_variables()
            self.gradients = tf.gradients(self.loss, self.params)
            if self.clip_gradients:
                self.gradients, _ = tf.clip_by_global_norm(self.gradients, 5)
            self.tf_optimizer = self.optimizer(self.learning_rate)
            self.opt_step = self.tf_optimizer.apply_gradients(zip(self.gradients, self.params),
                                                              self.global_step)

            # Necessary for tensorflow to build graph
            self.init = tf.global_variables_initializer()

    def train(self, num_steps=30000, output_every_nth=None):
        """Trains using stochastic gradient descent (SGD). 
        
        Runs batches of training data through the model for a given
        number of steps.
        """
        if output_every_nth is not None:
            self.output_every_nth = output_every_nth

        with self.graph.as_default():
            # Used to save model checkpoints.
            self.saver = tf.train.Saver()

            for step in range(num_steps):
                # Grab a batch of data to feed into the placeholders in the graph.
                X, Y = self.data_loader.get_supervised_train_batch(self.batch_size)
                feed_dict = {self.tf_X: X,
                             self.tf_Y: Y,
                             self.tf_dropout_prob: self.dropout_prob}
                
                # Update parameters in the direction of the gradient computed by
                # the optimizer.
                _ = self.session.run([self.opt_step], feed_dict)

                # Output/save the training and validation performance every few steps.
                if step % self.output_every_nth == 0:
                    # Grab a batch of validation data too.
                    val_X, val_Y = self.data_loader.get_val_data()
                    val_feed_dict = {self.tf_X: val_X,
                                     self.tf_Y: val_Y,
                                     self.tf_dropout_prob: 1.0} # no dropout during evaluation

                    train_score, loss = self.session.run([self.accuracy, self.loss], feed_dict)
                    val_score, loss = self.session.run([self.accuracy, self.loss], val_feed_dict)
                    
                    if self.verbose:
                        print "Training iteration", step
                        print "\t Training acc", train_score
                        print "\t Validation acc", val_score
                        print "\t Loss", loss
                    self.train_acc.append(train_score)
                    self.val_acc.append(val_score)

                    # Save a checkpoint of the model
                    self.saver.save(self.session, self.checkpoint_dir + self.model_name + '.ckpt', global_step=step)
    
    def predict(self, X, get_probabilities=False):
        """Gets the network's predictions for some new data X
        
        Args: 
            X: a matrix of data in the same format as the training
                data. 
            get_probabilities: a boolean that if true, will cause 
                the function to return the model's computed softmax
                probabilities in addition to its predictions. Only 
                works for classification.
        Returns:
            integer class predictions if the model is doing 
            classification, otherwise float predictions if the 
            model is doing regression.
        """
        feed_dict = {self.tf_X: X,
                     self.tf_dropout_prob: 1.0} # no dropout during evaluation
        
        probs, preds = self.session.run([self.class_probabilities, self.predictions], 
                                            feed_dict)
        if get_probabilities:
            return preds, probs
        else:
            return preds
        
    def plot_training_progress(self):
        """Plots the training and validation performance as evaluated 
        throughout training."""
        x = [self.output_every_nth * i for i in np.arange(len(self.train_acc))]
        plt.figure()
        plt.plot(x,self.train_acc)
        plt.plot(x,self.val_acc)
        plt.legend(['Train', 'Validation'], loc='best')
        plt.xlabel('Training epoch')
        plt.ylabel('Accuracy')
        plt.show()

    def test_on_validation(self):
        """Returns performance on the model's validation set."""
        score = self.get_performance_on_data(self.data_loader.val_X,
                                             self.data_loader.val_Y)
        print "Final accuracy on validation data is:", score
        return score
        
    def test_on_test(self):
        """Returns performance on the model's test set."""
        score = self.get_performance_on_data(self.data_loader.test_X,
                                             self.data_loader.test_Y)
        print "Final accuray on test data is:", score
        return score

    def get_performance_on_data(self, X, Y):
        """Returns the model's performance on input data X and targets Y.
        
        Args:
            X: A matrix of data
            Y: A matrix of labels
        
        Returns: classification accuracy
        """
        feed_dict = {self.tf_X: X,
                     self.tf_Y: Y,
                     self.tf_dropout_prob: 1.0} # no dropout during evaluation
        
        return self.session.run(self.accuracy, feed_dict)

    def save_model(self, file_name=None, directory=None):
        """Saves a checkpoint of the model and a .npz file with stored rewards.

        Args:
            file_name: String name to use for the checkpoint and rewards files.
                Defaults to self.model_name if None is provided.
            directory: The directory where the model should be saved.
        """
        if self.verbose: print "Saving model..."
        if file_name is None:
            file_name = self.model_name

        if directory is None:
            directory = self.checkpoint_dir
        else:
            save_dir = directory + file_name
            os.mkdir(save_dir)
            directory = save_dir + '/'

        save_loc = os.path.join(directory, file_name + '.ckpt')
        training_epochs = len(self.train_acc) * self.output_every_nth
        self.saver.save(self.session, save_loc, global_step=training_epochs)
        
        npz_name = os.path.join(directory, 
                                file_name + '-' + str(training_epochs))
        np.savez(npz_name,
                train_acc=self.train_acc,
                val_acc=self.val_acc)
    
    def load_saved_model(self, directory=None, checkpoint_name=None,
                         npz_file_name=None):
        """Restores this model from a saved checkpoint.

        Args:
            directory: Path to directory where checkpoint is located. If 
                None, defaults to self.checkpoint_dir.
            checkpoint_name: The name of the checkpoint within the 
                directory.
            npz_file_name: The name of the .npz file where the stored
                rewards are saved. If None, will not attempt to load stored
                rewards.
        """
        print "-----Loading saved model-----"
        if directory is None:
            directory = self.checkpoint_dir

        if checkpoint_name is not None:
            checkpoint_file = os.path.join(directory, checkpoint_name)
        else:
            checkpoint_file = tf.train.latest_checkpoint(directory)
            print "Looking for checkpoin in directory", directory

        if checkpoint_file is None:
            print "Error! Cannot locate checkpoint in the directory"
            return
        else:
            print "Found checkpoint file:", checkpoint_file

        if npz_file_name is not None:
            npz_file_name = os.path.join(directory, npz_file_name)
            print "Attempting to load saved reward values from file", npz_file_name
            npz_file = np.load(npz_file_name)

            self.train_acc = npz_file['train_acc']
            self.val_acc = npz_file['val_acc']
            
        self.graph = tf.Graph()
        self.build_graph()
        self.initialize_session()
        self.saver.restore(self.session, checkpoint_file)

def weight_variable(shape,name):
    """Initializes a tensorflow weight variable with random
    values centered around 0.

    Args:
        shape: A list of integer sizes describing the shape of the variable.
        name: A string name of the variable.

    Returns: the tensorflow variable. 
    """
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), dtype=tf.float32)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    """Initializes a tensorflow bias variable to a small constant value.
    
    Args:
        shape: A list of integer sizes describing the shape of the variable.
        name: A string name of the variable.

    Returns: the tensorflow variable. """
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class NNWrapper(ClassificationWrapper):
    """Wrapper class for performing a grid search over hyperparameter settings for a neural 
    net classifier. Inherits from the generic ClassificationWrapper"""
    def __init__(self, filename, layer_sizes=[[300,200,100],[200,100],[128,64],[200,100,50]],  
                 dropout_probs=[0.5,1.0], weight_penalties=[0.0,.01,.001,.0001], 
                 learning_rates=[.001], batch_sizes=[100], num_steps=5000, output_every_nth=5001,
                 cont=False, classifier_name='NN', num_cross_folds=DEFAULT_NUM_CROSS_FOLDS, 
                 dropbox_path=DEFAULT_MAIN_DIRECTORY, datasets_path='Data/Cleaned/', results_path=None, 
                 check_test=True, normalize_and_fill=False, normalization='between_0_and_1', 
                 optimize_for='val_acc', min_or_max='max', save_results_every_nth=1, 
                 check_noisy_data=True, cross_validation=True):
        """Initializes both the NNWrapper and its parent.

        Args: almost entirely the same as the parent class, except:
            layer_sizes: A list of sizes of the layers used in the network. 
            weight_penalities: A list of L2 weight regularization penalties to test.
            learning_rates: A list of initial learning rates to test.
            batch_sizes: A list of SGD batch sizes to test.
            num_steps: The number of training steps to use when training a model.
        """
        # Hyperparameters to test
        self.layer_sizes = layer_sizes
        self.dropout_probs = dropout_probs
        self.weight_penalties = weight_penalties
        self.batch_sizes = batch_sizes
        self.learning_rates = learning_rates

        # Fixed hyperparameters
        self.num_steps = num_steps
        self.output_every_nth = output_every_nth

        ClassificationWrapper.__init__(self, filename=filename, wanted_label=None, cont=cont, 
            classifier_name=classifier_name, num_cross_folds=num_cross_folds, dropbox_path=dropbox_path, 
            datasets_path=datasets_path, results_path=results_path, check_test=check_test, 
            normalize_and_fill=normalize_and_fill, normalization=normalization, 
            optimize_for=optimize_for, min_or_max=min_or_max, save_results_every_nth=save_results_every_nth, 
            check_noisy_data=check_noisy_data, cross_validation=cross_validation)

        self.model = None

    def define_params(self):
        """Defines the list of hyperparameters that will be tested."""
        self.params = {}
        self.params['architecture'] = self.layer_sizes
        self.params['dropout_prob'] = self.dropout_probs
        self.params['weight_penalty'] = self.weight_penalties
        self.params['learning_rate'] = self.learning_rates
        self.params['batch_size'] = self.batch_sizes

    def predict_on_data(self, X):
        """Gets the classification predictions on some data X. 

        Args:
            X: a matrix of data

        Returns: the predicted Y labels.
        """
        return self.model.predict(X)
    
    def train_and_predict(self, param_dict, predict_on='Val'):
        """Initializes a NN classifier according to the desired parameter settings, 
        trains it, and returns the predictions on the appropriate evaluation dataset.

        Args:
            param_dict: A dictionary with keys representing parameter names and 
                values representing settings for those parameters.
            predict_on: The dataset used for evaluating the model. Can set to 
                'Test' to get final results.
        
        Returns: The predicted Y labels.
        """
        if predict_on == 'Test':
            predict_X = self.data_loader.test_X
        else:
            predict_X = self.data_loader.val_X
        
        self.model = NeuralNetwork(layer_sizes=param_dict['architecture'], 
                                   batch_size=param_dict['batch_size'], 
                                   learning_rate=param_dict['learning_rate'], 
                                   dropout_prob=param_dict['dropout_prob'], 
                                   weight_penalty=param_dict['weight_penalty'],
                                   data_loader=self.data_loader, 
                                   verbose=False)
        self.model.train(num_steps=self.num_steps, output_every_nth=self.output_every_nth)
        if predict_on=="df":
            return self.get_classification_predictions_from_df()
        else:
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
        num_labels = len(self.data_loader.wanted_labels)

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
            
            preds = self.train_and_predict(param_dict)
            true_y = self.data_loader.val_Y

            for l in range(num_labels):
                (all_acc[f,l], all_auc[f,l], all_f1[f,l], 
                 all_precision[f,l], all_recall[f,l]) = gen_wrap.compute_all_classification_metrics(preds[:,l], true_y[:,l])

            if self.check_noisy_data:
                noisy_preds = self.predict_on_data(self.data_loader.noisy_val_X)
                clean_preds = self.predict_on_data(self.data_loader.clean_val_X)
                
                for l in range(num_labels):
                    noisy_acc[f,l], noisy_auc[f,l], _, _, _ = gen_wrap.compute_all_classification_metrics(
                        noisy_preds[:,l], self.data_loader.noisy_val_Y[:,l])
                    clean_acc[f,l], clean_auc[f,l], _, _, _ = gen_wrap.compute_all_classification_metrics(
                        clean_preds[:,l], self.data_loader.clean_val_Y[:,l])

        param_dict['val_acc'] = np.nanmean(all_acc)
        param_dict['val_auc'] = np.nanmean(all_auc)
        param_dict['val_f1'] = np.nanmean(all_f1)
        param_dict['val_precision'] = np.nanmean(all_precision)
        param_dict['val_recall'] = np.nanmean(all_recall)
        print "Finished training all folds, average acc was", np.nanmean(all_acc)
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

            for i, label in enumerate(LABELS_TO_PREDICT):
                param_dict['noisy_val_acc_'+label] = np.nanmean(noisy_acc[:,i])
                param_dict['noisy_val_auc_'+label] = np.nanmean(noisy_auc[:,i])
                param_dict['clean_val_acc_'+label] = np.nanmean(clean_acc[:,i])
                param_dict['clean_val_auc_'+label] = np.nanmean(clean_auc[:,i])

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
        print "Will use the settings that produced the best", self.optimize_for

        print "batch size is an", type(best_setting['batch_size'])
        
        best_setting = self.convert_param_dict_for_use(dict(best_setting))

        print "batch size is an", type(best_setting['batch_size'])
        
        print "\nFINAL TEST RESULTS:"
        preds = self.test_on_test(best_setting)
        true_y = self.data_loader.test_Y
        accs = []
        aucs = []
        for i,label in enumerate(LABELS_TO_PREDICT):
            print "\n", label
            acc, auc, f1, precision, recall = gen_wrap.compute_all_classification_metrics(preds[:,i], true_y[:,i])
            print label, 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall
            accs.append(acc)
            aucs.append(auc)

            print "\nFINAL TEST RESULTS ON ALL", label, "DATA:"
            print 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall

            if self.check_noisy_data:
                noisy_preds = self.predict_on_data(self.data_loader.noisy_test_X)
                acc, auc, f1, precision, recall = gen_wrap.compute_all_classification_metrics(noisy_preds[:,i], self.data_loader.noisy_test_Y[:,i])
                print "\nFINAL TEST RESULTS ON NOISY", label, "DATA:"
                print 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall

                clean_preds = self.predict_on_data(self.data_loader.clean_test_X)
                acc, auc, f1, precision, recall = gen_wrap.compute_all_classification_metrics(clean_preds[:,i], self.data_loader.clean_test_Y[:,i])
                print "\nFINAL TEST RESULTS ON CLEAN", label, "DATA:"
                print 'Acc:', acc, 'AUC:', auc, 'F1:', f1, 'Precision:', precision, 'Recall:', recall
            
        print "Overall:", 'Acc:', np.mean(accs), 'AUC:', np.mean(aucs)

if __name__ == "__main__":
    print "NN MODEL SELECTION"
    print "\tThis code will sweep a set of parameters to find the ideal settings for a NN on a single dataset"

    if Z_SCORE_FILL_WITH_0:
        normalize_and_fill = True
        datasets_path = 'Data/'
        normalization = 'z_score'
    else:
        normalize_and_fill = False
        datasets_path = 'Data/Cleaned/'
        normalization = 'between_0_and_1'

    if len(sys.argv) < 2:
        print "Error: usage is python neural_net.py <filename> <label> <continue>"
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

    wrapper = NNWrapper(filename, dropbox_path=PATH_TO_DROPBOX, datasets_path=datasets_path,
                        cont=cont, normalize_and_fill=normalize_and_fill, 
                        normalization=normalization)

    print "\nThe validation results dataframe will be saved in:", wrapper.results_path + wrapper.save_prefix + '.csv'

    wrapper.run()