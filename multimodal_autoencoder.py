import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
import copy
import os
import time
import math

CODE_PATH = os.path.dirname(os.getcwd())
sys.path.append(CODE_PATH)

DEFAULT_MAIN_DIRECTORY = '/Your/path/here/'

import data_funcs

def reload_files():
    reload(data_funcs)

def weight_variable(shape, name, var_type='normal', const=1):
    """Initializes a tensorflow weight variable.

    Args:
        shape: An array representing shape of the weight variable
        name: A string name given to the variable.
        var_type: can be either 'normal', for weights following a Gaussian
            distribution around 0, or 'xavier', for the Xavier method
        const: Numeric value that controls the range of the weights within
            the Xavier method.
    Returns: Tensor variable for the weights
    """
    if var_type == 'xavier':
        """ Xavier initialization of network weights.
        Taken from: https://gist.github.com/blackecho/3a6e4d512d3aa8aa6cf9
        https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
        """
        assert len(shape) == 2
        low = -const * np.sqrt(6.0 / (shape[0] + shape[1]))
        high = const * np.sqrt(6.0 / (shape[0] + shape[1]))
        initial = tf.random_uniform((shape[0], shape[1]), minval=low, maxval=high)
    else:
        initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(float(shape[0])), dtype=tf.float32)
    
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):
    """Initializes a tensorflow bias variable to a small constant value.
    
    Args:
        shape: An array representing shape of the weight variable
        name: A string name given to the variable.
    Returns: a Tensor variabel for the biases"""
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

class MultimodalAutoencoder:
    def __init__(self, filename=None, layer_sizes=[128,64,32], variational=True,
                 tie_weights=True, batch_size=10, learning_rate=.0001, 
                 dropout_prob=1.0, weight_penalty=0.0, activation_func='softsign', 
                 loss_func='sigmoid_cross_entropy', decay=True, decay_steps=1000, 
                 decay_rate=0.95, clip_gradients=True, classification_layer_sizes=None,
                 classification_filename=None, weight_initialization='xavier', 
                 normalization='between_0_and_1', intelligent_noise=True,
                 num_modalities_to_drop=1,
                 subdivide_physiology=True, fill_missing_with=0.0, mask_with=-1.0,
                 checkpoint_dir=DEFAULT_MAIN_DIRECTORY + 'temp_saved_models/', 
                 model_name='multimodal_autoencoder', extra_data_filename=None, 
                 data_loader=None, classification_data_loader=None, verbose=True):
        '''Initialize the class by loading the required datasets and building 
        the graph.

        Args:
            filename: A string file path containing data to load.
            layer_sizes: A list of sizes of the neural network layers in the
                encoding portion of the network. Will be mirrored for decoding
                portion.
            variational: A boolean that if True will build a Variational 
                Autoencoder model. If False, model will simply be a denoising
                autoencoder. 
            tie_weights: A boolean. If True, the net will use the same weights
                for the encoder and decoder.
            batch_size: number of training examples in each training batch. 
            learning_rate: The initial learning rate used in stochastic 
                gradient descent.
            dropout_prob: The probability that a node in the network will not
                be dropped out during training. Set to < 1.0 to apply dropout, 
                1.0 to remove dropout.
            weight_penalty: The coefficient of the L2 weight regularization
                applied to the loss function. Set to > 0.0 to apply weight 
                regularization, 0.0 to remove.
            activation_func: String representing the activation function used
                on neurons. Could be 'relu', 'tanh', 'softsign', etc.
            loss_func: Distance function that measures reconstruction error. 
            decay: A bool for whether or not to apply learning rate decay. 
            decay_steps: Number of training steps after which to decay the 
                learning rate.
            decay_rate: Rate at which the learning rate decays. 
            clip_gradients: A bool indicating whether or not to clip gradients. 
                This is effective in preventing very large gradients from skewing 
                training, and preventing your loss from going to inf or nan. 
            classification_layer_sizes: A list of sizes of neural network layers 
                that will be attached to the embedding layer and designed to 
                perform classification. If None, the network will not perform
                classification.
            classification_filename: A file where classification data is located.
            weight_initialization: If 'normal' will initialize weights using the 
                typical truncated normal distribution. If 'xavier' will use the 
                xavier method. 
            normalization: Method for normalizing the features. Can be 'z_score'
                or 'between_0_and_1'.
            subdivide_physiology: A boolean. If True, will break the physiology 
                modality into smaller pieces, each of which can be missing 
                independently.
            intelligent_noise: If True, the denoising autoencoder will drop out
                modalities using a distribution designed to match that of the 
                training data. Otherwise it will drop modalities uniformly at
                random. 
            num_modalities_to_drop: If not using intelligent noise, how many
                modalities to drop
            fill_missing_with: Value to use for filling the missing entries 
                throughout the data array.
            mask_with: Value to use for blanking out whole modalities. 
            checkpoint_dir: The directly where the model will save checkpoints,
                saved files containing trained network weights.
            model_name: Name of the model being trained. Used in saving
                model checkpoints.
            extra_data_filename: A string where additional data that has been 
                genuinely corrupted with noise can be found. If provided, can
                be used for extra testing.
            data_loader: A DataLoader class object which already has pre-loaded
                data.
            classification_data_loader: A DataLoader class object which already
                has pre-loaded classification data. 
            verbose: Set to True to see output statements about model construction
                and training.
            '''
        # Hyperparameters
        self.layer_sizes = layer_sizes
        self.embedding_size = layer_sizes[-1]
        self.tie_weights = tie_weights
        self.variational = variational
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_prob = dropout_prob 
        self.weight_penalty = weight_penalty 
        self.weight_initialization = weight_initialization
        self.classification_layer_sizes = classification_layer_sizes
        self.classification_filename = classification_filename
        self.normalization = normalization
        self.fill_missing_with = fill_missing_with
        self.mask_with = mask_with
        self.clip_gradients = clip_gradients
        self.activation_func = activation_func
        self.loss_func = loss_func
        self.decay = decay
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.optimizer = tf.train.AdamOptimizer

        # Logistics
        self.checkpoint_dir = checkpoint_dir
        self.filename = filename
        self.model_name = model_name
        self.record_every_nth = 50
        self.save_every_nth = 100000
        self.subdivide_physiology = subdivide_physiology
        self.intelligent_noise = intelligent_noise
        self.num_modalities_to_drop = num_modalities_to_drop
        self.extra_data_filename = extra_data_filename
        self.verbose = verbose

        # Override settings necessary for VAE
        if self.variational:
            if self.verbose: print "Building VAE. Will use 0-1 normalization, cross entropy loss, and will not tie weights.\n"
            self.tie_weights = False
            self.normalization = 'between_0_and_1'
            self.loss_func = 'sigmoid_cross_entropy'

        if self.normalization == 'z_score' and (loss_func == 'cross_entropy' 
                                           or loss_func == 'sigmoid_cross_entropy'):
            print "ERROR! Cannot use cross entropy loss with z-score data. Changing normalization method to 0-1"
            self.normalization = 'between_0_and_1'

        # Extract the data from the filename
        if data_loader is not None:
            self.data_loader = data_loader
        elif filename is not None:
            self.data_loader = data_funcs.DataLoader(filename, supervised=False,
                    subdivide_physiology_features=subdivide_physiology, 
                    normalize_and_fill=False,
                    normalization=self.normalization,
                    fill_missing_with=self.fill_missing_with)
        else: 
            print "ERROR! Must set either filename or data_loader to a value so that MMAE has access to data."
            return 
        self.extra_noisy_data_loader = None

        if self.intelligent_noise:
            print "Using intelligent noise"
            self.noise_type_percentages = [ 0.64018104,  0.03168217,  0.25119437,  0.07694242]
            self.noise_types = [[],
                                ['call','sms','screen'],
                                ['location'],
                                ['location','call','sms','screen']]

        if self.classification_layer_sizes is not None:
            if self.verbose: print "Okay, preparing model to perform classification"
            self.train_acc = []
            self.val_acc = []
            self.classification_train_loss = []
            self.classification_val_loss = []

            self.classification_learning_rate = .0001
            self.classification_batch_size = 100
            self.classification_dropout_prob = self.dropout_prob
            self.classification_activation_func = self.activation_func
            self.classification_weight_penalty = 0.0
            self.classification_loss_func = 'sigmoid_cross_entropy'

            if classification_data_loader is None:
                self.classification_data_loader = data_funcs.DataLoader(self.classification_filename, 
                    supervised=True,
                    subdivide_physiology_features=subdivide_physiology, 
                    normalize_and_fill=False,
                    normalization=self.normalization,
                    fill_missing_with=self.fill_missing_with)
            else:
                self.classification_data_loader = classification_data_loader

        # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.initialize_session()

        # Use for plotting evaluation.
        self.train_loss = []
        self.val_loss = []

    def rebuild_reinitialize(self):
        """Resets the tensorflow graph to start training from scratch."""
         # Set up tensorflow computation graph.
        self.graph = tf.Graph()
        self.build_graph()

        # Set up and initialize tensorflow session.
        self.initialize_session()

        # Use for plotting evaluation.
        self.train_loss = []
        self.val_loss = []

    def initialize_network_weights(self):
        """Constructs Tensorflow variables for the weights and biases
        in each layer of the graph. These variables will be updated
        as the network learns.

        The number of layers and the sizes of each layer are defined
        in the class's layer_sizes field.
        """
        # Construct encoder and decoder layers.
        enc_sizes = []
        dec_sizes = []
        self.encode_weights = []
        self.decode_weights = []
        self.encode_biases = []
        self.decode_biases = []
        for i in range(len(self.layer_sizes)):
            if i==0:
                input_len = self.data_loader.num_feats # X second dimension
            else:
                input_len = self.layer_sizes[i-1]
            output_len = self.layer_sizes[i]
                
            layer_weights = weight_variable([input_len, output_len],
                                            name='weights' + str(i), 
                                            var_type=self.weight_initialization)
            self.encode_weights.append(layer_weights)
            
            if self.tie_weights:
                self.decode_weights.append(tf.transpose(layer_weights))
            else:
                decode_weights_l = weight_variable([output_len, input_len],
                                                    name='decode_weights' + str(i), 
                                                    var_type=self.weight_initialization)
                self.decode_weights.append(decode_weights_l)

            layer_biases_enc = bias_variable([output_len], name='encode_biases' + str(i))
            layer_biases_dec = bias_variable([input_len], name='decode_biases' + str(i))
            self.encode_biases.append(layer_biases_enc)
            self.decode_biases.append(layer_biases_dec)
            enc_sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))
            dec_sizes.append((str(output_len) + "x" + str(input_len), str(input_len)))
        
        if self.variational:
            self.variance_weights = weight_variable([self.layer_sizes[-2], self.embedding_size],
                                                     name='variance_weights', 
                                                     var_type=self.weight_initialization)
            self.variance_bias = bias_variable([self.embedding_size], name='variance_bias')

        self.decode_weights.reverse()
        self.decode_biases.reverse()
        
        if self.verbose:
            print("Okay, making a neural net with the following structure:")
            dec_sizes.reverse()
            print(enc_sizes + dec_sizes)

        # Construct classification layers of the network if necessary.
        if self.classification_layer_sizes is not None:
            self.classification_weights = []
            self.classification_biases = []
            classif_sizes = []

            for i in range(len(self.classification_layer_sizes)+1):
                if i==0:
                    input_len = self.embedding_size
                else:
                    input_len = self.classification_layer_sizes[i-1]
                if i==len(self.classification_layer_sizes):
                    if self.classification_data_loader.num_labels is not None:
                        output_len = self.classification_data_loader.num_labels
                    else:
                        output_len = 2
                else:
                    output_len = self.classification_layer_sizes[i]

                layer_weights = weight_variable([input_len, output_len],
                                                name='classification_weights' + str(i), 
                                                var_type=self.weight_initialization)
                layer_biases = bias_variable([output_len], name='classification_biases' + str(i))
                self.classification_weights.append(layer_weights)
                self.classification_biases.append(layer_biases)
                
                classif_sizes.append((str(input_len) + "x" + str(output_len), str(output_len)))
            
            if self.verbose:
                print("Okay, adding additional classification layers with the following structure:")
                print(classif_sizes)

    def build_graph(self):
        """Constructs the tensorflow computation graph containing all variables
        that will be trained."""
        if self.verbose: print '\nBuilding computation graph...'

        with self.graph.as_default():
            # Data placeholder
            self.noisy_X = tf.placeholder(tf.float32, name="noisy_X")
            self.true_X = tf.placeholder(tf.float32, name="true_X")
            
            # Enhancements
            self.tf_dropout_prob = tf.placeholder(tf.float32) 
            self.global_step = tf.Variable(0)  
            if self.decay:
                self.tf_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 
                                                                 self.decay_steps, self.decay_rate)
            else:
                self.tf_learning_rate = self.learning_rate

            # Network weights/parameters that will be learned 
            self.initialize_network_weights()

            # Pass data through the network to get an embedding
            self.embedding = self.encode(self.noisy_X)
        
            # If this is a Variational Autoencoder, need to sample embedding
            # from Normal distribution with mean of the previous embedding and
            # learned variance
            if self.variational:
                with tf.name_scope('sample_embedding'):
                    self.epsilon = tf.random_normal(tf.shape(self.log_var), 0, 1, name='epsilon')
                    self.embedding = self.embedding + self.epsilon * tf.exp(self.log_var)

            # Pass embedding through the decode portion of the network
            self.decoded_X = self.decode(self.embedding)

            # Compute the reconstruction loss.
            with tf.name_scope('loss'):
                if self.loss_func == 'mean_squared':
                    self.squared_errors = tf.square(self.decoded_X - self.true_X)
                    self.reconstruction_loss = tf.sqrt(tf.reduce_mean(self.squared_errors))
                elif self.loss_func == 'cross_entropy':
                    self.reconstruction_loss = - tf.reduce_sum(self.true_X * tf.log(self.decoded_X))
                elif self.loss_func == 'sigmoid_cross_entropy':
                    self.reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.decoded_X, labels=self.true_X))
                    self.decoded_X = tf.nn.sigmoid(self.decoded_X)

            # Add weight decay regularization term to loss.
            with tf.name_scope('weight_regularization'):
                self.reg_loss = self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.encode_weights])
                self.reg_loss += self.weight_penalty * sum([tf.nn.l2_loss(w) for w in self.decode_weights])
                if self.variational:
                    self.reg_loss += self.weight_penalty * tf.nn.l2_loss(self.variance_weights)
            
            # Add variation loss, if necessary.
            if self.variational:
                with tf.name_scope('variational_loss'):
                    self.kl_divergence = -0.5 * tf.reduce_sum(1 + 2 * self.log_var 
                                                            - tf.pow(self.embedding, 2) 
                                                            - tf.exp(2 * self.log_var), 
                                                            reduction_indices=1)
                    self.total_loss = tf.reduce_mean(self.reconstruction_loss + self.kl_divergence) + self.reg_loss
            else:
                self.total_loss = self.reconstruction_loss + self.reg_loss
            
            # Training step with optimizer.
            self.opt_step = self.optimizer(self.tf_learning_rate).minimize(self.total_loss)

            # Additional classification layers
            if self.classification_layer_sizes is not None:
                self.build_classification_graph()

            # Logistics
            self.init = tf.global_variables_initializer()

    def build_classification_graph(self):
        """Builds additional tensorflow graph layers to perform classification."""
        with tf.name_scope('classification'):
            # Data placeholders.
            self.true_Y = tf.placeholder(tf.float32, name="true_Y")
            self.int_true_Y = tf.cast(self.true_Y, dtype=tf.int32)

            # Run classification portion of network from embedding.
            self.logits = self.classify(self.embedding)
            
            # Compute classification loss.
            if self.classification_loss_func == 'sigmoid_cross_entropy':
                self.classification_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.true_Y))
            else:
                print "Using softmax CE loss for classification"
                
                self.classification_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.int_true_Y))

            # Add weight decay regularization term to loss
            self.classification_loss += self.classification_weight_penalty * sum([tf.nn.l2_loss(w) for w in self.classification_weights])

            self.classification_opt_step = self.optimizer(self.classification_learning_rate).minimize(self.classification_loss)

            # Code for making predictions and evaluating them.
            self.class_probabilities = tf.nn.sigmoid(self.logits)
            if self.classification_loss_func == 'sigmoid_cross_entropy':
                self.predictions = tf.cast(tf.round(self.class_probabilities), dtype=tf.int32)
            else:
                self.predictions = tf.cast(tf.argmax(self.class_probabilities, axis=1), dtype=tf.int32)
            self.correct_prediction = tf.equal(self.predictions, self.int_true_Y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def encode(self, X):
        """Runs data through the encoder portion of the network.
        
        Args:
            X: A tensor containing input data.
        Returns: A tensor embedding
        """
        hidden = X
        for i in range(len(self.encode_weights)):
            with tf.name_scope('enc_layer' + str(i)) as scope:
                if self.variational and i == len(self.encode_weights) - 1:
                    self.log_var = tf.matmul(hidden, self.variance_weights) + self.variance_bias

                hidden = tf.matmul(hidden, self.encode_weights[i]) + self.encode_biases[i]
                
                if i < len(self.encode_weights)-1:
                    # Apply activation function
                    hidden = self.apply_activation(hidden)

                    # Apply dropout
                    hidden = tf.nn.dropout(hidden, self.tf_dropout_prob) 
        return hidden

    def apply_activation(self, X, activation_func=None):
        """Applies a particular tensorflow activation function based on 
        a string description.
        
        Args:
            X: A tensor to be passed through the activation function
            activation_func: A string description. If None, will use the class
                default.
        """
        if activation_func is None:
            activation_func = self.activation_func

        if activation_func == 'relu':
            return tf.nn.relu(X) 
        elif activation_func == 'tanh':
            return tf.nn.tanh(X)
        elif activation_func == 'softsign':
            return tf.nn.softsign(X)
        elif activation_func == 'softplus':
            return tf.nn.softplus(X)
        return X # linear

    def decode(self, embedding):
        """Runs an embedding through the decoder portion of the network.
        
        Args: 
            embedding: A tensor containing an embedding created by the encoder.
        Returns: A tensor X', the decoded version of the embedding
        """
        X = embedding
        for i,w in enumerate(self.decode_weights):
            with tf.name_scope('dec_layer' + str(i)) as scope:
                # tf.matmul is a simple fully connected layer. 
                X = tf.matmul(X, w) + self.decode_biases[i]
                
                if i < len(self.decode_weights)-1:
                    # Apply activation function
                    X = self.apply_activation(X)

                    # Apply dropout
                    X = tf.nn.dropout(X, self.tf_dropout_prob) 
        return X

    def classify(self, embedding):
        """Runs an embedding vector through the classification layers of the 
        network.
        
        Args:
            embedding: A tensor embedding.
        """
        X = embedding
        for i,w in enumerate(self.classification_weights):
            with tf.name_scope('classification_layer' + str(i)) as scope:
                # tf.matmul is a simple fully connected layer. 
                X = tf.matmul(X, w) + self.classification_biases[i]
                
                if i < len(self.decode_weights)-1:
                    # Apply activation function
                    X = self.apply_activation(X, 
                            activation_func=self.classification_activation_func) 

                    # Apply dropout
                    X = tf.nn.dropout(X, self.tf_dropout_prob) 
        return X

    def initialize_session(self):
        """Initializes a tensorflow session and saver before training the network."""
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init)
        with self.graph.as_default():
            self.saver = tf.train.Saver()

    def train(self, num_steps=30000, record_every_nth=None, save_every_nth=None):
        """Trains using stochastic gradient descent (SGD). 
        
        Runs batches of training data through the model for a given
        number of steps.
        
        Args:
            num_steps: The number of times a batch of training data will be used
                to train the network.
            record_every_nth: The number of steps before it will evaluate and save
                the current training and validation loss. 
            save_every_nth: The number of steps before it will save a checkpoint of
                the model.
        """
        self.set_record_save(record_every_nth, save_every_nth)

        with self.graph.as_default():
            for step in range(num_steps):
                # Grab a batch of data to feed into the placeholders in the graph.
                X = self.data_loader.get_unsupervised_train_batch(self.batch_size)
                noisy_X = self.add_noise_to_batch(X)
                feed_dict = {self.noisy_X: noisy_X, self.true_X: X, 
                             self.tf_dropout_prob: self.dropout_prob}

                # Output/save the training and validation performance every few steps.
                if step % self.record_every_nth == 0:
                    train_loss, val_loss = self.evaluate_performance(feed_dict)
                    self.train_loss.append(train_loss)
                    self.val_loss.append(val_loss)

                    if self.verbose:
                        print "Training iteration", step
                        print "\t Training loss", train_loss
                        print "\t Validation loss", val_loss

                if step > 0 and step % self.save_every_nth == 0:
                    # Save a checkpoint of the model
                    self.save_model()
    
                # Update parameters in the direction of the gradient computed by
                # the optimizer.
                _ = self.session.run([self.opt_step], feed_dict)

    def set_record_save(self, record_every_nth, save_every_nth):
        """Set the number of steps before the model records and saves its progress.
        
        Args:
            record_every_nth: The number of steps before it will evaluate and save
                the current training and validation loss. 
            save_every_nth: The number of steps before it will save a checkpoint of
                the model.
        """
        if record_every_nth is not None:
            self.record_every_nth = record_every_nth
        if save_every_nth is not None:
            self.save_every_nth = save_every_nth

    def train_classification(self, num_steps=30000, record_every_nth=None, save_every_nth=None):
        """Perform supervised training of the embedding and classification layers via training labels.
        
        Args:
            num_steps: The number of times a batch of data will be used for training.
            record_every_nth: The number of steps before it will evaluate and save
                the current training and validation loss. 
            save_every_nth: The number of steps before it will save a checkpoint of
                the model.
        """
        self.set_record_save(record_every_nth, save_every_nth)

        with self.graph.as_default():
            for step in range(num_steps):
                # Get a batch of training data and associated labels
                X, Y = self.classification_data_loader.get_supervised_train_batch(self.classification_batch_size)
                noisy_X = self.add_noise_to_batch(X)
                feed_dict = {self.noisy_X: noisy_X, self.true_Y: Y, 
                             self.tf_dropout_prob: self.classification_dropout_prob}

                # Output/save the training and validation performance every few steps.
                if step % self.record_every_nth == 0:
                    train_loss, train_acc, val_loss, val_acc = self.evaluate_classification_performance(feed_dict)
                    self.train_acc.append(train_acc)
                    self.val_acc.append(val_acc)
                    self.classification_train_loss.append(train_loss)
                    self.classification_val_loss.append(val_loss)

                    if self.verbose:
                        print "Training iteration", step
                        print "\t Training loss", train_loss
                        print "\t Validation loss", val_loss
                        print "\t Training accuracy", train_acc
                        print "\t Validation accuracy", val_acc

                if step > 0 and step % self.save_every_nth == 0:
                    # Save a checkpoint of the model
                    self.save_model()
    
                # Update parameters in the direction of the gradient computed by
                # the optimizer.
                _ = self.session.run([self.classification_opt_step], feed_dict)

    def mask_modality(self, X, row, mod_i):
        """Given a design matrix X, will mask all data from the set of features 
        associated with modality mod_i in a certain row.
        
        Args:
            X: A numpy matrix.
            row: The index of the row in which the modality should be masked.
            mod_i: The index of the modality to mask. 
        Returns:
            The X matrix with the modality masked. 
        """
        # Computes the column indices for the masked modality
        start_i = self.data_loader.modality_start_indices[mod_i]
        end_i = self.data_loader.modality_start_indices[mod_i+1]

        # Masks the modality with the class's mask value, self.mask_with
        X[row,start_i:end_i] = self.mask_with * np.ones(end_i-start_i)
        return X

    def add_noise_to_batch(self, X, missing_modes=[]):
        """Alters a batch of data X so that it contains noise. 
        
        Args: 
            X: A 2D numpy array containing the batch of data.
            missing_modes: A list of modality indices used to specify which 
                modalities can go missing.
        Returns:
            A new numpy array containing the noisy batch.
        """
        new_X = copy.deepcopy(X)
        num_feats = np.shape(new_X)[1]
        for i in range(len(new_X)):
            # randomly 0 out 5% of the data 
            idx = np.random.choice(num_feats, size=int(num_feats*.05))
            new_X[i,idx] = 0

            # drop out a modality or modalities
            if self.intelligent_noise:
                # Intelligent or structured noise is used to make modalities go missing in the
                # same proportion that they are missing in the real noisy data.
                noise_i = np.argmax(np.random.multinomial(1, pvals=self.noise_type_percentages))
                missing_modalities = self.noise_types[noise_i]
                if len(missing_modes)>0:
                    missing_modalities = missing_modes
                for m in missing_modalities:
                    mod_i = self.data_loader.modality_names.index(m)
                    new_X = self.mask_modality(new_X, i, mod_i)
            else:
                # If not using intelligent noise, just drop a modality randomly.
                for _ in range(self.num_modalities_to_drop):
                    mod_i = np.random.randint(0, self.data_loader.num_modalities)
                    new_X = self.mask_modality(new_X, i, mod_i)
                
        return new_X
    
    def evaluate_performance(self, train_feed_dict=None):
        """Tests the reconstruction performance of the autoencoder on a batch of training
        and validation data.
        
        Args:
            train_feed_dict: If a dictionary containing training data has already been 
                constructed in the caller function, can send it here for efficiency.
        Returns:
            2 floats: the training loss and the validation loss."""
        # Get a batch of training data if one was not sent to the function.
        if train_feed_dict is None:
            X = self.data_loader.get_unsupervised_train_batch(self.batch_size)
            train_feed_dict = {self.noisy_X: X, self.true_X: X, self.tf_dropout_prob: 1.0}

        # Grab a batch of validation data too.
        val_X = self.data_loader.get_unsupervised_val_batch(200)
        noisy_val_X = self.add_noise_to_batch(val_X)
        val_feed_dict = {self.noisy_X: noisy_val_X, 
                         self.true_X: val_X, 
                         self.tf_dropout_prob: 1.0} # no dropout during evaluation

        # Compute the losses.
        train_loss, step = self.session.run([self.reconstruction_loss, 
                                            self.global_step], 
                                            train_feed_dict)
        val_loss = self.session.run(self.reconstruction_loss, 
                                    val_feed_dict)

        # Normalize the loss by the size of the batch for comparison purposes
        if 'entropy' in self.loss_func:
            train_loss = train_loss/len(train_feed_dict[self.true_X])
            val_loss = val_loss/len(val_X)

        return train_loss, val_loss 
    
    def evaluate_classification_performance(self, train_feed_dict=None):
        """Tests the classification performance on training and validation data.

        Args:
            train_feed_dict: If a dictionary containing training data has already been 
                constructed in the caller function, can send it here for efficiency.
        Returns:
            4 floats: training loss, training accuracy, validation loss, validation
            accuracy
        """
        if train_feed_dict is None:
            X, Y = self.classification_data_loader.get_supervised_train_batch(self.classification_batch_size)
            train_feed_dict = {self.noisy_X: X, self.true_Y: Y, 
                               self.tf_dropout_prob: self.dropout_prob}
        
        val_X, val_Y = self.classification_data_loader.get_supervised_val_batch(200)
        val_feed_dict = {self.noisy_X: val_X, 
                         self.true_Y: val_Y, 
                         self.tf_dropout_prob: 1.0} # no dropout during evaluation

        train_loss, train_acc = self.session.run([self.classification_loss, 
                                                  self.accuracy],train_feed_dict)
        val_loss, val_acc = self.session.run([self.classification_loss, 
                                              self.accuracy], val_feed_dict)

        return train_loss, train_acc, val_loss, val_acc

    def save_model(self, file_name=None, directory=None):
        """Saves a checkpoint of the model and a .npz file with stored rewards.

        Args:
            file_name: String name to use for the checkpoint and rewards files.
                Defaults to self.model_name if None is provided.
            directory: Directory where the checkpoint will be saved. Defaults to
                self.checkpoint_dir if None is provided.
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
        training_epochs = len(self.train_loss) * self.record_every_nth
        self.saver.save(self.session, save_loc, global_step=training_epochs)
        
        npz_name = os.path.join(directory, 
                                file_name + '-' + str(training_epochs))
        np.savez(npz_name,
                train_loss=self.train_loss,
                val_loss=self.val_loss,
                layer_sizes=self.layer_sizes,
                variational=self.variational,
                dropout_prob=self.dropout_prob,
                weight_penalty=self.weight_penalty,
                activation_func=self.activation_func,
                loss_func=self.loss_func,
                weight_initialization=self.weight_initialization)

    def load_saved_model(self, directory=None, checkpoint_name=None,
                         npz_file_name=None):
        """Restores this model from a saved checkpoint.

        Args:
        directory: Path to directory where checkpoint is located. If 
            None, defaults to self.output_dir.
        checkpoint_name: The name of the checkpoint within the 
            directory.
        npz_file_name: The name of the .npz file where the stored
            rewards are saved. If None, will not attempt to load stored
            rewards.
        """
        print "-----Loading saved model-----"
        if directory is None:
            directory = self.output_dir

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

            self.train_loss = list(npz_file['train_loss'])
            self.val_loss = list(npz_file['val_loss'])
            
            if self._print_if_saved_setting_differs(self.layer_sizes, 'layer_sizes', npz_file):
                self.layer_sizes = npz_file['layer_sizes']
            if self._print_if_saved_setting_differs(self.variational, 'variational', npz_file):
                self.variational = npz_file['variational']
            if self._print_if_saved_setting_differs(self.dropout_prob, 'dropout_prob', npz_file):
                self.dropout_prob = npz_file['dropout_prob']
            if self._print_if_saved_setting_differs(self.weight_penalty, 'weight_penalty', npz_file):
                self.weight_penalty = npz_file['weight_penalty']
            if self._print_if_saved_setting_differs(self.activation_func, 'activation_func', npz_file):
                self.activation_func = npz_file['activation_func']
            if self._print_if_saved_setting_differs(self.loss_func, 'loss_func', npz_file):
                self.loss_func = npz_file['loss_func']
            if self._print_if_saved_setting_differs(self.weight_initialization, 'weight_initialization', npz_file):
                self.weight_initialization = npz_file['weight_initialization']

        # Re-initializes the tensorflow graph based on the hyperparameters loaded from the file
        self.graph = tf.Graph()
        self.build_graph()
        self.initialize_session()
        self.saver.restore(self.session, checkpoint_file)
    
    def _print_if_saved_setting_differs(self, class_var, setting_name, npz_file):
        """If the saved hyperparameter in an npz_file is different than the current
        class setting, will print an error message.
        
        Args:
            class_var: The variable containing the current setting for this 
                hyperparameter in the class.
            setting_name: The string name of the setting, used to index the npz_file 
            npz_file: An variable containing data loaded from an npz file 
        Returns:
            A Boolean that will be true if the setting is different. 
        """
        if setting_name not in npz_file.keys():
            print "ERROR! The setting", setting_name, "is not in the saved model file."
            print "Using default value:", class_var
            print ""
            return False
        
        equal = True
        if type(class_var) is list:
            if len(class_var) != len(npz_file[setting_name]):
                equal = False
            else:
                for i in range(len(class_var)):
                    if class_var[i] != npz_file[setting_name][i]:
                        equal = False
        elif class_var != npz_file[setting_name]:
            equal = False
            
        if not equal:
            print "WARNING! Saved setting for", setting_name, "is different!"
            print "\tModel's current value for", setting_name, "is", class_var
            print "\tBut it was saved as", npz_file[setting_name]
            print "Overwriting setting", setting_name, "with new value:", npz_file[setting_name]
            print ""
        return True
    
    def set_classification_params(self, weight_penalty=None, learning_rate=None,
                                  dropout_prob=None, activation_func=None, batch_size=None, 
                                  loss_func=None, suppress_warning=False):
        """Sets all of the model's classification hyperparameters if classification will be used. If
        a hyperparameter is not included in the arguments will use the class default.

        Note: if these settings are changed, the model we need to re-construct the computation graph, 
        erasing any learned weights that are unsaved. 
        
        Args:
            weight_penalty: The strength of the L2 weight regularization penalty.
            learning_rate: The initial learning rate used for training with classification. 
            dropout_prob: The probability that a node in the network will not
                be dropped out during training. Set to < 1.0 to apply dropout, 
                1.0 to remove dropout.
            activation_func: A string naming the activation function used in the classification 
                portion of the network. 
            batch_size: Number of samples in a classification training batch.
            loss_func: The classification loss function to use. 
            suppress_warning: A Boolean which if true, will not print a statement warning that the graph 
                will be reset. 
        """

        self.classification_learning_rate = learning_rate if learning_rate is not None else self.classification_learning_rate
        self.classification_dropout_prob = dropout_prob if dropout_prob is not None else self.classification_dropout_prob
        self.classification_activation_func = activation_func if activation_func is not None else self.classification_activation_func
        self.classification_weight_penalty = weight_penalty if weight_penalty is not None else self.classification_weight_penalty
        self.classification_batch_size = batch_size if batch_size is not None else self.classification_batch_size
        self.classification_loss_func = loss_func if loss_func is not None else self.classification_loss_func

        if not suppress_warning:
            print "In order for these changes to take effect, the model will now reconstruct the computation graph. Unsaved changes will be lost."
        self.rebuild_reinitialize()

    def predict(self, X):
        """Gets the autoencoder's reconstructed version of some data X
        
        Args: 
            X: a matrix of data in the same format as the training
                data. 
        Returns:
            Reconstructed version of the data
        """
        feed_dict = {self.noisy_X: X,
                     self.true_X: X, 
                     self.tf_dropout_prob: 1.0} # no dropout during evaluation
        
        reconstruction, loss =  self.session.run([self.decoded_X, 
                                                  self.reconstruction_loss], 
                                                 feed_dict)
        if 'entropy' in self.loss_func:
            loss = loss / len(X)
        return reconstruction, loss
    
    def plot_training_progress(self):
        """Plots the training and validation performance as evaluated 
        throughout training.
        """
        x = [self.record_every_nth * i for i in np.arange(len(self.train_loss))]
        plt.figure()
        plt.plot(x,self.train_loss)
        plt.plot(x,self.val_loss)
        plt.legend(['Train', 'Validation'], loc='best')
        plt.xlabel('Training epoch')
        plt.ylabel('Loss')
        plt.show()

    def plot_classification_training_progress(self):
        """Plots the training and validation progress as evaluated 
        throughout training the classification layers.
        """
        x = [self.record_every_nth * i for i in np.arange(len(self.train_acc))]
        plt.figure()
        plt.plot(x,self.train_acc)
        plt.plot(x,self.val_acc)
        plt.legend(['Train', 'Validation'], loc='best')
        plt.xlabel('Training epoch')
        plt.ylabel('Accuracy')
        plt.show()

        x = [self.record_every_nth * i for i in np.arange(len(self.classification_train_loss))]
        plt.figure()
        plt.plot(x,self.classification_train_loss)
        plt.plot(x,self.classification_val_loss)
        plt.legend(['Train', 'Validation'], loc='best')
        plt.xlabel('Training epoch')
        plt.ylabel('Classification loss')
        plt.show()

    def test_on_validation(self):
        """Returns performance on the model's validation set.
        
        Returns: Float loss"""
        loss = self.get_performance_on_data(self.data_loader.val_X)
        print "Final loss on validation data is:", loss
        return loss
        
    def test_on_test(self):
        """Returns performance on the model's test set.
        
        Returns: Float loss"""
        print "WARNING! Only test on the test set when you have finished choosing all of your hyperparameters!"
        print "\tNever use the test set to choose hyperparameters!!!"
        loss = self.get_performance_on_data(self.data_loader.test_X)
        print "Final loss on test data is:", loss
        return loss

    def get_performance_on_data(self, X):
        """Computes the autoencoder's reconstruction loss on some data X.

        Args:
            X: an array of data.
        Returns: Float loss. 
        """
        feed_dict = {self.noisy_X: X, self.true_X: X, self.tf_dropout_prob: 1.0} 
        loss = self.session.run(self.reconstruction_loss, feed_dict)

        if 'entropy' in self.loss_func:
            loss = loss / len(X)
        return loss
    
    def get_performance_on_data_with_noise(self, X):
        """Computes the autoencoder's loss in reconstructing some data X after
        adding noise to X.

        Args:
            X: an array of data.
        Returns: Float loss. 
        """
        noisy_X = self.add_noise_to_batch(X)
        feed_dict = {self.noisy_X: noisy_X, self.true_X: X, self.tf_dropout_prob: 1.0} 
        loss = self.session.run(self.reconstruction_loss, feed_dict)

        if 'entropy' in self.loss_func:
            loss = loss / len(X)
        return loss

    def get_classification_predictions(self, X):
        """Gets class label predictions from the classification portion of the 
        network.
        
        Args:
            X: an array of data.
        Returns: A vector of predicted class labels.
        """
        feed_dict = {self.noisy_X: X, self.tf_dropout_prob: 1.0} 
        preds = self.session.run(self.predictions, feed_dict)
        return preds
    
    def get_classification_predictions_from_df(self):
        """Gets class label predictions by classifying directly from the data 
        contained in the model's classification data pandas dataframe.
        
        Returns:
            The same dataframe with the predictions added as a column. 
        """
        df = copy.deepcopy(self.classification_data_loader.df)
        X = df[self.classification_data_loader.wanted_feats].as_matrix()
        preds = self.get_classification_predictions(X)
        assert(len(X) == len(preds))
        for i,label in enumerate(self.classification_data_loader.wanted_labels):
            df['predictions_'+label] = preds[:,i]
        return df

    def get_embedding(self, X, add_noise=False):
        """Get the autoencoder's embedding (encoding) of some data X. 

        Args:
            X: an array of data (N x M)
            add_noise: A Boolean. If True, will add noise to the batch
                before obtaining the embedding.
        Returns: A N x E matrix of embeddings, where N is the original
            length of X, and E is the size of the embedding.
        """
        if add_noise:
            noisy_X = self.add_noise_to_batch(X)
        else: 
            noisy_X = X
        
        # with noise
        embedding = self.session.run(self.embedding, feed_dict={self.noisy_X: noisy_X, 
                                                                self.tf_dropout_prob: 1.0})
        return embedding

    def get_performance_on_extra_noisy_data(self):
        """Computes reconstruction performance on a set of extra data that may contain real noise.

        If this data has not been loaded before the function will load it.

        Returns: reconstruction loss on the training data in the extra file. 
        """
        if self.extra_noisy_data_loader is None:
            if self.extra_data_filename is None:
                print "Error! Was not provided with location of extra data. Cannot perform this command"
                return
            else:
                self.extra_noisy_data_loader = data_funcs.DataLoader(
                    self.extra_data_filename, normalize_and_fill=False,
                    subdivide_physiology_features=self.subdivide_physiology,
                    normalization=self.normalization,
                    fill_missing_with=self.fill_missing_with, 
                    fill_gaps_with=self.mask_with)
        
        return self.get_performance_on_data(self.extra_noisy_data_loader.train_X)

    def view_reconstruction(self, dataset, with_noise=True):
        """Plots original and reconstructed version of a random data vector. 

        Args:
            dataset: A matrix of data
            with_noise: A Boolean. If true, will make a corrupted/noisy version
                of the vector and plot this as well. 
        """
        i = np.random.randint(0, len(dataset))
        X = dataset[i,:]
        X = np.reshape(X, [1,-1])
        
        if with_noise:
            noisy_X = self.add_noise_to_batch(X)
            noisy_X = np.reshape(noisy_X, [1,-1])
            legend = ["Noisy X","X","X'"]
        else: 
            noisy_X = X
            legend = ["X", "X'"]
        
        X_bar = self.session.run(self.decoded_X, feed_dict={self.noisy_X: noisy_X, 
                                                            self.true_X: X,
                                                            self.tf_dropout_prob: 1.0})
        
        plt.figure()
        if with_noise:
            plt.plot(np.reshape(noisy_X,-1))
        plt.plot(np.reshape(X,-1))
        plt.plot(np.reshape(X_bar, -1), c='r')
        plt.legend(legend, loc='best')
        plt.show()

    def convert_file_to_embeddings(self, filename, path, file_descriptor=""):
        """Takes an entire csv file of data, converts each row to an autoencoder
        embedding, and saves it.

        Args:
            filename: String name of the file to convert
            path: String path to the file to convert
            file_descriptor: A string to use in saving the new embedding file. 
        """
        # Load file with pandas
        df = pd.DataFrame.from_csv(path + filename)

        wanted_feats = data_funcs.get_wanted_feats_from_df(df)
        # TODO: add an assert to check this is exactly the same as self.data_loader.wanted_feats
        
        # Pull extra information out of the file and save it
        other_feats = [f for f in df.columns.values if f not in wanted_feats]
        embed_df = df[other_feats]

        # Get the actual data out of the file 
        X = data_funcs.get_matrix_for_dataset(df, wanted_feats, dataset=None)
        
        # Compute the embeddings
        embedding = self.session.run(self.embedding, 
                                     feed_dict={self.noisy_X: X, 
                                                self.tf_dropout_prob: 1.0})
        
        # Save embeddings into the dataframe, save to a .csv file
        for c in range(np.shape(embedding)[1]):
            embed_df['ae_embedding_dim' + str(c)] = X[:,c]
        embed_df.to_csv(path + 'embedding-' + file_descriptor + filename)

    def fill_missing_data_in_file(self, filename, path, file_descriptor=""):
        """Creates a reconstruction for every row of data in a file, then replaces portions 
        of data that were missing in the original file with the reconstructed version.

        Args:
            filename: String name of the file to convert
            path: String path to the file to convert
            file_descriptor: A string to use in saving the new reconstructed file. 
        """
        # Load csv with pandas
        df = pd.DataFrame.from_csv(path + filename)

        # Get actual data from file, feed it through the autoencoder
        X = df[self.data_loader.wanted_feats].as_matrix()
        Xbar, loss = self.predict(X)

        # Fill gaps that were originally missing in the file with the reconstruction
        df = self.data_loader.fill_df_with_reconstruction(df, Xbar, plot_to_debug=False)

        # Save to csv
        df.to_csv(path + 'MMAE_filled-' + file_descriptor + filename)

    def get_reconstruction_loss_per_modality(self, X):
        """Given a matrix of data X, loops through each modality in turn and masks out the data
        from that modality, reconstructs it using the autoencoder, and computes the reconstruction
        error.
        
        Args:
            X: an array of data
        Returns: An array of root mean squared error values, one for each modality."""
        rms = [np.nan] * len(self.data_loader.modality_names)
        for i, name in enumerate(self.data_loader.modality_names):
            # Make a noisy version of X with the data from modality i masked.
            start_i = self.data_loader.modality_start_indices[i]
            end_i = self.data_loader.modality_start_indices[i+1]
            noisy_X = copy.deepcopy(X)
            noisy_X[:,start_i:end_i] = -1.0 * np.ones((len(X),end_i-start_i))

            if self.verbose: print "DEBUG: feeding in noisy matrix. Average value over masked area:", np.mean(noisy_X[:,start_i:end_i])

            # Get reconstruction
            reconstruction, loss = self.predict(noisy_X)
            
            # Get reconstruction error on the masked modality data only
            feats = X[:,start_i:end_i]
            reconstruct_feats = reconstruction[:,start_i:end_i]
            rms[i] = get_rmse(feats,reconstruct_feats)
            print "RMS for modality", name, "is", rms[i], "\n"
        
        return rms

def get_rmse(x,y):
    """Computes the root mean squared error between two arrays, x and y."""
    return np.sqrt(mean_squared_error(x,y))
