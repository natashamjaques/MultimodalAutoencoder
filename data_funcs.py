""" This file provides functions for loading data from a saved 
    .csv file. 

    To use this code with classification, the file should contain 
    at least one column with 'label' in the name. This column
    should be an outcome we are trying to classify; e.g. if we 
    are trying to predict if someone is happy or not, we might 
    have a column 'happy_label' that has values that are either 
    0 or 1. 
    
    We also assume the file contains a column named 'dataset' 
    containing the dataset the example belongs to, which can be 
    either 'Train', 'Val', or 'Test'. 

    Other typical columns include 'user_id', 'timestamp', 'ppt_id',
    or columns with 'logistics_' as a prefix. A common logistics 
    column is 'logistics_noisy', which describes whether the data
    has missing modalities or not. 
"""

import pandas as pd
import numpy as np
import operator
import copy
import matplotlib.pyplot as plt

NUM_CROSS_VAL_FOLDS = 5

class DataLoader:
    def __init__(self, filename, supervised=True, suppress_output=False, cross_validation=False,
                 normalize_and_fill=True, normalization='between_0_and_1', fill_missing_with=0, 
                 fill_gaps_with=None, extract_modalities=True, subdivide_physiology_features=False, 
                 wanted_label=None, labels_to_sign=False, separate_noisy_data=True):
        """Class that handles extracting numpy data matrices for train, validation,
        and test sets from a .csv file. Also normalizes and fills the data. 
        
        filename: Name of .csv file from which data will be loaded.
        supervised: If True, will load a Y matrix representing classification labels. 
            Otherwise will not.
        suppress_output: If True, will not print statements about what is being done 
            to load the data.
        cross_validation: If True, will divide the data into cross validation folds.
            Setting the fold to a new value will cause the train/validation data to
            be altered.
        normalize_and_fill: If True, will normalize the features and fill missing values.
        normalization: How to normalize the data. Can be 'z_score', 'between_0_and_1', 
            or None.
        fill_missing_with: Value to use for filling the missing data.
        fill_gaps_with: Value to use for filling large missing swatches of data corresponding
            to entire missing sensors/modalities.
        extract_modalities: If True, will look for feature subtypes within the features.
            Will only work if features are named with a prefix indicating their feature type. 
            I.e. "survey_sleep, survey_exercise, call_unique_contacts, call_duration", etc
        subdivide_physiology_features: Our affective data contains many features 
            related to physiology. If True, will subdivide them into smaller 
            feature sets.
        wanted_label: If this is set to a string instead of None, the data_loader will 
            only try to classify one of the many possible labels in the file.
        labels_to_sign: If set to True and the model is supervised, the classification
            labels will be changed from 0/1 to -1/1. 
        separate_noisy_data: Specific to denoising applications with some clean and some
            noisy data. If true, will store extra variables for the noisy/clean data.
        """
        if not suppress_output:
            print "-----Loading data-----"

        # memorize arguments
        self.filename = filename
        self.supervised = supervised
        self.normalize_and_fill = normalize_and_fill
        self.normalization = normalization
        self.cross_validation = cross_validation
        self.subdivide_phys = subdivide_physiology_features
        self.suppress_output = suppress_output
        self.extract_modalities = extract_modalities
        self.labels_to_sign = labels_to_sign
        self.fill_missing_with = fill_missing_with
        self.fill_gaps_with = fill_gaps_with
        self.separate_noisy_data = separate_noisy_data

        # Extract dataframe from csv
        self.df = pd.DataFrame.from_csv(filename)
        if self.cross_validation:
            self.df = self.assign_cross_val_folds(self.df)
            self.fold = 0
        self.wanted_feats = get_wanted_feats_from_df(self.df)

        # Decide if there are classification labels to care about
        if not supervised:
            self.wanted_labels = None
        elif wanted_label is not None:
            self.wanted_labels = [wanted_label]
            self.num_labels = None
        else:
            self.wanted_labels = [y for y in self.df.columns.values if 'label' in y or 'Label' in y]
            self.num_labels = len(self.wanted_labels)
            if len(self.wanted_labels) == 1:
                self.num_classes = len(self.df[self.wanted_labels[0]].unique())
        self.df = remove_rows_with_no_label(self.df, self.wanted_labels)

        # Normalize the dataframe and fill missing values
        if normalize_and_fill:
            self.df = normalize_fill_df(self.df, self.wanted_feats, 
                                        suppress_output=suppress_output, remove_cols=True, 
                                        normalization=normalization, fill_missing=fill_missing_with,
                                        fill_gaps=self.fill_gaps_with)
    
        # Extract data matrices from dataframe
        self.get_matrices_from_df()

        self.num_feats = self.get_feature_size()
        if not suppress_output: 
            print len(self.train_X), "rows in training data"
            print len(self.val_X), "rows in validation data"
            print len(self.test_X), "rows in testing data"
            print "Number of features:", self.num_feats
        
        # This code is used to find feature types within the feature names. 
        if extract_modalities:
            self.modality_dict = get_modality_dict(self.wanted_feats, subdivide_phys=self.subdivide_phys)
            self.modality_names, self.modality_start_indices = get_modality_names_indices(self.modality_dict)
            self.modality_start_indices.append(self.num_feats)
            self.num_modalities = len(self.modality_dict)

            if not suppress_output:
                print "Found the following feature types:"
                for (i, val) in enumerate(self.modality_start_indices[:-1]):
                    print "\t", self.modality_names[i], "starting at feature", val
            
        
        print ""
    
    def get_matrices_from_df(self):
        """Extracts X and Y matrices from the class's dataframe based on the wanted 
        features and labels"""
        # Load all data classes 
        self.test_X, self.test_Y = get_matrices_for_dataset(self.df, self.wanted_feats, 
                                                            self.wanted_labels, 'Test',
                                                            labels_to_sign=self.labels_to_sign)
        if self.separate_noisy_data:
            (self.clean_test_X, self.clean_test_Y, self.noisy_test_X, 
            self.noisy_test_Y) = self.get_noisy_clean_data_for_dataset('Test')

        if not self.cross_validation:
            self.train_X, self.train_Y = get_matrices_for_dataset(self.df, self.wanted_feats, 
                                                                self.wanted_labels, 'Train',
                                                                labels_to_sign=self.labels_to_sign)
            
            self.val_X, self.val_Y = get_matrices_for_dataset(self.df, self.wanted_feats, 
                                                            self.wanted_labels, 'Val',
                                                            labels_to_sign=self.labels_to_sign)
            
            if self.separate_noisy_data:
                (self.clean_train_X, clean_train_Y, self.noisy_train_X, 
                self.noisy_train_Y) = self.get_noisy_clean_data_for_dataset('Train')
                (self.clean_val_X, clean_val_Y, self.noisy_val_X, 
                self.noisy_val_Y) = self.get_noisy_clean_data_for_dataset('Val')
        else:
            self.set_to_cross_validation_fold(0)

    def get_unsupervised_train_batch(self, batch_size):
        """Get a random batch of data from the X training matrix
        
        Args:
            batch_size: Integer number of examples in the batch.
        """
        idx = np.random.choice(len(self.train_X), size=batch_size)
        return self.train_X[idx]

    def get_supervised_train_batch(self, batch_size):
        """Get a random batch of X data and Y labels from the training set
        
        Args:
            batch_size: Integer number of examples in the batch.
        """
        idx = np.random.choice(len(self.train_X), size=batch_size)
        return self.train_X[idx], self.train_Y[idx]
            
    def get_unsupervised_val_batch(self, batch_size):
        """Randomly sample a set of X data from the validation set

        Args:
            batch_size: Integer number of examples in the batch.
        """
        idx = np.random.choice(len(self.val_X), size=batch_size)
        return self.val_X[idx]

    def get_supervised_val_batch(self, batch_size):
        """Randomly sample a set of X data and Y labels from the validation set

        Args:
            batch_size: Integer number of examples in the batch.
        """
        idx = np.random.choice(len(self.val_X), size=batch_size)
        return self.val_X[idx], self.val_Y[idx]

    def get_val_data(self):
        """Return the entire validation set, both X data and Y labels
        
        Returns: matrices X and Y
        """
        return self.val_X, self.val_Y

    def get_feature_size(self):
        """Get the number of features in the dataset
        
        Returns: integer number of features
        """
        return np.shape(self.train_X)[1]

    def assign_cross_val_folds(self, df):
        """Ensures a dataframe has a column representing the cross validation 
        fold to which every example has been randomly assigned.

        Args:
            df: A pandas dataframe containing X and Y data and following the 
                conventions described at the top of this file.
        Returns: the modified df
        """
        if 'logistics_cv_fold' not in df.columns.values:
            df['logistics_cv_fold'] = df.apply(assign_cv_fold, axis=1)
            df.to_csv(self.filename)
        return df

    def get_noisy_clean_data_for_dataset(self, dset):
        """Returns the noisy and clean portions of a dataset such as 'Train'

        Args:
            dset: String describing the dataset required. Convention is to use
                'Train', 'Test', or 'Val'
        Returns:
            Four numpy matrices: the first are the X data and Y labels from the
                clean portion of the data, the second are noisy X and Y
        """
        clean_df = self.df[self.df['logistics_noisy'] == False]
        clean_X, clean_Y = get_matrices_for_dataset(clean_df, self.wanted_feats, 
                                                    self.wanted_labels, dset,
                                                    labels_to_sign=self.labels_to_sign)

        noisy_df = self.df[self.df['logistics_noisy'] == True]
        noisy_X, noisy_Y = get_matrices_for_dataset(noisy_df, self.wanted_feats, 
                                                    self.wanted_labels, dset,
                                                    labels_to_sign=self.labels_to_sign)

        return clean_X, clean_Y, noisy_X, noisy_Y

    def get_noisy_or_clean_data_matrices(self, df, noisy=True):
        """Returns the X and Y matrices from either the clean or noisy portion
        of the data.

        Args:
            df: A pandas dataframe containing both noisy and clean data, and a
                column entitled 'logistics_noisy' indicating which is which. 
            noisy: A Boolean. If True, will return the noisy data. Otherwise will
                return the clean data.
        """
        noisy_df = df[df['logistics_noisy']==noisy]
        X, Y = get_matrices_for_dataset(noisy_df, self.wanted_feats, 
                                        self.wanted_labels, None,
                                        labels_to_sign=self.labels_to_sign)
        return X,Y

    def set_noisy_clean_data_for_fold(self, fold):
        """Sets the internal class variables for the noisy and clean data matrices
        based on the indicated cross-validation fold.
        
        Args:
            fold: An integer representing the fold number. 
        """
        val_df = self.df[self.df['logistics_cv_fold']==fold]
        train_df = self.df[(self.df['logistics_cv_fold'] != fold) & (self.df['logistics_cv_fold'] != -1)]

        self.noisy_train_X, self.noisy_train_Y = self.get_noisy_or_clean_data_matrices(train_df, True)
        self.clean_train_X, self.clean_train_Y = self.get_noisy_or_clean_data_matrices(train_df, False)
        self.noisy_val_X, self.noisy_val_Y = self.get_noisy_or_clean_data_matrices(val_df, True)
        self.clean_val_X, self.clean_val_Y = self.get_noisy_or_clean_data_matrices(val_df, False)

    def get_cross_val_data_for_fold(self, fold):
        """Gets the training and validation matrices based on the indicated 
        cross-validation fold.  

        Args:
            fold: An integer representing the fold number. 
        """
        val_df = self.df[self.df['logistics_cv_fold']==fold]
        train_df = self.df[(self.df['logistics_cv_fold'] != fold) & (self.df['logistics_cv_fold'] != -1)]
        
        train_X, train_Y = get_matrices_for_dataset(train_df, self.wanted_feats, 
                                                    self.wanted_labels, None,
                                                    labels_to_sign=self.labels_to_sign)
        val_X, val_Y = get_matrices_for_dataset(val_df, self.wanted_feats, 
                                                self.wanted_labels, None,
                                                labels_to_sign=self.labels_to_sign)
        
        return train_X, train_Y, val_X, val_Y
    
    def set_to_cross_validation_fold(self, fold):
        """Sets the internal class fields for the training and validation 
        matrices based on data from the appropriate cross-validation fold.  

        Args:
            fold: An integer representing the fold number. 
        """
        self.fold = fold
        self.train_X, self.train_Y, self.val_X, self.val_Y = self.get_cross_val_data_for_fold(fold)

        if self.separate_noisy_data:
            self.set_noisy_clean_data_for_fold(fold)

    def fill_df_with_reconstruction(self, df, Xbar, plot_to_debug=True):
        """Takes a pandas dataframe containing X data and fills any missing
        data using a reconstructed version in Xbar.

        Note: does not replace data that was not originally missing. Only 
        modifies gaps in the data.

        Args: 
            df: A pandas dataframe containing the data and having the same 
                features and columns as the class's data. 
            Xbar: A matrix with reconstructed data of the same size and shape
                as the original data. 
            plt_to_debug: If True, will generate a plot of how the original 
                data was filled with the reconstructed, for debugging purposes. 

        Returns: the dataframe with the missing data replaced with the 
            reconstruction. 
        """
        X = df[self.wanted_feats].as_matrix()

        num_filled = 0
        for i in range(len(df)):
            x = df[self.wanted_feats].iloc[i]
            missing_idxs = self.find_missing_modalities_indices(x)

            if len(missing_idxs) > 0:
                xfill = copy.deepcopy(x)
                xfill[missing_idxs] = Xbar[i, missing_idxs]
                
                if plot_to_debug:
                    plt.figure()
                    plt.plot(x)
                    plt.plot(Xbar[i,:])
                    plt.plot(xfill, c='r')
                    plt.legend(['Original X',"Reconstructed X",'Filled X'], loc='best')
                    plt.show()
                
                true_i = df.index.values[i]
                df.loc[true_i,self.wanted_feats] = xfill
                
                if plot_to_debug:
                    x2 = df[self.wanted_feats].iloc[i]
                    plt.figure()
                    plt.plot(x2)
                    plt.title('What was actually stored back into the df')
                    plt.show()
            
                num_filled += 1
            
            if plot_to_debug and num_filled > 10:
                print "Okay you've looked at", num_filled, "plots, quitting now"
                break

        print "Filled", num_filled, "rows with reconstruction. This is", num_filled / float(i), "percent"
        return df

    def find_missing_modalities_indices(self, x):
        """Checks a row of data to see which modalities are missing. 

        Args:
            x: A single sample of features for one day - one row of a data matrix.

        Returns:
            The indices of any features that belong to a modality which is missing
        """
        missing_idxs = []
        for i in range(self.num_modalities):
            start_i = self.modality_start_indices[i]
            end_i = self.modality_start_indices[i+1]
            if np.sum(x[start_i:end_i]) == -1 * (end_i - start_i):
                missing_idxs.extend(np.arange(start_i,end_i))
        return missing_idxs

""" Code for cleaning up data to use with classification algorithms / tensorflow """

def normalize_fill_df(data_df, wanted_feats, normalization='z_score', suppress_output=False, 
                      remove_cols=True, fill_missing=0.0, fill_gaps=None):
    """Normalizes the features within a pandas dataframe and fills any missing 
    values.
    
    Args:
        data_df: A pandas dataframe containing the data as well as addtional
            columns following the usual conventions described at the top of
            this file. 
        wanted_feats: A list of string names of columns storing the actual data.
        normalization: A string describing how to normalize the data. Can be 
            'z_score', 'between_0_and_1', or None
        suppress_output: If True, no output statements describing progress will
            be printed. 
        remove_cols: If True, will remove columns that were found to have no data
        fill_missing: A value used to fill missing data that is randomly dispersed
            throughout the dataframe. I.e. If a single feature is missing from a row. 
        fill_gaps: A value used to fill missing data that is part of a large block 
            of data missing because an entire sensor or data source could not be 
            collected. I.e. if all of the location data is missing for a particular
            day, all the missing features would be filled with this value.
    
    Returns:
        A pandas dataframe with the data normalized and filled. 
    """
    if normalization is not None:
        data_df = normalize_columns(data_df, wanted_feats, normalization)
    
    if remove_cols:
        data_df, wanted_feats = remove_null_cols(data_df, wanted_feats)

    if fill_gaps is not None:
        data_df = fill_gaps_in_modalities(data_df, fill_gaps, suppress_output=suppress_output)

    if not suppress_output: print "Filling nan values with", fill_missing
    data_df = data_df.fillna(fill_missing) #if dataset is already filled, won't do anything

    # Shuffle data
    if not suppress_output: print "Randomly reshuffling data"
    data_df = data_df.sample(frac=1)

    return data_df

def remove_rows_with_no_label(data_df, wanted_labels, suppress_output=False):
    """Takes a pandas dataframe and remove any rows which do not have all of 
    the desired supervised classification labels.

    Args:
        data_df: A pandas dataframe containing the data as well as addtional
            columns following the usual conventions described at the top of
            this file. 
        wanted_labels: A list of string names of columns containing the 
            required labels. 
        suppress_output: If True, no output statements describing progress will
            be printed.
    
    Returns: The dataframe with rows removed. 
    """
    if wanted_labels is not None:
        if not suppress_output: print "Original data length was", len(data_df)
        data_df = data_df.dropna(subset=wanted_labels, how='any')
        if not suppress_output: print "After dropping rows with nan in any label column, length is", len(data_df)
    return data_df

def get_wanted_feats_from_df(df):
    """Pulls out the names of columns containing data by removing those 
    columns known to exist for logistics purposes, such as 'label', 'dataset',
    'user_id', etc. 

    Args:
        df: A pandas dataframe containing data as well as addtional
            columns following the usual conventions described at the top of
            this file. 

    Returns: A list of the names of columns thought to contain data.
    """ 
    wanted_feats =  [x for x in df.columns.values if 'user_id' not in x and 
                                                     'timestamp' not in x and
                                                     'label' not in x and 
                                                     'Label' not in x and
                                                     'dataset' not in x and 
                                                     'logistics' not in x and
                                                     'ppt_id' not in x]
    return wanted_feats

def get_matrix_for_dataset(data_df, wanted_feats, dataset):
    """Pulls out a matrix of data X from a pandas dataframe based on the 
    columns listed in wanted_feats and the dataset required. 

    Args: 
        data_df: A pandas dataframe containing data as well as addtional
            columns following the usual conventions described at the top of
            this file. 
        wanted_feats: A list of string names of columns storing the actual data.
        dataset: A string that can be either 'Test', 'Train', or 'Val', by 
            convention.

    Returns: A numpy matrix of data X.
    """
    if dataset is None:
        set_df = data_df
    else:
        set_df = data_df[data_df['dataset']==dataset]
    
    X = set_df[wanted_feats].astype(float).as_matrix()
    X = convert_matrix_tf_format(X)

    return X

def get_matrices_for_dataset(data_df, wanted_feats, wanted_labels, dataset=None, 
                             labels_to_sign=False):
    """Pulls out a matrix of data from a pandas dataframe based on the 
    columns listed in wanted_feats and the dataset required. 

    Args: 
        data_df: A pandas dataframe containing data as well as addtional
            columns following the usual conventions described at the top of
            this file. 
        wanted_feats: A list of string names of columns storing the actual data.
        wanted_labels: A list of string names of columns containing the 
            required labels. 
        dataset: A string that can be either 'Test', 'Train', or 'Val', by 
            convention.
        labels_to_sign: If True, will change labels from {0,1} to {-1,1}.

    Returns: Two numpy matrices X and Y for the data and labels, respectively.
    """
    if dataset is None:
        set_df = data_df
    else:
        set_df = data_df[data_df['dataset']==dataset]
    
    X = set_df[wanted_feats].astype(float).as_matrix()
    X = convert_matrix_tf_format(X)
    
    if wanted_labels is None:
        y = None
    else:
        if len(wanted_labels) == 1:
            y = set_df[wanted_labels[0]].tolist()
        else:
            y = set_df[wanted_labels].as_matrix()
        y = np.asarray(y)
    
        if labels_to_sign:
            y = 2 * y - 1

    return X,y

def convert_matrix_tf_format(X):
    """Take a matrix X and convert it to the appropriate 64-bit numpy
    array that can be used with tensorflow.

    Args:
        X: a matrix of data.
    
    Returns: A 64-bit numpy array of the same data.
    """ 
    X = np.asarray(X)
    X = X.astype(np.float64)
    return X

def normalize_columns(df, wanted_feats, normalization='z_score'):
    """Normalizes the data in columns of a pandas dataframe using one of
    several methods.

    Args:
        df: A pandas dataframe containing data. 
        wanted_feats: A list of string names of columns storing the actual data.
        normalization: A string describing how to normalize the data. Can be 
            'z_score', 'between_0_and_1', or None
    
    Returns: the pandas dataframe with data modified so that it is normalized. 
    """
    print "Normalizing data using", normalization, "method"
    train_df = df[df['dataset']=='Train']
    for feat in wanted_feats:
        feat_list = train_df[feat].dropna().tolist()
        if normalization == 'z_score':
            train_mean = np.mean(feat_list)
            train_std = np.std(feat_list)
            norm_func = lambda x: (x - train_mean) / train_std
        else:
            minx = min(feat_list)
            maxx = max(feat_list)
            norm_func = lambda x: (x - minx) / (maxx - minx)
        df[feat] = df[feat].apply(norm_func)
    return df

def find_null_columns(df, features):
    """Locates columns in a pandas dataframe that have no values. 

    Args:
        df: A pandas dataframe containing data. 
        wanted_feats: A list of string names of columns storing the actual data.
    
    Returns: A list of string names of the null columns.
    """
    df_len = len(df)
    bad_feats = []
    for feat in features:
        null_len = len(df[df[feat].isnull()])
        if df_len == null_len:
            bad_feats.append(feat)
    return bad_feats

def remove_null_cols(df, features):
    """Checks whether any of the columns in a pandas dataframe is completely null
    in any of the datasets (Train, Test, or Val). If so, it will remove it.
    
    Args:
        df: A pandas dataframe containing data. 
        features: A list of string names of columns storing the actual data.
    
    Returns: The pandas dataframe with the bad columns removed, as well as a new 
        list of the remaining columns/features which store the data."""
    train_df = df[df['dataset']=='Train']
    test_df = df[df['dataset']=='Test']
    val_df = df[df['dataset']=='Val']

    null_cols = find_null_columns(train_df,features)
    null_cols_test= find_null_columns(test_df,features)
    null_cols_val = find_null_columns(val_df,features)

    if len(null_cols) > 0 or len(null_cols_test) > 0 or len(null_cols_val) > 0:
        for feat in null_cols_test:
            if feat not in null_cols:
                null_cols.append(feat)
        for feat in null_cols_val:
            if feat not in null_cols:
                null_cols.append(feat)
        print "Found", len(null_cols), "columns that were completely null. Removing", null_cols

        df = dropCols(df,null_cols)
        for col in null_cols:
            features.remove(col)
    return df, features

def assign_cv_fold(row, num_folds=NUM_CROSS_VAL_FOLDS):
    """Randomly ssigns a cross-validation fold number to one row of a pandas dataframe.

    Args:
        row: A row of a pandas dataframe, containing the column 'dataset'
        num_folds: The total number of cross-validation folds.
    
    Returns: The cross-validation fold number. 
    """  
    if row['dataset'] == 'Test':
        return -1
    else:
        return np.random.randint(0,5)

""" Code for extracting data modalities """
def get_modality_dict(wanted_feats, subdivide_phys=False):
    """Extracts a dictionary of the different data sources used to extract features, 
    and the start index of each set of features.

    This is accomplished by looking at the prefix of each feature. I.e. 
    'location_log_likelihood' is the log_likelihood feature from the 'location' modality.

    Args:
        wanted_feats: A list of string names of the features.
        subdivide_phys: If True, will subdivide features starting with 'phys' based on the next
         prefix. I.e. 'phys_10-17H' would be a different modality than 'phys_17-0H'.

    Returns: A dictionary mapping each modality prefix to the index of the first feature 
        belonging to that modality. 
    """
	modalities = list(set([get_feat_prefix(x, subdivide_phys=subdivide_phys) for x in wanted_feats]))
	mod_dict = dict()
	for modality in modalities:
		mod_dict[modality] = get_start_index(wanted_feats, modality)
	return mod_dict

def get_start_index(wanted_feats, modality):
    """Gets the index of the first feature in the list that belongs to the modality. 

    Args:
        wanted_feats: A list of string names of the features.
        modality: A string prefix of a given modality, e.g. 'location', or 'phys'
    
    Returns: An integer start index. 
    """
	for i,s in enumerate(wanted_feats):
		if modality[0:4] == 'phys' and 'H' in modality and modality != 'physTemp':
			if modality + ':' in s:
				return i
		else:
			if modality + '_' in s:
				return i

def get_feat_prefix(feat_name, subdivide_phys=False):
    """Gets the prefix of a given string based on the first location of 
    an underscore. 

    Args:
        feat_name: The string name of the feature
        subdivide_phys: If True, will subdivide features starting with 'phys' based 
            on the next prefix. I.e. 'phys_10-17H' would be a different modality 
            than 'phys_17-0H'.
    
    Returns: The feature prefix not including the underscore.
    """
	idx = feat_name.find('_')
	prefix = feat_name[0:idx]
	if not subdivide_phys or prefix != 'phys':
		return prefix
	else:
		idx = feat_name.find(':')
		return feat_name[0:idx]

def get_modality_names_indices(modality_dict):
    """Given a modality dict, sorts it according to the start indices (values)
    and returns two correspondingly sorted lists of the modality names and 
    their start indices.

    Args:
        modality_dict: A dictionary containing the names of modalities (e.g. 'phys')
            as keys, and the index of the first feature of that modality as values.
    
    Returns: two sorted lists of the modality names and their start indices.
    """
    sorted_tuples = sorted(modality_dict.items(), key=operator.itemgetter(1))
    names = [n for (n,i) in sorted_tuples] 
    indices = [i for (n,i) in sorted_tuples]
    return names,indices

def fill_gaps_in_modalities(df, fill_value, suppress_output=False, verbose=True):
    """Goes through the rows of data in a dataframe, looking for cases when a 
    single row is missing more than 80% of the data from one source/modality/feature 
    prefix. If this is detected, the entire modality in that row is filled with the special
    fill value.

    Args:
        df: A pandas dataframe containing data as well as addtional
            columns following the usual conventions described at the top of
            this file. WARNING: this function will break if there are additional 
            columns not covered by the conventions or not prefixed with 'logistics'.
        fill_value: The value with which to fill the missing modalities. 
        suppress_output: If True, no output statements describing progress will
            be printed.
        verbose: If True, will print every time it filled a missing modality, in what
            row, and how much data was missing. 
    
    Returns:
        The pandas dataframe with missing modalities filled.
    """
    num_rows_filled = 0
    for row in df.index.values:
        current_prefix = get_feat_prefix(df.columns.values[2], subdivide_phys=True)
        prefix_start_index = 2
        num_nan_this_prefix = 0
        num_this_prefix = 0
        for i_f, feat in enumerate(df.columns.values):
            if ('user_id' in feat or 'timestamp' in feat or 'logistics' in feat or 'label' in feat 
                or 'dataset' in feat or 'Label' in feat):
                    continue
                    
            prefix = get_feat_prefix(feat, subdivide_phys=True)
                
            if prefix != current_prefix:
                percent_nan = float(num_nan_this_prefix) / num_this_prefix

                if percent_nan > 0.8:
                    if verbose:
                        print "Filling", current_prefix, 
                        print "(index", prefix_start_index, "to", i_f, ") for person",
                        print row, "because it was", percent_nan*100.0, "% empty"
                    for f in range(prefix_start_index, i_f):
                        df.ix[row,f] = fill_value
                    num_rows_filled += 1

                num_nan_this_prefix = 0
                num_this_prefix = 0
                prefix_start_index = i_f

            if pd.isnull(df[feat][row]):
                num_nan_this_prefix += 1
            num_this_prefix += 1

            current_prefix = prefix

    if not suppress_output:
        print "Filled gaps in", num_rows_filled, "rows with", fill_value
    return df

def count_gaps_in_modalities(df):
    """Duplicated code (oops) that simply counts the number of missing modalities,
    and keeps track of which modalities tend to go missing at the same time.

    Args:
        df: a pandas dataframe containing data as well as addtional
            columns following the usual conventions described at the top of
            this file. WARNING: this function will break if there are additional 
            columns not covered by the conventions or not prefixed with 'logistics'.
    
    Returns:
        A dictionary with keys describing a collection of modalities that went missing 
        simultaneously in at least one row, and values giving the number of rows in 
        which this pattern was observed.
    """
    missing_dict = {}
    
    for row in df.index.values:
        current_prefix = get_feat_prefix(df.columns.values[2], subdivide_phys=True)
        prefix_start_index = 2
        num_nan_this_prefix = 0
        num_this_prefix = 0

        missing_list = ''

        for i_f, feat in enumerate(df.columns.values):
            if ('user_id' in feat or 'timestamp' in feat or 'logistics' in feat or 'label' in feat 
                or 'dataset' in feat or 'Label' in feat):
                    continue
                    
            prefix = get_feat_prefix(feat, subdivide_phys=True)
                
            if prefix != current_prefix:
                percent_nan = float(num_nan_this_prefix) / num_this_prefix

                if percent_nan > 0.8:
                    if missing_list == '':
                        missing_list = current_prefix
                    else:
                        missing_list += ', ' + current_prefix

                num_nan_this_prefix = 0
                num_this_prefix = 0
                prefix_start_index = i_f

            if pd.isnull(df[feat][row]):
                num_nan_this_prefix += 1
            num_this_prefix += 1

            current_prefix = prefix
        
        if missing_list == '':
            missing_list = 'None'
        if missing_list in missing_dict.keys():
            missing_dict[missing_list] += 1
        else:
            missing_dict[missing_list] = 1
        
        if row % 20 == 0:
            print "On row", row, "missing dict is:"
            print missing_dict
    
    return missing_dict