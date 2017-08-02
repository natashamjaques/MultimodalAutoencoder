import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def load_raw_data(datasets_path, mmae_filename):
	"""Loads the training data from a .csv file. Decides which features
	to load based on the file conventions described in data_funcs.py

	Args:
		datasets_path: A string path to the file location.
		mmae_filename: The name of the actual file.
	"""
    df = pd.DataFrame.from_csv(datasets_path+mmae_filename)
    
    feat_cols = [x for x in df.columns.values if 'user_id' not in x and 
                                                     'timestamp' not in x and
                                                     'label' not in x and 
                                                     'Label' not in x and
                                                     'dataset' not in x and 
                                                     'logistics' not in x and
                                                     'ppt_id' not in x]

    logistic_cols = [c for c in df.columns.values if c not in feat_cols]
    
    X_train = df[df["dataset"]=="Train"][feat_cols].as_matrix()
    X_all = df[feat_cols].as_matrix()

    return df, X_train, X_all, logistic_cols 

def transform_PCA(X_train,X_all,n_components=100):
	"""Given some training data, performs a Principle Components Analysis (PCA)
	and modifies the rest of the data based on the learned PCA.

	Args:
		X_train: A matrix containing training data
		X_all: A matrix containing all the data
		n_components: The number of components to use in the PCA

	Returns:
		The transformed data and the PCA object
	"""
	pca = PCA(n_components=n_components)
	pca.fit(X_train)
	print("Total explained variance:", sum(pca.explained_variance_ratio_))

	return pca.transform(X_all),pca

def transform_select_K_best(X_train,Y_train, X_all, K=100):
	"""Selects the best K features given the training data.

	Args:
		X_train: A matrix containing training data
		Y_train: Classification labels for the training data
		X_all: A matrix containing all the data
		K: The number of features to select
	"""
	skb = SelectKBest(f_classif,K)
	skb.fit(X_train,Y_train)

	return skb.transform(X_all)


def create_transformed_dataset(datasets_path, filename, transform_type,num_features,label=None):
	"""Loads a file, performs a form of feature selection on the file's data, and saves the 
	transformed version of the file to the same location.

	Args:
		datasets_path: A string path to the file location.
		filename: The name of the actual file.
		transform_type: The type of feature reduction to perform. Can be either 'pca' or 'skb'
		num_feature: The number of features/components to keep.
		label: The string name of the classification label - needed if using 'skb'
	"""
	df, X_train, X_all, logistic_cols = load_raw_data(datasets_path, mmae_filename, True)

	if transform_type=="pca":
		transformed_X, model = transform_PCA(X_train,X_all,num_features)
		transform_prefix = "pca_"

		return model
	elif transform_type=="skb":
		assert label is not None, "label parameter required for skb transformation"
		assert label in logistic_cols, "label must be in the dataframe"

		Y_train = df[label][df["dataset"]=="Train"].as_matrix()
		idx = np.isnan(Y_train)

		X_train = X_train[~idx,:]
		Y_train = Y_train[~idx]

		transformed_X = transform_select_K_best(X_train,Y_train, X_all, num_features)
		transform_prefix = "skb_"+label+"_"

	transformed_cols = logistic_cols + ["{0}_dim{1}".format(transform_type,i) for i in range(num_features)]

	transformed_df = pd.DataFrame(np.hstack([df[logistic_cols].as_matrix(), transformed_X]),columns = transformed_cols)
	
	transformed_df.to_csv(datasets_path+transform_prefix+mmae_filename)


