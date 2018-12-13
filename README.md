# MultimodalAutoencoder
Code supporting the following paper: <br />

 Jaques N., Taylor S., Sano A., Picard R.,<strong>"Multimodal Autoencoder: A Deep Learning Approach to Filling In Missing Sensor Data and Enabling Better Mood Prediction", </strong> International Conference on Affective Computing and Intelligent Interaction, October 2017, Texas, USA. <a href="https://affect.media.mit.edu/pdfs/17.Jaques_autoencoder_ACII.pdf">pdf</a> <br/>

## Description

The MultimodalAutoencoder (MMAE) is designed to deal with data in which large, contiguous blocks of features go missing at once; specifically all the features  extracted from the same data source or *modality*. For example, all the features extracted from a skin conductance sensor may go missing if the sensor  experiences a technical issue when recording data for a particular sample. By randomly blocking out different modalities from the training data and learning to reconstruct them, the MMAE is able to reconstruct real missing data. 

## Files, file names, and folders

 * multimodal_autoencoder.py - The main code for the MMAE model.
 * run_jobs.py - Code for running jobs to train the models on a server and emailing you when they finish.
 * generic_wrapper.py - Generic classes that can be inherited to build wrappers that will perform grid searches over hyperparameter settings for different models.
 * Any *wrapper* file - An inherited version of the generic wrapper for a specific model. 
 * data_funcs.py - Functions dealing with loading data from a file, organizing it into cross validation folds, normalizing it, filling missing data, etc.
 * feature_selection.py - Implements two feature selection methods.
 * comparison_algorithms/ - Code for four basic ML classifiers to compare against: SVM, Random Forest, Logistic Regression, and a basic Neural Network.

 ## Dependencies

 * tensorflow
 * numpy
 * pandas
 * matplotlib
 * sklearn





