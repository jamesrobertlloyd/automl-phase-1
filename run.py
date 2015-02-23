from __future__ import division

#!/usr/bin/env python

#############################
# ChaLearn AutoML challenge #
#############################

# Usage: python run.py input_dir output_dir

# This sample code can be used either 
# - to submit RESULTS deposited in the res/ subdirectory or
# - as a template for CODE submission.
#
# The input directory input_dir contains 5 subdirectories named by dataset,
# including:
# 	dataname/dataname_feat.type          -- the feature type "Numerical", "Binary", or "Categorical" (Note: if this file is abscent, get the feature type from the dataname.info file)
# 	dataname/dataname_public.info        -- parameters of the data and task, including metric and time_budget
# 	dataname/dataname_test.data          -- training, validation and test data (solutions/target values are given for training data only)
# 	dataname/dataname_train.data
# 	dataname/dataname_train.solution
# 	dataname/dataname_valid.data
#
# The output directory will receive the predicted values (no subdirectories):
# 	dataname_test_000.predict            -- Provide predictions at regular intervals to make sure you get some results even if the program crashes
# 	dataname_test_001.predict
# 	dataname_test_002.predict
# 	...
# 	dataname_valid_000.predict
# 	dataname_valid_001.predict
# 	dataname_valid_002.predict
# 	...
# 
# Result submission:
# =================
# Search for @RESULT to locate that part of the code.
# ** Always keep this code. **
# If the subdirectory res/ contains result files (predicted values)
# the code just copies them to the output and does not train/test models.
# If no results are found, a model is trained and tested (see code submission).
#
# Code submission:
# ===============
# Search for @CODE to locate that part of the code.
# ** You may keep or modify this template or subtitute your own code. **
# The program saves predictions regularly. This way the program produces
# at least some results if it dies (or is terminated) prematurely. 
# This also allows us to plot learning curves. The last result is used by the
# scoring program.
# We implemented 2 classes:
# 1) DATA LOADING:
#    ------------
# Use/modify 
#                  D = DataManager(basename, input_dir, ...) 
# to load and preprocess data.
#     Missing values --
#       Our default method for replacing missing values is trivial: they are replaced by 0.
#       We also add extra indicator features where missing values occurred. This doubles the number of features.
#     Categorical variables --
#       The location of potential Categorical variable is indicated in D.feat_type.
#       NOTHING special is done about them in this sample code. 
#     Feature selection --
#       We only implemented an ad hoc feature selection filter efficient for the 
#       dorothea dataset to show that performance improves significantly 
#       with that filter. It takes effect only for binary classification problems with sparse
#       matrices as input and unbalanced classes.
# 2) LEARNING MACHINE:
#    ----------------
# Use/modify 
#                 M = MyAutoML(D.info, ...) 
# to create a model.
#     Number of base estimators --
#       Our models are ensembles. Adding more estimators may improve their accuracy.
#       Use M.model.n_estimators = num
#     Training --
#       M.fit(D.data['X_train'], D.data['Y_train'])
#       Fit the parameters and hyper-parameters (all inclusive!)
#       What we implemented hard-codes hyper-parameters, you probably want to
#       optimize them. Also, we made a somewhat arbitrary choice of models in
#       for the various types of data, just to give some baseline results.
#       You probably want to do better model selection and/or add your own models.
#     Testing --
#       Y_valid = M.predict(D.data['X_valid'])
#       Y_test = M.predict(D.data['X_test']) 
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS". 
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS. 
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL, 
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS, 
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE. 
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013

# =========================== BEGIN USER OPTIONS ==============================
# Verbose mode: 
##############
# Recommended to keep verbose = True: shows various progression messages
verbose = True # outputs messages to stdout and stderr for debug purposes

# Debug level:
############## 
# 0: run the code normally, using the time budget of the tasks
# 1: run the code normally, but limits the time to max_time
# 2: run everything, but do not train, generate random outputs in max_time
# 3: stop before the loop on datasets
# 4: just list the directories and program version
debug_mode = 0

# Time budget
#############
# Maximum time of training in seconds PER DATASET (there are 5 datasets). 
# The code should keep track of time spent and NOT exceed the time limit 
# in the dataset "info" file, stored in D.info['time_budget'], see code below.
# If debug >=1, you can decrease the maximum time (in sec) with this variable:
max_time = 90 

# Maximum number of cycles
##########################
# Your training algorithm may be fast, so you may want to limit anyways the 
# number of points on your learning curve (this is on a log scale, so each 
# point uses twice as many time than the previous one.)
max_cycle = 50

# ZIP your code
###############
# You can create a code submission archive, ready to submit, with zipme = True.
# This is meant to be used on your LOCAL server.
import datetime
zipme = True  # use this flag to enable zipping of your code submission
the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
submission_filename = 'phase0_submission_' + the_date

# I/O defaults
##############
# Use default location for the input and output data:
# If no arguments to run.py are provided, this is where the data will be found
# and the results written to. Change the root_dir to your local directory.
import os
# import sys
root_dir = os.path.dirname(__file__)
# root_dir = "/Users/isabelleguyon/Documents/Projects/Codalab/AutoMLcompetition/StartingKit/"
# default_input_dir = os.path.join(root_dir, '..', 'data', 'dsss_bin_class_fold_01')
default_input_dir = os.path.join(root_dir, '..', 'data', 'phase_0')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'cv_rf_msl')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dss_bin_class_fold_01', 'cv_rf_gbm')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'automl')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'automl_rf')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf_gbm')
# default_output_dir = os.path.join(root_dir, '..', 'predictions', 'dsss_bin_class_fold_01', 'ft_cv_rf_gbm_v2')
default_output_dir = os.path.join(root_dir, 'res')

# =========================== END USER OPTIONS ================================

# Version of the sample code
# Change in 1.1: time is measured by time.time(), not time.clock(): we keep track of wall time
# Change in 1,2: James Robert Lloyd taking control of code
# Change in 1.3: Prediction code run in separate process to keep an eye of memory and time
version = 1.3

# General purpose functions
# import os
from sys import argv, path
import numpy as np
import time
overall_start = time.time()

# Our directories
# Note: On cadalab, there is an extra sub-directory called "program"
running_on_codalab = False
run_dir = os.path.abspath(".")
codalab_run_dir = os.path.join(run_dir, "program")
if os.path.isdir(codalab_run_dir): 
    run_dir=codalab_run_dir
    running_on_codalab = True
    print "Running on Codalab!"
lib_dir = os.path.join(run_dir, "lib")
res_dir = os.path.join(run_dir, "res")

# Our libraries  
path.append (run_dir)
path.append (lib_dir)
from automl_lib import data_io                       # general purpose input/output functions
from automl_lib.data_io import vprint           # print only in verbose mode
from automl_lib.data_manager import DataManager # load/save data and get info about them
from automl_lib.models import MyAutoML          # example model from scikit learn

if debug_mode >= 4 or running_on_codalab: # Show library version and directory structure
    data_io.show_version()
    data_io.show_dir(run_dir)

from multiprocessing import Process
import psutil

import util

import automl
import traceback


# Define function to be called to start process
def run_automl(input_dir, output_dir, data_name, time_budget):
    print('input_dir = "%s"' % input_dir)
    print('output_dir = "%s"' % output_dir)
    print('data_name = "%s"' % data_name)
    print('time_budget = %s' % time_budget)
    try:
        # automl.data_doubling_rf(input_dir, output_dir, data_name, time_budget, 20)
        # automl.cv_growing_rf(input_dir, output_dir, data_name, time_budget)
        # automl.cv_growing_rf_gbm(input_dir, output_dir, data_name, time_budget)

        # automl.competition_example(input_dir, output_dir, data_name, time_budget)
        # automl.competition_example_only_rf(input_dir, output_dir, data_name, time_budget)
        # automl.freeze_thaw_cv_rf(input_dir, output_dir, data_name, time_budget)
        # automl.freeze_thaw_cv_rf_gbm(input_dir, output_dir, data_name, time_budget, compute_quantum=10)
        automl.automl_phase_0(input_dir, output_dir, data_name, time_budget)
    except:
        traceback.print_exc()

# =========================== BEGIN PROGRAM ================================

if __name__=="__main__" and debug_mode<4:	
    #### Check whether everything went well (no time exceeded)
    execution_success = True
    
    #### INPUT/OUTPUT: Get input and output directory names
    if len(argv)==1: # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
    else:
        input_dir = argv[1]
        output_dir = os.path.abspath(argv[2]); 
    # Move old results and create a new output directory 
    data_io.mvdir(output_dir, output_dir+'_'+the_date) 
    data_io.mkdir(output_dir) 
    
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(input_dir)
    
    #### DEBUG MODE: Show dataset list and STOP
    if debug_mode>=3:
        data_io.show_io(input_dir, output_dir)
        print('\n****** Sample code version ' + str(version) + ' ******\n\n' + '========== DATASETS ==========\n')        	
        data_io.write_list(datanames)      
        datanames = [] # Do not proceed with learning and testing
        
    # ==================== @RESULT SUBMISSION (KEEP THIS) =====================
    # Always keep this code to enable result submission of pre-calculated results
    # deposited in the res/ subdirectory.
    if len(datanames)>0:
        vprint( verbose,  "************************************************************************")
        vprint( verbose,  "****** Attempting to copy files (from res/) for RESULT submission ******")
        vprint( verbose,  "************************************************************************")
        OK = data_io.copy_results(datanames, res_dir, output_dir, verbose) # DO NOT REMOVE!
        if OK: 
            vprint( verbose,  "[+] Success")
            datanames = [] # Do not proceed with learning and testing
        else:
            vprint( verbose, "======== Some missing results on current datasets!")
            vprint( verbose, "======== Proceeding to train/test:\n")
    # =================== End @RESULT SUBMISSION (KEEP THIS) ==================

    if zipme:
        vprint( verbose,  "========= Zipping this directory to prepare for submit ==============")
        data_io.zipdir(submission_filename + '.zip', ".")

    # ================ @CODE SUBMISSION (SUBTITUTE YOUR CODE) =================

    for basename in datanames:
        print('Processing dataset : ' + basename)
        # Keep track of time
        start_time = time.time()
        # Write a file to record start time
        open(os.path.join(output_dir, basename + '.firstpost'), 'wb').close()
        print('\nStarting\n')
        # Read time budget
        with open(os.path.join(input_dir, basename, basename + '_public.info'), 'r') as info_file:
            for line in info_file:
                if line.startswith('time_budget'):
                    time_budget = int(line.split('=')[-1])
        print('Time budget = %ds' % time_budget)
        # Start separate process to analyse file
        p = Process(target=run_automl, args=(input_dir, output_dir, basename, time_budget - (time.time() - start_time)))
        p.start()
        # Monitor the process, checking to see if it is complete or using too much memory
        while True:
            time.sleep(1)
            if p.is_alive():
                try:
                    process_mem = util.resident_memory_usage(p.pid)
                except:
                    print('Cannot get memory of process %d :(' % p.pid)
                else:
                    system_mem = psutil.virtual_memory().total
                    prop_mem = process_mem / system_mem
                    if prop_mem > 0.9:
                        print('Memory usage of process %d too damn high' % p.pid)
                        p.terminate()
                        p.join()
                        print('Process %d terminated' % p.pid)
                        break
                    print('Proportion of memory used by process %d : %02.1f%%' %
                          (p.pid, 100 * (process_mem / system_mem)))
                if (time.time() - start_time) > (time_budget - 10):
                    print('Time limit approaching - terminating')
                    p.terminate()
                    p.join()
                    print('Process %d terminated' % p.pid)
                    break
                else:
                    print('Remaining time budget = %s' % (time_budget - (time.time() - start_time)))
            else:
                print('Process terminated of its own accord')
                break
        print('\nFinished\n')

              
    if running_on_codalab: 
        if execution_success:
            exit(0)
        else:
            exit(1)