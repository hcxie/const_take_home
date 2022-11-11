"""
helper_fucntions.py

Authors:
    Haochen Xie jbdx6307@gmail.com

Description:
    functions DengAI compititon. Wrapper functions to prepeare data, build model and predict on test set

Created:
    11/8/2022

"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

def create_X_Y(train_data, train_label, test_data, data_pipeline, city):
    """
    - load training data, test data, test_data
    - filter by city name
    - Apply data pipeline to create modeing dataset
    """
    
    # prepare train features
    train_data = train_data.loc[train_data["city"] == city,:].copy()
    train_data.reset_index(drop = True, inplace = True)
    X = train_data.drop(columns = ["city"])
    
    # prepare train labels
    Y = train_label.loc[train_label["city"] == city,:].copy()
    Y.reset_index(drop = True, inplace = True)
    Y = Y["total_cases"]
    
    # prepare test features
    test_data = test_data.loc[test_data["city"] == city,:].copy()
    test_data.reset_index(drop = True, inplace = True)
    X_test = test_data.drop(columns = ["city"])
    
    # Apply pipelines
    X_train = data_pipeline.fit_transform(X, Y)
    Y_train = Y.to_numpy()
    # Apply pipelines
    X_test = data_pipeline.transform(X_test)
    
    return X_train, Y_train, X_test


def single_reg_model_fit_trans(input_path, 
                               sj_data_pipeline,
                               iq_data_pipeline,
                               reg_model_pipeline_sj,
                               reg_model_pipeline_iq,
                               reg_parameters,
                               model_scoring,
                               output_name = None):
    """
    - wrapper function to process data, select the best model based on cv score and prediction on test set
    - input:
        input_path - str, input location
        sj_data_pipeline - obj, data pipeline for city sj
        iq_data_pipeline - obj, data pipeline for city iq
        reg_model_pipeline_sj - obj, model pipeline for city sj
        reg_model_pipeline_iq - obj, model pipeline for city iq
        reg_parameters - list[dict], a list of different model and hyperparameters
        model_scoring - str, cv scoring name.
        output_name - str, the name of test prediction file
    - output:
        pred_train - prediction on train
        pred_test - prediction on test
        overall_cv_score - best model's cv score
    
    """
    # load the data
    train_data = pd.read_csv(input_path + "dengue_features_train.csv")
    train_label = pd.read_csv(input_path + "dengue_labels_train.csv")
    test_data = pd.read_csv(input_path + "dengue_features_test.csv")

    # apply data pipeline and create data
    X_train_sj, Y_train_sj, X_test_sj = create_X_Y(train_data, train_label, test_data, sj_data_pipeline, "sj")
    X_train_iq, Y_train_iq, X_test_iq = create_X_Y(train_data, train_label, test_data, iq_data_pipeline, "iq")

    # select the best model and parameters based on CV scores for city SJ
    reg_model_sj = GridSearchCV(estimator = reg_model_pipeline_sj, 
                          param_grid = reg_parameters, 
                          scoring = model_scoring, 
                          cv = 5, 
                          verbose = 1,
                          n_jobs = -1,
                          return_train_score = True,
                          error_score = Y_train_sj.mean())
    reg_model_sj.fit(X_train_sj, Y_train_sj)

    # select the best model and parameters based on CV scores for city IQ
    reg_model_iq = GridSearchCV(estimator = reg_model_pipeline_iq, 
                          param_grid = reg_parameters, 
                          scoring = model_scoring, 
                          cv = 5, 
                          verbose = 1,
                          n_jobs = -1,
                          return_train_score = True,
                          error_score = Y_train_iq.mean())
    reg_model_iq.fit(X_train_iq, Y_train_iq)

    # predict on SJ and IQ train set
    pred_train_sj = reg_model_sj.best_estimator_.predict(X_train_sj)
    pred_train_iq = reg_model_iq.best_estimator_.predict(X_train_iq)
    # predict on SJ and IQ test set.
    pred_test_sj = reg_model_sj.best_estimator_.predict(X_test_sj)
    pred_test_iq = reg_model_iq.best_estimator_.predict(X_test_iq)
    
    # combine SJ and IQ prediction together
    pred_train = np.concatenate([pred_train_sj, pred_train_iq])
    pred_test = np.concatenate([pred_test_sj, pred_test_iq])
    
    # MAE on train data
    train_score = mean_absolute_error(train_label["total_cases"],pred_train)
    
    print("Best Train MAE: {}".format(train_score))
    print("Mean SJ CV score {}".format(reg_model_sj.best_score_))
    print("Mean IQ CV score {}".format(reg_model_iq.best_score_))
    print("Best Selected Parameters for SJ reg model: ")
    print(list(reg_model_sj.best_params_.items())[1:])
    print("Best Selected Parameters for IQ reg model: ")
    print(list(reg_model_iq.best_params_.items())[1:])
    
    # plot the prediction
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))
    ax.plot(train_label["total_cases"])
    ax.plot(pred_train)
    
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_test)
    
    # save the prediction to the submission_format
    if output_name is not None:
        submission = pd.read_csv(input_path + "submission_format.csv")
        submission["total_cases"] = pred_test.astype(int)
        submission.to_csv(input_path + output_name, index = False)
    
    # based on the size of the training data, estimate the overall cv score of this model
    # overall_cv_score = reg_model_sj.best_score_*9/14 + reg_model_iq.best_score_*5/14
    
    return (pred_train_sj, 
            pred_train_iq, 
            pred_test_sj, 
            pred_test_iq, 
            reg_model_sj.best_score_, 
            reg_model_iq.best_score_)



def stacking_reg_model_fit_trans(input_path, 
                                sj_data_pipeline,
                                iq_data_pipeline,
                                reg_model_pipeline_sj_1,
                                reg_model_pipeline_iq_1,
                                reg_parameters_1,
                                reg_model_pipeline_sj_2,
                                reg_model_pipeline_iq_2,
                                reg_parameters_2,
                                reg_model_pipeline_sj_3,
                                reg_model_pipeline_iq_3,
                                reg_parameters_3,
                                reg_model_pipeline_sj_4,
                                reg_model_pipeline_iq_4,
                                reg_parameters_4,
                                model_scoring,
                                output_name = None):
    """
    - wrapper function create stacking results. Each model pipeline contains different
      types of models
    - input:
        input_path - str, input location
        sj_data_pipeline - obj, data pipeline for city sj
        iq_data_pipeline - obj, data pipeline for city iq
        reg_model_pipeline_sj_1 - obj, model pipeline 1 for city sj
        reg_model_pipeline_iq_1 - obj, model pipeline 1 for city iq
        reg_parameters_1 - list[dict], a list of model 1 and its hyperparameters
        reg_model_pipeline_sj_2 - obj, model pipeline 2 for city sj
        reg_model_pipeline_iq_2 - obj, model pipeline 2 for city iq
        reg_parameters_2 - list[dict], a list of model 2 and its hyperparameters
        reg_model_pipeline_sj_3 - obj, model pipeline 3 for city sj
        reg_model_pipeline_iq_3 - obj, model pipeline 3 for city iq
        reg_parameters_3 - list[dict], a list of model 3 and its hyperparameters
        reg_model_pipeline_sj_4 - obj, model pipeline 4 for city sj
        reg_model_pipeline_iq_4 - obj, model pipeline 4 for city iq
        reg_parameters_4 - list[dict], a list of model 4 and its hyperparameters
        model_scoring - str, cv scoring name.
        output_name - str, the name of test prediction file
    - output:
    
    """
    
    # get the true label on training set
    train_label = pd.read_csv(input_path + "dengue_labels_train.csv")
    Y = train_label["total_cases"]

    # apply the wrapper function to build model 1
    (pred_train_sj_1, 
     pred_train_iq_1, 
     pred_test_sj_1, 
     pred_test_iq_1, 
     cv_score_sj_1, 
     cv_score_iq_1) = single_reg_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                reg_model_pipeline_sj_1,
                                                reg_model_pipeline_iq_1,
                                                reg_parameters_1,
                                                model_scoring,
                                                None)
    # apply the wrapper function to build model 2
    (pred_train_sj_2, 
     pred_train_iq_2, 
     pred_test_sj_2, 
     pred_test_iq_2, 
     cv_score_sj_2, 
     cv_score_iq_2) = single_reg_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                reg_model_pipeline_sj_2,
                                                reg_model_pipeline_iq_2,
                                                reg_parameters_2,
                                                model_scoring,
                                                None)
    # apply the wrapper function to build model 3                                                            
    (pred_train_sj_3, 
     pred_train_iq_3, 
     pred_test_sj_3, 
     pred_test_iq_3, 
     cv_score_sj_3, 
     cv_score_iq_3) = single_reg_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                reg_model_pipeline_sj_3,
                                                reg_model_pipeline_iq_3,
                                                reg_parameters_3,
                                                model_scoring,
                                                None)
    # apply the wrapper function to build model 4
    (pred_train_sj_4, 
     pred_train_iq_4, 
     pred_test_sj_4, 
     pred_test_iq_4, 
     cv_score_sj_4, 
     cv_score_iq_4) = single_reg_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                reg_model_pipeline_sj_4,
                                                reg_model_pipeline_iq_4,
                                                reg_parameters_4,
                                                model_scoring,
                                                None)
    # based on the cv score, estimate the weight of prediction of each model
    weight_sj = np.array([cv_score_sj_1, 
                          cv_score_sj_2, 
                          cv_score_sj_3, 
                          cv_score_sj_4])

    weight_sj = weight_sj/weight_sj.sum()
    
    # stacking the results together based on weight
    pred_train_sj = (pred_train_sj_1 * weight_sj[0] 
                    + pred_train_sj_2 * weight_sj[1] 
                    + pred_train_sj_3 * weight_sj[2] 
                    + pred_train_sj_4 * weight_sj[3])

    pred_test_sj = (pred_test_sj_1 * weight_sj[0] 
                    + pred_test_sj_2 * weight_sj[1] 
                    + pred_test_sj_3 * weight_sj[2] 
                    + pred_test_sj_4 * weight_sj[3])
    

    # based on the cv score, estimate the weight of prediction of each model
    weight_iq = np.array([cv_score_iq_1, 
                          cv_score_iq_2, 
                          cv_score_iq_3, 
                          cv_score_iq_4])

    weight_iq = weight_iq/weight_iq.sum()
    
    # stacking the results together based on weight
    pred_train_iq = (pred_train_iq_1 * weight_iq[0] 
                    + pred_train_iq_2 * weight_iq[1] 
                    + pred_train_iq_3 * weight_iq[2] 
                    + pred_train_iq_4 * weight_iq[3])

    pred_test_iq = (pred_test_iq_1 * weight_iq[0] 
                    + pred_test_iq_2 * weight_iq[1] 
                    + pred_test_iq_3 * weight_iq[2] 
                    + pred_test_iq_4 * weight_iq[3])


    pred_train = np.concatenate([pred_train_sj, pred_train_iq])
    pred_test = np.concatenate([pred_test_sj, pred_test_iq])

    # calculate mae
    train_score = mean_absolute_error(Y, pred_train)
    
    print("Best Train MAE: {}".format(train_score))
    
    # plot prediction
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))
    ax.plot(Y)
    ax.plot(pred_train)
    
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_test)
    
    # save the prediction to submission_format.csv
    if output_name is not None:
        submission = pd.read_csv(input_path + "submission_format.csv")
        submission["total_cases"] = pred_test.astype(int)
        submission.to_csv(input_path + output_name, index = False)
    
    return



def single_clf_model_fit_trans(input_path, 
                               sj_data_pipeline,
                               iq_data_pipeline,
                               clf_model_pipeline_sj,
                               clf_model_pipeline_iq,
                               clf_parameters,
                               model_scoring,
                               Y_threshold_p,
                               output_name = None
                               ):
    """
    - wrapper function to process data, select the best perpensity model based on cv score and prediction on test set
    - input:
        input_path - str, input location
        sj_data_pipeline - obj, data pipeline for city sj
        iq_data_pipeline - obj, data pipeline for city iq
        clf_model_pipeline_sj - obj, perpensity model pipeline for city sj
        clf_model_pipeline_iq - obj, perpensity model pipeline for city iq
        clf_parameters - list[dict], a list of different model and hyperparameters
        model_scoring - str, cv scoring name.
        Y_threshold_p - int, the percentile we used to determine the outlier (positive label)
        output_name - str, the name of test prediction file
    - output:
        pred_train - prediction probability on train data
        pred_train_cls - prediction class on train data
        pred_test - prediction probability on test data
        pred_test_cls - prediction probability on test data
        overall_cv_score - overall cv score
    
    """

    # load the data     
    train_data = pd.read_csv(input_path + "dengue_features_train.csv")
    train_label = pd.read_csv(input_path + "dengue_labels_train.csv")
    test_data = pd.read_csv(input_path + "dengue_features_test.csv")

    # apply the data pipeline and build the train data    
    X_train_sj, Y_train_sj, X_test_sj = create_X_Y(train_data, train_label, test_data, sj_data_pipeline, "sj")
    X_train_iq, Y_train_iq, X_test_iq = create_X_Y(train_data, train_label, test_data, iq_data_pipeline, "iq")
    
    # figure out the target value based on the percentile user defined for city sj
    Y_threshold = np.percentile(Y_train_sj, Y_threshold_p)
    # convert the numerical target variable to binary 
    Y_train_sj = (Y_train_sj >= Y_threshold).astype(int)
    
    # figure out the target value based on the percentile user defined for city iq
    Y_threshold = np.percentile(Y_train_iq, Y_threshold_p)
    # convert the numerical target variable to binary 
    Y_train_iq = (Y_train_iq >= Y_threshold).astype(int)

    # Since SJ and IQ data have different data distribution, their numerical threshold will be different

    # build and tune model through CV
    clf_model_sj = GridSearchCV(estimator = clf_model_pipeline_sj, 
                          param_grid = clf_parameters, 
                          scoring = model_scoring, 
                          cv = 5, 
                          verbose = 1,
                          n_jobs = -1,
                          return_train_score = True,
                          error_score = "raise")
    
    clf_model_sj.fit(X_train_sj, Y_train_sj)

    # build and tune model through CV
    clf_model_iq = GridSearchCV(estimator = clf_model_pipeline_iq, 
                          param_grid = clf_parameters, 
                          scoring = model_scoring, 
                          cv = 5, 
                          verbose = 1,
                          n_jobs = -1,
                          return_train_score = True,
                          error_score = "raise")

    clf_model_iq.fit(X_train_iq, Y_train_iq)

    # get the predict proability on postive label (outliers)
    pred_train_sj = clf_model_sj.best_estimator_.predict_proba(X_train_sj)[:,1]
    pred_train_iq = clf_model_iq.best_estimator_.predict_proba(X_train_iq)[:,1]
    
    # get the prediction probaility on positive label for test set
    pred_test_sj = clf_model_sj.best_estimator_.predict_proba(X_test_sj)[:,1]
    pred_test_iq = clf_model_iq.best_estimator_.predict_proba(X_test_iq)[:,1]
    
    # find out the predict probaility threshold that maximize precision score for SJ model
    precision, recall, thresholds = precision_recall_curve(Y_train_sj, pred_train_sj)
    threshold_index = np.argmax(precision)
    if threshold_index > len(thresholds) - 1:
        threshold_index -= 1
    cls_threshold_sj = thresholds[threshold_index]

    # find out the predict probaility threshold that maximize precision score for IQ model
    precision, recall, thresholds = precision_recall_curve(Y_train_iq, pred_train_iq)
    threshold_index = np.argmax(precision)
    if threshold_index > len(thresholds) - 1:
        threshold_index -= 1
    cls_threshold_iq = thresholds[threshold_index]
    
    # combine the predict proability results toegther
    pred_train = np.concatenate([pred_train_sj, pred_train_iq])
    pred_test = np.concatenate([pred_test_sj, pred_test_iq])
    
    # get the true labels
    Y = np.concatenate([Y_train_sj, Y_train_iq])
    
    # calcualte the average precison score on train data
    train_score = average_precision_score(Y, pred_train)
    
    print("Best Train Mean Precision: {}".format(train_score))
    print("Mean SJ CV score {}".format(clf_model_sj.best_score_))
    print("Mean IQ CV score {}".format(clf_model_iq.best_score_))
    print("Best Selected Parameters for SJ reg model: ")
    print(list(clf_model_sj.best_params_.items())[1:])
    print("Max precision threshold: {}".format(cls_threshold_sj))
    print("Best Selected Parameters for IQ reg model: ")
    print(list(clf_model_iq.best_params_.items())[1:])
    print("Max precision threshold: {}".format(cls_threshold_iq))
    
    # get the predict label on train and test set
    pred_train_sj_cls = (pred_train_sj >= cls_threshold_sj).astype(int)
    pred_train_iq_cls = (pred_train_iq >= cls_threshold_iq).astype(int)

    pred_test_sj_cls = (pred_test_sj >= cls_threshold_sj).astype(int)
    pred_test_iq_cls = (pred_test_iq >= cls_threshold_iq).astype(int)

    pred_train_cls = np.concatenate([pred_train_sj_cls, pred_train_iq_cls])
    
    pred_test_cls = np.concatenate([pred_test_sj_cls, pred_test_iq_cls])
    # plot the result
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_train_cls)

    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_test_cls)
    
    # save the prediction to submission_format
    if output_name is not None:
        submission = pd.read_csv(input_path + "submission_format.csv")
        submission["predict_cls"] = pred_test_cls
        submission["predict_prob"] = pred_test
        submission.to_csv(input_path + output_name, index = False)
    # overall cv score
    #overall_cv_score = clf_model_sj.best_score_*9/14 + clf_model_iq.best_score_*5/14
    
    return (pred_train_sj, 
            pred_train_sj_cls, 
            pred_train_iq,
            pred_train_iq_cls,
            pred_test_sj,
            pred_test_sj_cls,
            pred_test_iq,
            pred_test_iq_cls,
            clf_model_sj.best_score_,
            clf_model_iq.best_score_)


def stacking_clf_model_fit_trans(input_path, 
                                sj_data_pipeline,
                                iq_data_pipeline,
                                clf_model_pipeline_sj_1,
                                clf_model_pipeline_iq_1,
                                clf_parameters_1,
                                clf_model_pipeline_sj_2,
                                clf_model_pipeline_iq_2,
                                clf_parameters_2,
                                clf_model_pipeline_sj_3,
                                clf_model_pipeline_iq_3,
                                clf_parameters_3,
                                clf_model_pipeline_sj_4,
                                clf_model_pipeline_iq_4,
                                clf_parameters_4,
                                model_scoring,
                                threshold,
                                output_name = None):
    """
    - wrapper function create stacking results. Each model pipeline contains different
      types of models
    - input:
        input_path - str, input location
        sj_data_pipeline - obj, data pipeline for city sj
        iq_data_pipeline - obj, data pipeline for city iq
        clf_model_pipeline_sj_1 - obj, perpensity model pipeline 1 for city sj
        clf_model_pipeline_iq_1 - obj, perpensity model pipeline 1 for city iq
        clf_parameters_1 - list[dict], a list of model 1 and its hyperparameters
        clf_model_pipeline_sj_2 - obj, perpensity model pipeline 2 for city sj
        clf_model_pipeline_iq_2 - obj, perpensity model pipeline 2 for city iq
        clf_parameters_2 - list[dict], a list of model 2 and its hyperparameters
        clf_model_pipeline_sj_3 - obj, perpensity model pipeline 3 for city sj
        clf_model_pipeline_iq_3 - obj, perpensity model pipeline 3 for city iq
        clf_parameters_3 - list[dict], a list of model 3 and its hyperparameters
        clf_model_pipeline_sj_4 - obj, perpensity model pipeline 4 for city sj
        clf_model_pipeline_iq_4 - obj, perpensity model pipeline 4 for city iq
        clf_parameters_4 - list[dict], a list of model 4 and its hyperparameters
        model_scoring - str, cv scoring name.
        threshold - int, the percentile we used to determine the outlier (positive label)
        output_name - str, the name of test prediction file
    - output: 
    """
    
    # get the target values
    train_label = pd.read_csv(input_path + "dengue_labels_train.csv")
    Y = train_label["total_cases"]
    
    # rune the wrapper function for find the best parameters for all 4 models
    (pred_train_sj_1, 
    pred_train_sj_cls_1, 
    pred_train_iq_1,
    pred_train_iq_cls_1,
    pred_test_sj_1,
    pred_test_sj_cls_1,
    pred_test_iq_1,
    pred_test_iq_cls_1,
    cv_score_sj_1,
    cv_score_iq_1) = single_clf_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                clf_model_pipeline_sj_1,
                                                clf_model_pipeline_iq_1,
                                                clf_parameters_1,
                                                model_scoring,
                                                threshold,
                                                None)

    (pred_train_sj_2, 
    pred_train_sj_cls_2, 
    pred_train_iq_2,
    pred_train_iq_cls_2,
    pred_test_sj_2,
    pred_test_sj_cls_2,
    pred_test_iq_2,
    pred_test_iq_cls_2,
    cv_score_sj_2,
    cv_score_iq_2) = single_clf_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                clf_model_pipeline_sj_2,
                                                clf_model_pipeline_iq_2,
                                                clf_parameters_2,
                                                model_scoring,
                                                threshold,
                                                None)
    
    (pred_train_sj_3, 
    pred_train_sj_cls_3, 
    pred_train_iq_3,
    pred_train_iq_cls_3,
    pred_test_sj_3,
    pred_test_sj_cls_3,
    pred_test_iq_3,
    pred_test_iq_cls_3,
    cv_score_sj_3,
    cv_score_iq_3) = single_clf_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                clf_model_pipeline_sj_3,
                                                clf_model_pipeline_iq_3,
                                                clf_parameters_3,
                                                model_scoring,
                                                threshold,
                                                None)

    (pred_train_sj_4, 
    pred_train_sj_cls_4, 
    pred_train_iq_4,
    pred_train_iq_cls_4,
    pred_test_sj_4,
    pred_test_sj_cls_4,
    pred_test_iq_4,
    pred_test_iq_cls_4,
    cv_score_sj_4,
    cv_score_iq_4) = single_clf_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                clf_model_pipeline_sj_4,
                                                clf_model_pipeline_iq_4,
                                                clf_parameters_4,
                                                model_scoring,
                                                threshold,
                                                None)
    
    # based on the cv score, estimate the weight of prediction of each model                                                            
    weight_sj = np.array([cv_score_sj_1, 
                          cv_score_sj_2, 
                          cv_score_sj_3, 
                          cv_score_sj_4])

    weight_sj = weight_sj/weight_sj.sum()
    
    # combine the prediction probability results based on weight
    pred_train_sj = (pred_train_sj_1 * weight_sj[0] 
                    + pred_train_sj_2 * weight_sj[1] 
                    + pred_train_sj_3 * weight_sj[2] 
                    + pred_train_sj_4 * weight_sj[3])

    pred_test_sj = (pred_test_sj_1 * weight_sj[0] 
                    + pred_test_sj_2 * weight_sj[1] 
                    + pred_test_sj_3 * weight_sj[2] 
                    + pred_test_sj_4 * weight_sj[3])

    # combinne the prediction class based on weight
    pred_train_sj_cls = (pred_train_sj_cls_1 * weight_sj[0] 
                        + pred_train_sj_cls_2 * weight_sj[1] 
                        + pred_train_sj_cls_3 * weight_sj[2] 
                        + pred_train_sj_cls_4 * weight_sj[3])

    pred_test_sj_cls = (pred_test_sj_cls_1 * weight_sj[0] 
                        + pred_test_sj_cls_2 * weight_sj[1] 
                        + pred_test_sj_cls_3 * weight_sj[2] 
                        + pred_test_sj_cls_4 * weight_sj[3])


    # based on the cv score, estimate the weight of prediction of each model                                                            
    weight_iq = np.array([cv_score_iq_1, 
                          cv_score_iq_2, 
                          cv_score_iq_3, 
                          cv_score_iq_4])

    weight_iq = weight_iq/weight_iq.sum()
    
    # combine the prediction probability results based on weight
    pred_train_iq = (pred_train_iq_1 * weight_iq[0] 
                    + pred_train_iq_2 * weight_iq[1] 
                    + pred_train_iq_3 * weight_iq[2] 
                    + pred_train_iq_4 * weight_iq[3])

    pred_test_iq = (pred_test_iq_1 * weight_iq[0] 
                    + pred_test_iq_2 * weight_iq[1] 
                    + pred_test_iq_3 * weight_iq[2] 
                    + pred_test_iq_4 * weight_iq[3])

    # combinne the prediction class based on weight
    pred_train_iq_cls = (pred_train_iq_cls_1 * weight_iq[0] 
                        + pred_train_iq_cls_2 * weight_iq[1] 
                        + pred_train_iq_cls_3 * weight_iq[2] 
                        + pred_train_iq_cls_4 * weight_iq[3])

    pred_test_iq_cls = (pred_test_iq_cls_1 * weight_iq[0] 
                        + pred_test_iq_cls_2 * weight_iq[1] 
                        + pred_test_iq_cls_3 * weight_iq[2] 
                        + pred_test_iq_cls_4 * weight_iq[3])
    
    # based on the predict probability, we loook at the top X%, where X = 1- threshold
    # find out the lower limit of the top X% proability
    sj_threshold = np.percentile(pred_train_sj, threshold)
    iq_threshold = np.percentile(pred_train_iq, threshold)

    # if the predict proability is higher than the threshold, convert it to positive label
    pred_train_sj = (pred_train_sj >= sj_threshold).astype(int)
    pred_train_iq = (pred_train_iq >= iq_threshold).astype(int)
    pred_train = np.concatenate([pred_train_sj, pred_train_iq])

    pred_test_sj = (pred_test_sj >= sj_threshold).astype(int)
    pred_test_iq = (pred_test_iq >= iq_threshold).astype(int)
    pred_test = np.concatenate([pred_test_sj, pred_test_iq]) 

    
    # put prediction together
    pred_train_cls = np.concatenate([pred_train_sj_cls, pred_train_iq_cls])
    pred_test_cls = np.concatenate([pred_test_sj_cls, pred_test_iq_cls])
    pred_train_cls = (pred_train_cls >= 0.5).astype(int)
    pred_test_cls = (pred_test_cls >= 0.5).astype(int)

    # plot the results
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_train)
    ax.set_title("Train - Stacked label based on probability")

    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_train_cls)
    ax.set_title("Train - Stacked label based on class")
    
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_test)
    ax.set_title("Test - Stacked label based on probability")

    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_test_cls)
    ax.set_title("Test - Stacked label based on class")

    # save the prediction to submission_format
    if output_name is not None:
        submission = pd.read_csv(input_path + "submission_format.csv")
        submission["predict_basedon_cls"] = pred_test_cls
        submission["predict_basedon_prob"] = pred_test
        submission.to_csv(input_path + output_name, index = False)
    
    return


def single_sev_model_fit_trans(input_path, 
                               sj_data_pipeline,
                               iq_data_pipeline,
                               reg_model_pipeline_sj,
                               reg_model_pipeline_iq,
                               reg_parameters,
                               model_scoring,
                               Y_threshold_p,
                               output_name = None):
    """
    - wrapper function to process data, select the best perpensity model based on cv score and prediction on test set
    - input:
        input_path - str, input location
        sj_data_pipeline - obj, data pipeline for city sj
        iq_data_pipeline - obj, data pipeline for city iq
        reg_model_pipeline_sj - obj, severity model pipeline for city sj
        reg_model_pipeline_iq - obj, severity model pipeline for city iq
        reg_parameters - list[dict], a list of different model and hyperparameters
        model_scoring - str, cv scoring name.
        Y_threshold_p - int, the percentile we used to determine the outlier (positive label)
        output_name - str, the name of test prediction file
    - output:
        pred_train - prediction probability on train data
        pred_test - prediction probability on test data
        overall_cv_score - overall cv score
    
    """
    # load the data and create the train, test data set
    train_data = pd.read_csv(input_path + "dengue_features_train.csv")
    train_label = pd.read_csv(input_path + "dengue_labels_train.csv")
    test_data = pd.read_csv(input_path + "dengue_features_test.csv")
    
    X_train_sj, Y_train_sj, X_test_sj = create_X_Y(train_data, train_label, test_data, sj_data_pipeline, "sj")
    X_train_iq, Y_train_iq, X_test_iq = create_X_Y(train_data, train_label, test_data, iq_data_pipeline, "iq")
    
    # add the threshold and sererity_mode parameter to the pipeline parameter set
    # in the model pipeline, if severity_mode is on, then the model will be only
    # be built on the outliers
    Y_threshold = np.percentile(Y_train_sj, Y_threshold_p)
    for i in range(len(reg_parameters)):
        reg_parameters[i]["general_estimator__threshold"] = [Y_threshold]
        reg_parameters[i]["general_estimator__severity_mode"] = [1]
    
    #print(reg_parameters)

    # build and tune the severity model through CV for SJ
    reg_model_sj = GridSearchCV(estimator = reg_model_pipeline_sj, 
                          param_grid = reg_parameters, 
                          scoring = model_scoring, 
                          cv = 5, 
                          verbose = 1,
                          n_jobs = -1,
                          return_train_score = True,
                          error_score = Y_train_sj.mean())
    reg_model_sj.fit(X_train_sj, Y_train_sj)
    
    # apply the same approach to IQ
    Y_threshold = np.percentile(Y_train_iq, Y_threshold_p)
    for i in range(len(reg_parameters)):
        reg_parameters[i]["general_estimator__threshold"] = [Y_threshold]
        reg_parameters[i]["general_estimator__severity_mode"] = [1]
    
    #print(reg_parameters)

    reg_model_iq = GridSearchCV(estimator = reg_model_pipeline_iq, 
                          param_grid = reg_parameters, 
                          scoring = model_scoring, 
                          cv = 5, 
                          verbose = 1,
                          n_jobs = -1,
                          return_train_score = True,
                          error_score = Y_train_iq.mean())

    reg_model_iq.fit(X_train_iq, Y_train_iq)

    # predict the severity on the whole train data
    pred_train_sj = reg_model_sj.best_estimator_.predict(X_train_sj)
    pred_train_iq = reg_model_iq.best_estimator_.predict(X_train_iq)

    # predict the severity on the whole test data
    pred_test_sj = reg_model_sj.best_estimator_.predict(X_test_sj)
    pred_test_iq = reg_model_iq.best_estimator_.predict(X_test_iq)
    
    # put results togehter
    pred_train = np.concatenate([pred_train_sj, pred_train_iq])
    pred_test = np.concatenate([pred_test_sj, pred_test_iq])
    
    print("Mean SJ CV score {}".format(reg_model_sj.best_score_))
    print("Mean IQ CV score {}".format(reg_model_iq.best_score_))
    print("Best Selected Parameters for SJ reg model: ")
    print(list(reg_model_sj.best_params_.items())[1:])
    print("Best Selected Parameters for IQ reg model: ")
    print(list(reg_model_iq.best_params_.items())[1:])
    
    # plot the results
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))
    ax.plot(pred_train)
    
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_test)
    
    # save the resuls to submission_format.csv
    if output_name is not None:
        submission = pd.read_csv(input_path + "submission_format.csv")
        submission["total_cases"] = pred_test.astype(int)
        submission.to_csv(input_path + output_name, index = False)
    
    #overall_cv_score = reg_model_sj.best_score_*9/14 + reg_model_sj.best_score_*5/14
    
    return (pred_train_sj, 
            pred_train_iq, 
            pred_test_sj, 
            pred_test_iq, 
            reg_model_sj.best_score_, 
            reg_model_iq.best_score_)


def stacking_sev_model_fit_trans(input_path, 
                                sj_data_pipeline,
                                iq_data_pipeline,
                                sev_model_pipeline_sj_1,
                                sev_model_pipeline_iq_1,
                                sev_parameters_1,
                                sev_model_pipeline_sj_2,
                                sev_model_pipeline_iq_2,
                                sev_parameters_2,
                                sev_model_pipeline_sj_3,
                                sev_model_pipeline_iq_3,
                                sev_parameters_3,
                                sev_model_pipeline_sj_4,
                                sev_model_pipeline_iq_4,
                                sev_parameters_4,
                                model_scoring,
                                Y_threshold_p,
                                output_name = None):
    """
    - wrapper function create stacking results. Each model pipeline contains different
      types of models
    - input:
        input_path - str, input location
        sj_data_pipeline - obj, data pipeline for city sj
        iq_data_pipeline - obj, data pipeline for city iq
        sev_model_pipeline_sj_1 - obj, severity model pipeline 1 for city sj
        sev_model_pipeline_iq_1 - obj, severity model pipeline 1 for city iq
        sev_parameters_1 - list[dict], a list of model 1 and its hyperparameters
        sev_model_pipeline_sj_2 - obj, severity model pipeline 2 for city sj
        sev_model_pipeline_iq_2 - obj, severity model pipeline 2 for city iq
        sev_parameters_2 - list[dict], a list of model 2 and its hyperparameters
        sev_model_pipeline_sj_3 - obj, severity model pipeline 3 for city sj
        sev_model_pipeline_iq_3 - obj, severity model pipeline 3 for city iq
        sev_parameters_3 - list[dict], a list of model 3 and its hyperparameters
        sev_model_pipeline_sj_4 - obj, severity model pipeline 4 for city sj
        sev_model_pipeline_iq_4 - obj, severity model pipeline 4 for city iq
        sev_parameters_4 - list[dict], a list of model 4 and its hyperparameters
        model_scoring - str, cv scoring name.
        Y_threshold_p - int, the percentile we used to determine the outlier 
        output_name - str, the name of test prediction file
    - output: 
    """ 

    # apply model pipeline for each type of severity model
    (pred_train_sj_1, 
     pred_train_iq_1, 
     pred_test_sj_1, 
     pred_test_iq_1, 
     cv_score_sj_1, 
     cv_score_iq_1) = single_sev_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                sev_model_pipeline_sj_1,
                                                sev_model_pipeline_iq_1,
                                                sev_parameters_1,
                                                model_scoring,
                                                Y_threshold_p,
                                                None)
    
    (pred_train_sj_2, 
     pred_train_iq_2, 
     pred_test_sj_2, 
     pred_test_iq_2, 
     cv_score_sj_2, 
     cv_score_iq_2) = single_sev_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                sev_model_pipeline_sj_2,
                                                sev_model_pipeline_iq_2,
                                                sev_parameters_2,
                                                model_scoring,
                                                Y_threshold_p,
                                                None)

    (pred_train_sj_3, 
     pred_train_iq_3, 
     pred_test_sj_3, 
     pred_test_iq_3, 
     cv_score_sj_3, 
     cv_score_iq_3) = single_sev_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                sev_model_pipeline_sj_3,
                                                sev_model_pipeline_iq_3,
                                                sev_parameters_3,
                                                model_scoring,
                                                Y_threshold_p,
                                                None)
    
    (pred_train_sj_4, 
     pred_train_iq_4, 
     pred_test_sj_4, 
     pred_test_iq_4, 
     cv_score_sj_4, 
     cv_score_iq_4) = single_sev_model_fit_trans(input_path, 
                                                sj_data_pipeline,
                                                iq_data_pipeline,
                                                sev_model_pipeline_sj_4,
                                                sev_model_pipeline_iq_4,
                                                sev_parameters_4,
                                                model_scoring,
                                                Y_threshold_p,
                                                None)

    # based on the cv score, estimate the weight of prediction of each model
    weight_sj = np.array([cv_score_sj_1, 
                          cv_score_sj_2, 
                          cv_score_sj_3, 
                          cv_score_sj_4])

    weight_sj = weight_sj/weight_sj.sum()
    
    # stacking the results together based on weight
    pred_train_sj = (pred_train_sj_1 * weight_sj[0] 
                    + pred_train_sj_2 * weight_sj[1] 
                    + pred_train_sj_3 * weight_sj[2] 
                    + pred_train_sj_4 * weight_sj[3])

    pred_test_sj = (pred_test_sj_1 * weight_sj[0] 
                    + pred_test_sj_2 * weight_sj[1] 
                    + pred_test_sj_3 * weight_sj[2] 
                    + pred_test_sj_4 * weight_sj[3])
    

    # based on the cv score, estimate the weight of prediction of each model
    weight_iq = np.array([cv_score_iq_1, 
                          cv_score_iq_2, 
                          cv_score_iq_3, 
                          cv_score_iq_4])

    weight_iq = weight_iq/weight_iq.sum()
    
    # stacking the results together based on weight
    pred_train_iq = (pred_train_iq_1 * weight_iq[0] 
                    + pred_train_iq_2 * weight_iq[1] 
                    + pred_train_iq_3 * weight_iq[2] 
                    + pred_train_iq_4 * weight_iq[3])

    pred_test_iq = (pred_test_iq_1 * weight_iq[0] 
                    + pred_test_iq_2 * weight_iq[1] 
                    + pred_test_iq_3 * weight_iq[2] 
                    + pred_test_iq_4 * weight_iq[3])


    pred_train = np.concatenate([pred_train_sj, pred_train_iq])
    pred_test = np.concatenate([pred_test_sj, pred_test_iq])
    
    # plot the results
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))
    ax.plot(pred_train)
    
    fig, ax = plt.subplots(nrows = 1,
                           figsize = (6,4))    
    ax.plot(pred_test)
    
    # save predictin to output file
    if output_name is not None:
        submission = pd.read_csv(input_path + "submission_format.csv")
        submission["total_cases"] = pred_test.astype(int)
        submission.to_csv(input_path + output_name, index = False)
    
    return