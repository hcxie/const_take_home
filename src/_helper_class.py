"""
_helper_class.py

Authors:
    Haochen Xie jbdx6307@gmail.com

Description:
    classes for DengAI compititon. These classes are used in the data and model pipeline

Created:
    11/7/2022

"""

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from statsmodels.tsa.stattools import adfuller

import collections

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import cross_val_score


#------------------------------------------------------------------------------
#--------------Data process, feature engineering and argumentation classes-----
#------------------------------------------------------------------------------

class Dengue_cat_encoder(BaseEstimator, TransformerMixin):
    """
    This class serves 3 functions:
    1. one-hot encode "month" variable
    2. target encode "weekofyear" variable
    3. remove column: "week_start_date", "year"
    """
    
    def __init__(self):
        self.target_encoder_dict = None
        self.month_list = None
        pass
    
    def fit(self, X, y):
        # ---------------------input check-------------------------------
        for column in ["weekofyear", "week_start_date"]:
            if column not in X.columns:
                raise ValueError("{} column must exist in the data".format(column))
        #----------------------------------------------------------------
        
        # print("Fitting Cat_encoder")
        
        # create a dict to store the target encoding values for week of year
        weekofyear = X.loc[:,["weekofyear"]].copy()
        weekofyear.loc[:, "total_cases"] = y.values
        self.target_encoder_dict = weekofyear.groupby("weekofyear")["total_cases"].mean().to_dict()
        
        # just in case, if we accdentily split a small portion of data and do not have the
        # all week numbers in a year, we impute it with mean cases each week from the training
        # data
        mean_cases = np.mean(list(self.target_encoder_dict.values()))
        week_within_a_year = int(365/7)+1
        for i in range(1, week_within_a_year+1):
            if i not in self.target_encoder_dict:
                self.target_encoder_dict[i] = mean_cases
                
        # check the month value in the training data
        # store it in the class for future verificaiton
        
        # print("Fitting Cat_encoder completed")
                
        return self
    
    def transform(self, X):
        # ---------------------input check-------------------------------
        for column in ["weekofyear", "week_start_date", "year"]:
            if column not in X.columns:
                raise ValueError("{} column must exist in the data".format(column))
                
        if not self.target_encoder_dict:
            raise ValueError("Please fit the training data first")
        #----------------------------------------------------------------
        
        # print("Transforming data using Cat_encoder")
        
        # extract month and one hot encodeing it
        temp_date = pd.to_datetime(X.loc[:, "week_start_date"])
        X.loc[:, "month"] = temp_date.dt.month.astype(object)
        new_X = X.drop(columns=["week_start_date", "year"])
        dummy_month = pd.get_dummies(new_X["month"], drop_first = True)
        
        # when we first fit the training data, record the month we included
        if not self.month_list:
            self.month_list = set(dummy_month.columns)
        
        # check with month_list to make sure all month names appear in the data
        # if we transform the test data and a month does not exist, we can add a all-0 column to that month
        check_list = list(self.month_list - set(dummy_month.columns))
        if len(check_list) != 0:
            for col in check_list:
                print("{} month does not exist in the current data, append column with all 0".format(col))
                dummy_month.loc[:, col] = 0

        # adding the following comments to fix the issue when build serverity model
        check_list = list(set([2,3,4,5,6,7,8,9,10,11,12]) - self.month_list)
        if len(check_list) != 0:
            for col in check_list:
                print("{} month does not exist in the current data, append column with all 0".format(col))
                dummy_month.loc[:, col] = 0
               
        # add prefix to dummy month and combine it with original data
        dummy_month = dummy_month.add_prefix("month_")
        new_X = new_X.drop(columns=["month"])
        new_X = pd.concat([new_X, dummy_month], axis = 1)
        
        # apply target encoding value to weekofyear
        new_X.loc[:, "weekofyear"] = new_X.loc[:, "weekofyear"].map(self.target_encoder_dict)
        
        # print("Transforming data using Cat_encoder completed")
        
        return new_X

#------------------------------------------------------------------------------
class Stationarity_adjustment(BaseEstimator, TransformerMixin):
    """
    This class serves three functions:
    1. Stationarity check through ADF
    2. Remove unit-root using differecning method
    3. Check the input data and adjust the data based on function 1 and 2
    
    This class will only apply to the original columns, then encoded columns like
    month_ and weekofyear will be ingored automatically
    
    """

    def __init__(self, p_value = 0.05):
        """
        p_value: the siginficance level for adf test
        """
        self.p_value = p_value
        pass

    def stationarity_check_adf(self, predictor):
        """
        stationary check using adf method
        
        input: predictor - pandas series:the column of the selected feature
        output: adf_checker - bool: If ture, reject the null hypothesis
                              the predictor does not suffer unit-root stationary
        """
        # remove missing value
        temp_predictor = predictor.dropna()
        # apply adf test
        adf_checker = adfuller(x=temp_predictor, 
                                  regression="c", 
                                  autolag="AIC")[1] < self.p_value
        return adf_checker

    def differencing(self, predictor, adf_checker):
        """
        adjust the predicor who suffers the unit-root stationary using differencing until pass the adf test
        
        input: predictor - pandas series:the column of the selected feature
               adf_checker - bool: If ture, reject the null hypothesis
                              the predictor does not suffer unit-root stationary
        """
        predictor.dropna(inplace=True)
        # apply adifferencing until it pass the adf test
        while not adf_checker:
            predictor = predictor.diff()
            adf_checker = self.stationarity_check_adf(predictor)
            
        return predictor

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        """
        stationarity check on all original columns
        Ingore the month encoding columns and weekofyear column since they are encoded column
        if check failed, apply differencing
        
        If the data does not have enough obs, adf test may failed, in that case, we skip. This 
        will not happen as long as we carefully split the data
        """

        try:
            # print("checking stationarity")
            original_columns = [x for x in X.columns if not x.startswith("month_") and "weekofyear" not in x]
            for col in original_columns:
                predictor = X.loc[:, col].copy()
                adf_checker = self.stationarity_check_adf(predictor)
                if not adf_checker:
                    # print("{} failed the stationarity check, applying differencing".format(col))
                    predictor = self.differencing(predictor, adf_checker)
                
                X.loc[:, col] = predictor.copy()
            
            # print("Checking stationarity completed")
                
        except ValueError as e:
            print("ERROR!:{}".format(e))
        return X

#------------------------------------------------------------------------------
class Feature_argumentation1(BaseEstimator, TransformerMixin):
    def __int__(self):
        pass
        
    def fit(self, X, y = None):
        return self
        
    def transform(self, X):
        # print("Argumenting feature - part 1")
        output_X = X.copy()
        
        # ndvi std between ne, nw, se and sw
        temp_X = X.loc[:, ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw"]].copy()
        output_X.loc[:, "fa_ndvi_std"] = temp_X.std(axis = 1)
        
        # create: range in station_max_temp
        #         skewness in stationa temperature
        temp_X = X.loc[:,["station_max_temp_c", 
                          "station_min_temp_c", 
                          "station_avg_temp_c"]].copy()
        
        output_X.loc[:, "fa_station_temp_range_c"] = temp_X["station_max_temp_c"] - temp_X["station_min_temp_c"]        
        output_X.loc[:, "fa_station_temp_skew_c"] = (temp_X["station_max_temp_c"] + temp_X["station_min_temp_c"])/2 - temp_X["station_avg_temp_c"]
        
        # create: range in reanalysis temperature
        #         skewness in reanalysis temperature
        #         temp difference between average and dew point
        temp_X = X.loc[:,["reanalysis_dew_point_temp_k", 
                          "reanalysis_air_temp_k", 
                          "reanalysis_max_air_temp_k",
                          "reanalysis_min_air_temp_k",
                          "reanalysis_avg_temp_k"]].copy()
        
        output_X.loc[:, "fa_reanalysis_temp_range_k"] = temp_X["reanalysis_max_air_temp_k"] - temp_X["reanalysis_min_air_temp_k"]
        output_X.loc[:, "fa_reanalysis_temp_skew_k"] = (temp_X["reanalysis_max_air_temp_k"] + temp_X["reanalysis_min_air_temp_k"])/2 - temp_X["reanalysis_avg_temp_k"]
        output_X.loc[:, "fa_reanalysis_dew_diff_k"] = temp_X["reanalysis_avg_temp_k"] - temp_X["reanalysis_dew_point_temp_k"]
        
        # print("Argumenting feature - part 1 completed")
        
        return output_X

#------------------------------------------------------------------------------
class Standardization(BaseEstimator, TransformerMixin):
    """
    Standardizing the data and return the standarzed dataframe
    """

    def __init__(self):
        self.column_names = None
        self.fitted_standarizer = None       
        pass

    def fit(self, X, y=None):
        # print("Fitting Standarizer")
        self.column_names = X.columns
        scaler = StandardScaler()
        self.fitted_standarizer = scaler.fit(X)
        # print("Fitting Standarizer Completed")
        return self

    def transform(self, X):
        # ---------------------input check-------------------------------                
        if self.column_names is None or self.fitted_standarizer is None:
            raise ValueError("Please fit the standardizer first")
        #----------------------------------------------------------------
        
        # print("Applying Standarizer")
        X_scaled = self.fitted_standarizer.transform(X)
        X_scaled_df = pd.DataFrame(data=X_scaled, columns=self.column_names)
        # print("Applying Standarizer Completed")
        
        return X_scaled_df

class Imputer(BaseEstimator, TransformerMixin):
    """
    This class is used to impute mising values
    We have two mode:
        mode 0: impute missing value using linear interpolate, since the original features are time series based
        mode 1: impute missing value using KNN imputer.
    """

    def __init__(self, impute_mode = 1, n_neighbors = 10):
        """
        impute_mode: 1/0 binary flag
        n_nenighbours: int, the number of neighbors used in KNN imputer
        """
        self.impute_mode = impute_mode
        self.n_neighbors = n_neighbors
        self.columns_names = None
        self.fitted_imputer = None
        pass

    def fit(self, X, y=None):
        if self.impute_mode == 1:
            # print("Fitting KNN imputer")
            self.columns_names = X.columns
            impute = KNNImputer(n_neighbors=self.n_neighbors)
            self.fitted_imputer = impute.fit(X)          
            # print("Fitting KNN imputer Completed")
        #else:
            # print("Linear interpolatation imputation method will be used. No fitting needed")
            
        return self

    def transform(self, X):
        # ---------------------input check-------------------------------                
        if self.impute_mode == 1 and (self.columns_names is None or self.fitted_imputer is None):
            raise ValueError("Please fit the KNN imputer first")
        #----------------------------------------------------------------
        # print("Imputing Missing Values")
        if self.impute_mode == 0:
            imputed_X_df = X.interpolate()
        else:
            imputed_X = self.fitted_imputer.transform(X)
            imputed_X_df = pd.DataFrame(data=imputed_X, columns=self.columns_names)
        # print("Imputing Missing Values completed")
        
        return imputed_X_df

#------------------------------------------------------------------------------
class Feature_argumentation2(BaseEstimator, TransformerMixin):
    """
    - This class is used for feature argumentation part 2.
    - In this part, gradient and drift features are created
      - gradient feature: the gradient of the original feature based on a specified time period
      - drift feature: simply move the feature value forward based on a specified time period
    - the time period is pre-set as 4 weeks
    """
    
    def __init__(self, mode = 2, max_t_drift = 4):
        """
        input: mode - 1/0 flag, a swtich to determine whether to use this argumentation step
               max_t_drift: int, specified time period
               drift_data: the drift data used to append on test set
        """
        self.mode = mode
        self.max_t_drift = max_t_drift
        self.drift_data = None
        self.count = 0
        pass
    
    def fit(self, X, y = None):
        """
        store the drift data
        """
        self.drift_data = X.tail(self.max_t_drift)
        self.count = 1
        return self
    
    def _fitcheck(self):
        """
        - This function is used to check whether the transform is applied to the train data or not.
        - If the tail of input data is not identical to the drift data, the current input data is the test set, return True.
        - Otherwise, return False
        """
        
        if self.drift_data is not None and self.count > 1:
            return True
        
        return False
        
    
    def _cal_gradient(self, original_X, X, t_drift):
        """
        - This function is used to create the gradient features
        - input: original_X - dataframe, the original input dataframe. Used in _fitcheck only
                 X - dataframe, the original dataframe + the drift data. Used to derive gradient feature
                 t_drift - int, the specificed time window
        """
        
        temp_X = X.copy()
        temp_X.reset_index(drop = True, inplace = True)
        
        # create a dictionary to store the gradient features
        out_X = collections.defaultdict(list)
        
        for col in temp_X.columns:
            # the name of gradient feature comes with prefix gradient_
            new_col_name = "gradient_" + str(t_drift) + "_" + col
            # create a empty list to store the gradient for each obs
            temp_gradient = []
            for i in range(t_drift, temp_X.shape[0]):
                # since the time window is small, we simply use least square method
                # to get the gradient
                x = np.arange(0, t_drift)
                A = np.vstack([x, np.ones(len(x))]).T
                y = temp_X.loc[:, col].to_numpy()
                y = y[i - t_drift:i]
                m, c = np.linalg.lstsq(A, y, rcond = None)[0]
                # store the gradient to temp list
                temp_gradient.append(m)
            
            # for the first t_drift obs, we can not calcualte gradient
            # thus use bfill method to assign gradient value
            out_X[new_col_name] = [temp_gradient[0]]*t_drift + temp_gradient
        
        # conver to dataframe
        out_X = pd.DataFrame(out_X)        
        out_columns = out_X.columns
        
        # Get rid of the appended drift data
        out_array = out_X.to_numpy()[self.max_t_drift:,:]
        
        # If we are dealing with train data, bfill the feature values in time winodws
        #if not self._fitcheck(original_X):
        #    temp_array = np.repeat(out_array[[t_drift],:], t_drift, axis = 0)
        #    out_array = np.row_stack((temp_array, out_array[t_drift:, :]))
        
                
        return out_array, out_columns
    
    
    def _drift_feature(self, original_X, X, t_drift):
        """
        - This function is used to create the drift features
        - input: original_X - dataframe, the original input dataframe. Used in _fitcheck only
                 X - dataframe, the original dataframe + the drift data. Used to derive gradient feature
                 t_drift - int, the specificed time window
        """
        temp_X = X.copy()
        temp_X.reset_index(drop = True, inplace = True)
        
        # create a dictionary to store the gradient features
        out_X = collections.defaultdict(list)
        
        for col in temp_X.columns:
            # the name of drift feature comes with prefix drift_
            new_col_name = "drift_" + str(t_drift) + "_" + col
            out_X[new_col_name] = list(temp_X.loc[:, col])
        
        # conver to dataframe
        out_X = pd.DataFrame(out_X)
        out_columns = out_X.columns
        
        # Get rid of the appended drift data
        start_index = self.max_t_drift - t_drift
        end_index = out_X.shape[0] - t_drift
        out_array = out_X.to_numpy()[start_index:end_index,:]

        # If we are dealing with train data, bfill the feature values in time winodws
        #if not self._fitcheck(original_X):
        #    temp_array = np.repeat(out_array[[t_drift],:], t_drift, axis = 0)
        #    out_array = np.row_stack((temp_array, out_array[t_drift:, :]))
        
        
        return out_array, out_columns
            
    
    def transform(self, X):
        
        output_X = X.copy()
        
        # the drift features does not contains encoded feature
        drift_columns = [x for x in X.columns if not x.startswith("month_") and "weekofyear" not in x]
        # the gradient features must be derived from original predictors
        gradient_columns = [x for x in drift_columns if not x.startswith("fa_")]
        
        drift_X = X.loc[:, drift_columns].copy()
        gradient_X = X.loc[:, gradient_columns].copy()
        
        if self.mode > 1:
            # print("Argumenting feature - part 2")
            if self._fitcheck():
                print("trasfrom")
                temp_drift_data = self.drift_data.loc[:, drift_columns].copy()
                temp_gradient_data = self.drift_data.loc[:, gradient_columns].copy()
            else:
                print("fit_trasfrom")
                temp_drift_data = pd.DataFrame(np.repeat(drift_X.values[[0],:], self.max_t_drift, axis = 0), columns = drift_X.columns)
                temp_gradient_data = pd.DataFrame(np.repeat(gradient_X.values[[0],:], self.max_t_drift, axis = 0), columns = drift_X.columns)

            drift_X = pd.concat([temp_drift_data, drift_X], ignore_index = True)
            gradient_X = pd.concat([temp_gradient_data, gradient_X], ignore_index = True)
            
            # iterate from 1 to the max drift window
            # create drifted features and gradient features for each drift time window size
            for i in range(1, self.max_t_drift + 1):
                out_array1, out_columns1 = self._drift_feature(X, drift_X, i)
                out_array2, out_columns2 = self._cal_gradient(X, gradient_X, i)
                output_X.loc[:, out_columns1] = out_array1
                output_X.loc[:, out_columns2] = out_array2
            
            # print("Argumenting feature - part 2 completed")
        elif self.mode == 1:
            # print("Argumenting feature - part 2")

            if self._fitcheck():
                print("trasfrom")
                temp_drift_data = self.drift_data.loc[:, drift_columns].copy()
            else:
                print("fit_trasfrom")
                temp_drift_data = pd.DataFrame(np.repeat(drift_X.values[[0],:], self.max_t_drift, axis = 0), columns = drift_X.columns)

            drift_X = pd.concat([temp_drift_data, drift_X], ignore_index = True)
            
            # iterate from 1 to the max drift window
            # create drifted features and gradient features for each drift time window size
            for i in range(1, self.max_t_drift + 1):
                out_array1, out_columns1 = self._drift_feature(X, drift_X, i)
                output_X.loc[:, out_columns1] = out_array1
            
            # print("Argumenting feature - part 2 completed")
        
        self.count += 1

        return output_X

#------------------------------------------------------------------------------
class Feature_argumentation3(BaseEstimator, TransformerMixin):
    """
     - This feature argumentation step will add polynomial features
    """
    
    def __init__(self, mode = 1, degree = 2):
        """
        - input: mode - 1/0 flag, a swticher to determine whether to use argumentation part 3
                 degree - int, the degree of polynomial       
        """
        self.mode = mode
        self.degree = degree
        self.fitted_standarizer = None
        pass
    
    def _polyfeatures(self, X):
        """
        This function will apply polynomial transformation to create polynomials based on the original variables
        """
        # select the columns to be included in the polynominal transformation
        poly_columns = [x for x in X.columns if not x.startswith("month_") and "weekofyear" not in x]
        poly_columns = [x for x in poly_columns if not x.startswith("drift_") and not x.startswith("gradient_")]
        poly_data = X.loc[:, poly_columns].copy()
        
        # apply polynominal transformation
        poly = PolynomialFeatures(degree=self.degree, include_bias=False, interaction_only=False)
        poly_array = poly.fit_transform(poly_data)
        
        # save the generated feature into a dataframe
        poly_feature_names = poly.get_feature_names_out(poly_columns)
        poly_data_df = pd.DataFrame(data=poly_array, columns=poly_feature_names)
        poly_wo_old_feature = poly_data_df.loc[:, [x for x in poly_feature_names if x not in poly_columns]]
        return poly_wo_old_feature
        
    
    def fit(self, X, y = None):
        if self.mode == 1:
            # print("Fitting Standarizer for argumented feature in part 3")
            # To make sure the generated features are standarized. we create a standardscaler here
            # and save it in the obejct
            poly_wo_old_feature = self._polyfeatures(X)
            scaler = StandardScaler()
            self.fitted_standarizer = scaler.fit(poly_wo_old_feature)
            # print("Fitting Standarizer for argumented feature in part 3 Completed")
        return self
    
    
    def transform(self, X):
        
        if self.mode == 1:
            # ---------------------input check-------------------------------                
            if self.fitted_standarizer is None:
                raise ValueError("Please fit this object first.")
            #----------------------------------------------------------------
            # print("Argumenting feature - part 3")
            # obtain the polynomial features
            poly_wo_old_feature = self._polyfeatures(X)
            temp_columns = poly_wo_old_feature.columns
            
            # apply the standardizer and add the new features to original input data
            poly_wo_old_feature = self.fitted_standarizer.transform(poly_wo_old_feature)
            poly_wo_old_feature = pd.DataFrame(data=poly_wo_old_feature, 
                                               columns=temp_columns)
            
            X = pd.concat([X, poly_wo_old_feature], axis=1)
            
            # print("Argumenting feature - part 3 completed")
        
        return X

#------------------------------------------------------------------------------
class Feature_selection(BaseEstimator, TransformerMixin):
    """
    - This class is based on paper https://ieeexplore.ieee.org/abstract/document/8871132
    - linear model, svm model and random forest model will be used to estimate the feature
      importances.
    - For each model, ranked the feature by feature importance and select the top X% importance
      feature. Then we treate each feature as the row, each obs as the columns and apply clustering.
      If two features are similar in vector space. Then these two features will be selected by the 
      same cluster. For each cluster we pick up the most importance feature.
    - Affinity Propagation clustering is used. It is a heirical clustering method which does not
      need to specify the number of clusters: https://www.toptal.com/machine-learning/clustering-algorithms
    - Three models will generate three importance feature lists. Combine all selected feature together as
      the final list
    - In order to determine X%, cross validation is used to evaluate the selected feature on the final
      estimator. Then, the X% leads to the best cv score will be used as the final feature selection fraction
    """

    def __init__(self, 
                 feature_frac_list = [0.01, 0.05, 0.1, 0.25, 0.5], 
                 cv_metric = "r2", 
                 classify_mode = 0, 
                 estimator=None):
        # a list of feature fraction that will be explored
        self.feature_frac_list = feature_frac_list
        # the cv metric used to determine the best feature fraction
        self.cv_metric = cv_metric
        # 1/0 flag, a swtich to determine if feature selection is for a classfication task or not
        self.classify_mode = classify_mode
        # the estimator object for the final model
        self.estimator = estimator
        # A list of all input variable names
        self.X_columns = None
        # A list of the selected best features
        self.best_features = None
        pass

    def _importance_df_creator(self, importances):
        """
        - create a dataframe to store the feature importance
        - input: importances - array, store feature importance or coefficient
        - output: importance_df - dataframe, store feature names, importances and rank.
                                this dataframe is ranked by importance in descending order
        """
        # create a dataframe with feature names and their importances
        importances_df = pd.DataFrame({"feature_names": self.X_columns, 
                                       "importances": importances})
        # get the rank based on importances, rank 1 means the most importance
        # it is possible that some variables share the same rank
        importances_df.loc[:, "rank"] = importances_df.loc[:, "importances"].rank(method="dense", 
                                                                                  ascending=False)
        # sort the dataframe by rank
        importances_df.sort_values(by="rank", 
                                   inplace=True)
        return importances_df

    def _coef_feature_selection(self, model, X, y):
        """
        - fitting a baseline model and output the feature importance dataframe
        - input: model - model object
                 X - input variables
                 y - target variables
        - output: importances_df - dataframe, feature importance dataframe
        """
        # if the model used to determine feature importance is based on coefficient
        # we fit the model and get the coefficients
        model.fit(X, y)
        model_coef = abs(model.coef_)
        # some model output a 1-dimension coef, some model output a (1,X) shape coefficient
        # reshape the coefficent array to make sure it is only 1-dimension
        model_coef = model_coef.reshape(-1)
        # create the feature importance dataframe
        importances_df = self._importance_df_creator(model_coef)
        return importances_df

    def _non_coef_feature_selection(self, model, X, y):
        """
        - fitting a baseline model and output the feature importance dataframe
        - input: model - model object
                 X - input variables
                 y - target variables
        - output: importances_df - dataframe, feature importance dataframe
        """
        # for tree-based model, feature importances are obtained from
        # feature_importnaces_
        model.fit(X, y)
        model_fi = model.feature_importances_
        # get feature importance dataframe
        importances_df = self._importance_df_creator(model_fi)
        return importances_df

    def _clustering_features(self, df1, df2, df3, X):
        """
        - Apply affinity propagation clustering to get the top features
        - input: df1 - dataframe, importance df based on svm
                 df2 - dataframe, importance df based on linear model
                 df3 - dataframe, importance df based on random forest
                 X - dataframe, input data
        - output: unique_relevant_features_list - list, a list of relevant features
        """
        # a list to store selected relevant features
        relevant_feature_list = []
        
        cluster_algo = AffinityPropagation(random_state=42, max_iter=2500)
        
        # based on feature rankings determined by each model
        # select the relevant features based on clustering result
        for df in [df1, df2, df3]:
            subset_df = df.iloc[:self.n_keep, :].copy()
            chosen_features = subset_df.loc[:, "feature_names"].tolist()
            subset_feature_data = X.loc[:, chosen_features]
            feature_cluster = cluster_algo.fit_predict(subset_feature_data.T)
            subset_df.loc[:, "cluster"] = feature_cluster
            
            # iterate through each cluster and select the most relevant feature
            # in each cluster
            for cluster in set(feature_cluster):
                cluster_df = subset_df.loc[subset_df["cluster"] == cluster, :]
                lowest_rank = cluster_df.loc[:, "rank"].min()
                relevant_features = cluster_df.loc[cluster_df["rank"] == lowest_rank, :].loc[:, "feature_names"].tolist()
                relevant_feature_list += relevant_features
        
        # get rid of the overlapping features
        unique_relevant_features_list = list(set(relevant_feature_list))
        return unique_relevant_features_list

    def _adjusted_R2_score(self, R2_cv_scores, nfeatures, nobs):
        """
        A function to calculate adjusted R2 score
        """
        scores = np.mean(R2_cv_scores)
        adj_score = 1 - ((1-scores) * (nobs-1) / (nobs-nfeatures-1))
        return adj_score

    def _choose_best_features(self, X, y):
        """
        - A wrapper function to select the best relevant features
        """
        cv = 3
        nobs = len(X)
        list_scoring_results, list_selected_features = [], []
        
        # create the estimator object
        if self.classify_mode == 1:
            svm_model = svm.SVC(kernel="linear", random_state=42)
            linear_model = LogisticRegression(max_iter=5_000, random_state=42)
            rfr_model = RandomForestClassifier(random_state=42)
            if self.cv_metric == "r2":
                raise ValueError("Please assign a classification metric to cv_metric. The default R2 is for regression")
        else:
            svm_model = svm.SVR(kernel="linear")
            linear_model = Ridge()
            rfr_model = RandomForestRegressor(random_state=42)
            
            # since the max cv_metric value will be selected to determine the final features
            # restric regression metric to r2
            if self.cv_metric != "r2":
                # print("Swtiched cv_metric for R2")
                self.cv_metric = "r2"

        svm_columns_df = self._coef_feature_selection(svm_model, X, y)
        lgr_columns_df = self._coef_feature_selection(linear_model, X, y)
        rfr_columns_df = self._non_coef_feature_selection(rfr_model, X, y)
        
        # iterate through each feature fraction, evaluate feature selection through cv
        for feature_frac in self.feature_frac_list:
            self.n_keep = int(len(self.X_columns) * feature_frac)
            selected_features = self._clustering_features(svm_columns_df, lgr_columns_df, rfr_columns_df, X=X)

            list_selected_features.append(selected_features)
            subset_X = X.loc[:, selected_features].copy()
            cv_scores = cross_val_score(self.estimator, subset_X, y, cv=cv, scoring=self.cv_metric)

            nfeatures = len(subset_X.columns)
            if self.cv_metric == "r2":
                adj_score = self._adjusted_R2_score(cv_scores, nfeatures, nobs)
            else:
                adj_score = np.mean(cv_scores)
                
            list_scoring_results.append(adj_score)
      
        # select the best feature set based on cv score
        max_index = np.argmax(list_scoring_results)
        best_features = list_selected_features[max_index]
        
        # print("Selecting features: the best fraction is {}".format(self.feature_frac_list[max_index]))
        
        return best_features

    def fit(self, X, y):
        # print("Selecting features")
        self.X_columns = X.columns

        self.best_features = self._choose_best_features(X, y)
        # print("Selecting features completed")
        return self

    def transform(self, X):
        # ---------------------input check-------------------------------                
        if self.best_features is None:
                raise ValueError("Feature has not been selected based on training data")
        #----------------------------------------------------------------        
        subset_X = X.loc[:, self.best_features]
        
        # print("Applied selected features")
        
        return subset_X



#------------------------------------------------------------------------------
#----------estimator classes---------------------------------------------------
#------------------------------------------------------------------------------

class General_estimator(BaseEstimator, RegressorMixin):
    """
    A general estimator class. This class include any model estimator and applyied to different pipelines
    """
    def __init__(self, estimator=None, severity_mode = 0, threshold = 60):
        self.estimator = estimator
        self.severity_mode = severity_mode
        self.threshold = threshold
        pass

    def fit(self, X, y):
        if self.severity_mode == 1:
            #temp_X = X.copy()
            #temp_X["total_cases"] = y

            #subset_X = temp_X.loc[temp_X["total_cases"] >= self.threshold, :]
            #subset_X.reset_index(drop = True, inplace = True)
            #subset_Y = subset_X["total_cases"].copy()
            #subset_X.drop(columns=["total_cases"], inplace = True)

            #while len(subset_X) <= 0:
            #    self.threshold -= 1
            #    subset_X = temp_X.loc[temp_X["total_cases"] >= self.threshold, :]
            #    subset_X.reset_index(drop = True, inplace = True)
            #    subset_Y = subset_X["total_cases"].copy()
            #    subset_X.drop(columns=["total_cases"], inplace = True)
            subset_X = X[y >= self.threshold, :]
            subset_Y = y[y >= self.threshold]

            temp_threshold = self.threshold

            while len(subset_X) <= 0:
                temp_threshold -= 1
                subset_X = X[y >= temp_threshold, :]
                subset_Y = y[y >= temp_threshold]               


            self.estimator.fit(subset_X, subset_Y)
        else:
            self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        pred = self.estimator.predict(X)
        return pred
    
    def predict_proba(self, X, y=None):
        if not hasattr(self.estimator, 'predict_proba'):
            raise ValueError("This estimator does not have predict_prob method!")
        pred = self.estimator.predict_proba(X)
        return pred