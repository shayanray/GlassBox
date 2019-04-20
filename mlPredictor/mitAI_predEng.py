
# coding: utf-8

# In[1]:


import matplotlib
#get_ipython().magic('matplotlib inline')
import seaborn as sns
from dummyPy import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier


# In[2]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


# In[3]:


# library imports
import numpy as np
import pandas as pd
import scipy as sc

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split #training and testing data split
from sklearn.svm import SVR
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score

#from sklearn.preprocessing import CategoricalEncoder
 
import time


def train_model(white_list , isSortedTopKfeatures=False):
    '''
    Train the ds_model
    '''

    # In[4]:


    # Load Train and Test CSV

    headerNames = ["id","name","first","last","compas_screening_date","sex","dob","age",
                   "age_cat","race","juv_fel_count","decile_score","juv_misd_count","juv_other_count",
                   "priors_count","days_b_screening_arrest","c_jail_in","c_jail_out","c_case_number",
                   "c_offense_date","c_arrest_date","c_days_from_compas","c_charge_degree","c_charge_desc",
                   "is_recid","num_r_cases","r_case_number","r_charge_degree","r_days_from_arrest",
                   "r_offense_date","r_charge_desc","r_jail_in","r_jail_out","is_violent_recid","num_vr_cases",
                   "vr_case_number","vr_charge_degree","vr_offense_date","vr_charge_desc","v_type_of_assessment",
                   "v_decile_score","v_score_text","v_screening_date","type_of_assessment","decile_score",
                   "score_text","screening_date"]
    prefix = "./data/"

    # ID cannot be used for prediction 
    # hence setting index_col = 0 takes care of removing ID field from dataset in both train and test dataframes.
    datadf = pd.read_csv(prefix + "compas-scores.csv", header=None, delim_whitespace=False,  names=headerNames, index_col=0,skiprows=1) 



    # In[5]:

    #['a','b']
    ## Drop columns not useful at all


    if 'id' in datadf:
        datadf = datadf.drop('id', axis=1)
        
    if 'name' in datadf: 
        datadf = datadf.drop('name', axis=1)
        
    if 'first' in datadf:
        datadf = datadf.drop('first', axis=1)
        
    if 'last' in datadf:
        datadf = datadf.drop('last', axis=1)

    if 'c_case_number' in datadf:
        datadf = datadf.drop('c_case_number', axis=1)

    if 'r_case_number' in datadf:
        datadf = datadf.drop('r_case_number', axis=1)

    if 'vr_case_number' in datadf:
        datadf = datadf.drop('vr_case_number', axis=1)

        
    if 'decile_score.1' in datadf:
        datadf = datadf.drop('decile_score.1', axis=1)


    if 'c_charge_desc' in datadf:
        datadf = datadf.drop('c_charge_desc', axis=1)


    if 'r_charge_desc' in datadf:
        datadf = datadf.drop('r_charge_desc', axis=1)

    if 'vr_charge_desc' in datadf:
        datadf = datadf.drop('vr_charge_desc', axis=1)


    if 'num_r_cases' in datadf:
        datadf = datadf.drop('num_r_cases', axis=1)

    if 'num_vr_cases' in datadf:
        datadf = datadf.drop('num_vr_cases', axis=1)


    if 'v_score_text' in datadf:
        datadf = datadf.drop('v_score_text', axis=1)

    if 'dob' in datadf:
        datadf = datadf.drop('dob', axis=1)

    if 'vr_charge_degree' in datadf:
        datadf = datadf.drop('vr_charge_degree', axis=1)
        
    if 'c_charge_degree' in datadf:
        datadf = datadf.drop('c_charge_degree', axis=1)

    if 'r_charge_degree' in datadf:
        datadf = datadf.drop('r_charge_degree', axis=1)

    if 'c_charge_degree' in datadf:
        datadf = datadf.drop('c_charge_degree', axis=1)

    
    ## check
    if 'v_decile_score' in datadf:
        datadf = datadf.drop('v_decile_score', axis=1)


    if 'c_jail_out' in datadf:
        datadf = datadf.drop('c_jail_out', axis=1)

    if 'r_jail_out' in datadf:
        datadf = datadf.drop('r_jail_out', axis=1)

    ## days_b_screening_arrest, c_days_from_compas ,r_days_from_arrest 
    if 'days_b_screening_arrest' in datadf:
        datadf = datadf.drop('days_b_screening_arrest', axis=1)


    if 'c_days_from_compas' in datadf:
        datadf = datadf.drop('c_days_from_compas', axis=1)

    if 'r_days_from_arrest' in datadf:
        datadf = datadf.drop('r_days_from_arrest', axis=1)


    # In[6]:



    print(datadf.shape)


    # In[7]:

    '''
    sns.heatmap(datadf.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #data.corr()-->correlation matrix
    fig=plt.gcf()
    fig.set_size_inches(20,16)
    #plt.show()
    fig.savefig('Correlation_before.png')
    

    # In[8]:


    fig=plt.gcf()
    datadf.hist(figsize=(18, 16), alpha=0.5, bins=50)
    plt.show()
    fig.savefig('histograms1.png')
    '''

    # In[9]:


    datadf.head(10)


    # In[10]:


    ## fill NaN for categorical

    #datadf['v_score_text'].fillna(datadf['v_score_text'].value_counts().index[0], inplace=True)
    #datadf['vr_charge_degree'].fillna(datadf['vr_charge_degree'].value_counts().index[0], inplace=True)
    #datadf['c_charge_desc'].fillna(datadf['c_charge_desc'].value_counts().index[0], inplace=True)
    #datadf['r_charge_desc'].fillna(datadf['r_charge_desc'].value_counts().index[0], inplace=True)
    #datadf['vr_charge_desc'].fillna(datadf['vr_charge_desc'].value_counts().index[0], inplace=True)


    # In[11]:


    '''
    datadf['vr_charge_degree'] = datadf['vr_charge_degree'].str.replace('[^a-zA-Z]',' ')
    datadf['vr_charge_degree'] = datadf['vr_charge_degree'].str.replace('[^a-zA-Z]',' ')

    datadf['v_score_text'] = datadf['v_score_text'].str.replace('[^a-zA-Z]',' ')
    datadf['v_score_text'] = datadf['v_score_text'].str.replace('[^a-zA-Z]',' ')
    '''


    # In[12]:


    if 'age' in datadf:
        datadf = datadf.drop('age', axis=1)


    # In[13]:


    encoder = OneHotEncoder(["sex", "race", "v_type_of_assessment", "age_cat", 
                             "type_of_assessment"]) # ,"v_score_text","c_charge_desc","r_charge_desc", "vr_charge_desc",
    encoder.fit(datadf)
    encoder.transform(datadf).shape
    encoder.transform(datadf).head(10)


    # In[14]:


    datadf = encoder.transform(datadf)

    print("DF shape >>>>>>>>>>>>>>>> ",datadf.shape)
    print("DF columsn >>>>>>>>>>>>>>>> ",datadf.columns)


    

    # In[16]:


    '''# Set of Unique Values 
    print(traindf['sex'].unique())
    print(traindf['age_cat'].unique())
    print(traindf['race'].unique())
    print(traindf['score_text'].unique())
    print(traindf['r_charge_desc'].unique())
    print(traindf['c_charge_desc'].unique())
    print(traindf['c_charge_degree'].unique())
    print(traindf['r_charge_degree'].unique())
    print(traindf['r_charge_desc'].unique())
    print(traindf['vr_charge_desc'].unique())
    print(traindf['v_type_of_assessment'].unique())
    print(traindf['v_score_text'].unique())
    print(traindf['score_text'].unique())



    traindf.columns
    '''


    # In[17]:


    # Test Data stats
    datadf.describe()


    # In[18]:


    # age for different date fields 
    #datadf['dob'] = pd.to_datetime(datadf['dob'], dayfirst=True)
    datadf['compas_screening_date'] = pd.to_datetime(datadf['compas_screening_date'], dayfirst=True)

    datadf['c_offense_date'] = pd.to_datetime(datadf['c_offense_date'], dayfirst=True)
    datadf['c_arrest_date'] = pd.to_datetime(datadf['c_arrest_date'], dayfirst=True)
    datadf['r_offense_date'] = pd.to_datetime(datadf['r_offense_date'], dayfirst=True)


    datadf['vr_offense_date'] = pd.to_datetime(datadf['vr_offense_date'], dayfirst=True)
    datadf['v_screening_date'] = pd.to_datetime(datadf['v_screening_date'], dayfirst=True)
    datadf['screening_date'] = pd.to_datetime(datadf['screening_date'], dayfirst=True)

    datadf['c_jail_in'] = pd.to_datetime(datadf['c_jail_in'], dayfirst=True)
    #datadf['c_jail_out'] = pd.to_datetime(datadf['c_jail_out'], dayfirst=True)

    datadf['r_jail_in'] = pd.to_datetime(datadf['r_jail_in'], dayfirst=True)
    #datadf['r_jail_out'] = pd.to_datetime(datadf['r_jail_out'], dayfirst=True)

    ## ages
    #datadf['Age_in_days'] = (datadf['compas_screening_date']-datadf['dob'])/timedelta(days=1)
    datadf['c_offense_age_in_days'] = (datadf['compas_screening_date']-datadf['c_offense_date'])/timedelta(days=1)
    datadf['c_arrest_age_in_days'] = (datadf['compas_screening_date']-datadf['c_arrest_date'])/timedelta(days=1)

    datadf['r_offense_age_in_days'] = (datadf['compas_screening_date']-datadf['r_offense_date'])/timedelta(days=1)
    datadf['vr_offense_age_in_days'] = (datadf['compas_screening_date']-datadf['vr_offense_date'])/timedelta(days=1)
    datadf['v_screening_age_in_days'] = (datadf['compas_screening_date']-datadf['v_screening_date'])/timedelta(days=1)
    datadf['screening_age_in_days'] = (datadf['compas_screening_date']-datadf['screening_date'])/timedelta(days=1)


    datadf['c_jail_in_age_in_days'] = (datadf['compas_screening_date']-datadf['c_jail_in'])/timedelta(days=1)
    #datadf['c_jail_out_age_in_days'] = (datadf['compas_screening_date']-datadf['c_jail_out'])/timedelta(days=1)


    datadf['r_jail_in_age_in_days'] = (datadf['compas_screening_date']-datadf['r_jail_in'])/timedelta(days=1)
    #datadf['r_jail_out_age_in_days'] = (datadf['compas_screening_date']-datadf['r_jail_out'])/timedelta(days=1)

    print("white_list ",white_list)
    if len(white_list) > 0:
        white_list.append('decile_score')
        datadf.drop(datadf.columns.difference(white_list), 1, inplace=True)
    
    print("datadf ",datadf.columns)
    ## drop all date cols
    if 'dob' in datadf:
        datadf = datadf.drop('dob', axis=1)
        
    if 'compas_screening_date' in datadf:
        datadf = datadf.drop('compas_screening_date', axis=1)

    if 'c_offense_date' in datadf:
        datadf = datadf.drop('c_offense_date', axis=1)

    if 'c_arrest_date' in datadf:
        datadf = datadf.drop('c_arrest_date', axis=1)

    if 'r_offense_date' in datadf:
        datadf = datadf.drop('r_offense_date', axis=1)

    if 'vr_offense_date' in datadf:
        datadf = datadf.drop('vr_offense_date', axis=1)

    if 'screening_date' in datadf:
        datadf = datadf.drop('screening_date', axis=1)


    if 'v_screening_date' in datadf:
        datadf = datadf.drop('v_screening_date', axis=1)


    if 'c_jail_in' in datadf:
        datadf = datadf.drop('c_jail_in', axis=1)

    if 'c_jail_out' in datadf:
        datadf = datadf.drop('c_jail_out', axis=1)


    if 'r_jail_in' in datadf:
        datadf = datadf.drop('r_jail_in', axis=1)

    if 'r_jail_out' in datadf:
        datadf = datadf.drop('r_jail_out', axis=1)


    #prediction column - textual (decile_score is the numeric equivalent)
    if 'score_text' in datadf:
        datadf = datadf.drop('score_text', axis=1)



    # In[19]:


    # stats of categorical features
    datadf.describe(include=['O'])



    # In[20]:


    print(datadf.shape)
    datadf.head(10)


    # In[21]:


    # for starters, fill every nan value with mean column values across the dataset.

    #fill NaN values with mean
    #datadf['r_jail_in_age_in_days'].fillna(datadf['r_jail_in_age_in_days'].dropna().mean(), inplace=True) 
    datadf[:] = datadf[:].fillna(0)
    '''
    datadf['r_jail_in_age_in_days'].fillna(0) 
    datadf['r_jail_out_age_in_days'].fillna(0) 
    datadf['c_jail_in_age_in_days'].fillna(0) 
    datadf['c_jail_out_age_in_days'].fillna(0) 

    datadf['vr_offense_age_in_days'].fillna(0) 
    datadf['r_offense_age_in_days'].fillna(0) 
    datadf['c_arrest_age_in_days'].fillna(0) 
    datadf['c_offense_age_in_days'].fillna(0) 
    '''



    # In[22]:


    datadf.to_csv('datadf_dt.csv', index=False)


    # In[23]:


    # check if any null values are still present
    print(datadf.columns[datadf.isnull().any()].tolist())


    # In[24]:


    #sample data for a quick run ## TODO removenext line
    traindf, testdf = train_test_split(datadf, random_state=42, test_size=0.3)

    print(traindf.shape)
    print(testdf.shape)


    # In[25]:


    from sklearn import preprocessing
    print(traindf.columns)
    print(traindf.columns[traindf.isnull().any()].tolist())



    # In[26]:


    ## Prediction ds_model  -TRAIN DF
    print(traindf.columns)
    train_features = traindf.loc[:, traindf.columns != 'decile_score']

    print(train_features.columns.values)

    # extract label from training set - Approved
    train_label = traindf.loc[:, traindf.columns == 'decile_score']
    print(train_label.columns)


    ## Prediction ds_model - TEST DF
    print(traindf.columns)
    test_features = testdf.loc[:, testdf.columns != 'decile_score']

    print(test_features.columns)
    print(test_features.head(10))

    # extract label from training set - Approved
    test_label = testdf.loc[:, testdf.columns == 'decile_score']
    print(test_label.columns)



    # In[27]:



    #get_ipython().run_cell_magic('time', '', "#Train the ds_model with best parameters of RF\n# best params for RF using randomizedCV\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.decomposition import PCA\n\n\nds_model = make_pipeline(StandardScaler(with_std=True), \n                         OneVsRestClassifier(\n                             DecisionTreeClassifier(random_state=42,min_samples_split=9)\n                                                )) # n_es = 200, lr 0.001\n\n'''\nds_model = make_pipeline(StandardScaler(with_std=True, with_mean=True), \n       MLPClassifier(activation='relu', alpha=10.0, batch_size='auto', beta_1=0.9,\n       beta_2=0.999, early_stopping=False, epsilon=1e-08,\n       hidden_layer_sizes=(7, 7), learning_rate='adaptive',\n       learning_rate_init=0.001, max_iter=500, momentum=0.9,\n       nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,\n       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,\n       warm_start=False))\n\nds_model = make_pipeline(StandardScaler(with_std=True), \n                         OneVsRestClassifier(\n                             ExtraTreesClassifier(n_estimators=98,min_samples_split=10\n                             ,max_leaf_nodes=8,max_features='log2',max_depth=3,criterion='entropy')\n                                                ))\n'''\n\nds_model.fit(train_features, train_label)\ntrain_pred = ds_model.predict(train_features)\n\nprint(metrics.accuracy_score(train_label, train_pred)) # Training Accuracy Score\nprint (np.sqrt(mean_squared_error(train_label, train_pred))) # Training RMSE\n#print(roc_auc_score(train_label, train_pred)) # AUC-ROC values")

    #Train the model with best parameters of RF
    # best params for RF using randomizedCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA


    ds_model = make_pipeline(StandardScaler(with_std=True), 
                             OneVsRestClassifier(
                                 DecisionTreeClassifier(random_state=42,min_samples_split=9)
                                                    )) # n_es = 200, lr 0.001

    '''
    ds_model = make_pipeline(StandardScaler(with_std=True, with_mean=True), 
           MLPClassifier(activation='relu', alpha=10.0, batch_size='auto', beta_1=0.9,
           beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(7, 7), learning_rate='adaptive',
           learning_rate_init=0.001, max_iter=500, momentum=0.9,
           nesterovs_momentum=True, power_t=0.5, random_state=42, shuffle=True,
           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
           warm_start=False))

    ds_model = make_pipeline(StandardScaler(with_std=True), 
                             OneVsRestClassifier(
                                 ExtraTreesClassifier(n_estimators=98,min_samples_split=10
                                 ,max_leaf_nodes=8,max_features='log2',max_depth=3,criterion='entropy')
                                                    ))
    '''

    ds_model.fit(train_features, train_label)
    train_pred = ds_model.predict(train_features)

    train_acc = metrics.accuracy_score(train_label, train_pred)
    print(train_acc) # Training Accuracy Score
    print (np.sqrt(mean_squared_error(train_label, train_pred))) # Training RMSE
    #print(roc_auc_score(train_label, train_pred)) # AUC-ROC values

    # In[29]:




    # In[30]:


    #test_pred = ds_model.predict_proba(testdf) #test features are all in testdata
    test_pred = ds_model.predict(test_features) #test features are all in testdata

    print(metrics.accuracy_score(train_label, train_pred)) # Training Accuracy Score
    print (np.sqrt(mean_squared_error(train_label, train_pred))) # Training RMSE

    test_pred_prob = ds_model.predict_proba(test_features) #test features are all in testdata
    print("ds_model.classes_ :: ",ds_model.classes_)
    print("****************************************************************************************")
    print("Predicted Output  >>>>>>>>> ",test_pred_prob) # Predicted Values
    print("****************************************************************************************")
    print("test_pred[:,1] >> ",test_pred_prob[:,1][0])

    print(metrics.accuracy_score(test_label, test_pred)) # Testing Accuracy Score
    print (np.sqrt(mean_squared_error(test_label, test_pred))) # Testing RMSE


    top_k_features = getSortedTopKfeatures(train_features, train_label)
    #top_k_features.append('decile_score')
    #return {"pred_accu" : train_acc}
    all_feat =train_features.columns.tolist()
    all_feat = list(set(all_feat) - set(top_k_features))
    ans = {"pred_accu" : train_acc, "topk" : top_k_features, "all_feat": all_feat}
    return ans
    # In[31]:


    ## Use lasso regression to penalize and figure out the best features


    # In[32]:

def getSortedTopKfeatures(train_features, train_label):


    reg = LassoCV()
    reg.fit(train_features, train_label)
    print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
    print("Best score using built-in LassoCV: %f" %reg.score(train_features,train_label))
    coef = pd.Series(reg.coef_, index = train_features.columns)


    # In[33]:


    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


    # In[34]:


    imp_coef = coef.sort_values()
    '''
    import matplotlib
    matplotlib.rcParams['figure.figsize'] = (8.0, 25.0)
    imp_coef.plot(kind = "barh")
    plt.title("Feature importance using Lasso ds_model")
    fig=plt.gcf()
    fig.set_size_inches(10,20)
    #plt.show()
    fig.savefig('Features_importance.png')
    '''
    # In[35]:


    ## drop all columns except.
    #df.drop(df.columns.difference(['a','b']), 1, inplace=True)


    # In[47]:


    coef


    # In[38]:


    print(len(coef))


    # In[65]:


    filter(lambda a: a != 0, coef)


    # In[70]:


    from collections import OrderedDict, defaultdict
    coef_dict = (coef).to_dict()


    # In[75]:


    import collections

    sorted_dict = sorted(coef_dict.items(), key=lambda kv: kv[1] ,reverse=True) #OrderedDict(coef_dict)

    keys = [k for k,v in sorted_dict if v != 0]
    #return {"topk": keys}
    return keys


if __name__=="__main__":
    
    white_list = ['age_cat_Less than 25',
     'race_African-American',
     'priors_count',
     'c_arrest_age_in_days',
     'c_offense_age_in_days',
     'r_offense_age_in_days',
     'c_jail_in_age_in_days',
     'r_jail_in_age_in_days',
     'vr_offense_age_in_days',
     'age_cat_Greater than 45',
     'decile_score']

    #white_list = [] # keep default list of columns
    
    '''    
    top_features = train_model(white_list,True)
    print("top Features sorted by importance", top_features)
    '''

    accuracy = train_model(white_list,False)
    print("Predictive Accuracy >>",accuracy)
