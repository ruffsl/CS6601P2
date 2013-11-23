# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import csv
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import Pysolar as ps
import datetime
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from itertools import izip

# <codecell>

SEED = 42 # Random seed to keep consistent

# <codecell>

'''
Loads a list of GEFS files merging them into model format.
'''
def load_GEFS_data(directory,files_to_use,file_sub_str):
    for i,f in enumerate(files_to_use):
        if i == 0:
            T, X = load_GEFS_file(directory,files_to_use[i],file_sub_str)
            X = np.expand_dims(X, axis=1)
        else:
            T, X_new = load_GEFS_file(directory,files_to_use[i],file_sub_str)
            X_new = np.expand_dims(X_new, axis=1)
            X = np.hstack((X,X_new))
    return T, X

'''
Loads GEFS file using specified merge technique.
'''
def load_GEFS_file(directory,data_type,file_sub_str):
        print 'loading',data_type
        path = os.path.join(directory,data_type+file_sub_str)
        print 'this is the path: ', path
	data = nc.Dataset(path,'r+')
	T = data.variables['intTime'][:]
        X = data.variables.values()[-1][:,:,:,:,:] # get rid of some GEFS points
        #X = X.reshape(X.shape[0],55,4,10)                               # Reshape to merge sub_models and time_forcasts
        #X = np.mean(X,axis=1)                                            # Average models, but not hours
        #X = X.reshape(X.shape[0],np.prod(X.shape[1:]))                   # Reshape into (n_examples,n_features)
        return T, X
'''
Load csv test/train data splitting out times.
'''
def load_csv_data(path):
        data = np.loadtxt(path,delimiter=',',dtype=float,skiprows=1)
        Y = data[:,1:]
        return Y

'''
Saves out to a csv.
Just reads in the example csv and writes out 
over the zeros with the model predictions.
'''
def save_submission(preds,submit_name,data_dir):
        fexample = open(os.path.join(data_dir,'sampleSubmission.csv'))
        fout     = open(os.path.join(data_dir,submit_name),'wb')
        fReader = csv.reader(fexample,delimiter=',', skipinitialspace=True)
        fwriter = csv.writer(fout)
        for i,row in enumerate(fReader):
                if i == 0:
                        fwriter.writerow(row)
                else:
                        row[1:] = preds[i-1]
                        fwriter.writerow(row)
        fexample.close()
        fout.close()

'''
Get the average mean absolute error for models trained on cv splits
'''
def cv_loop(X, y, model, N):
    MAEs = 0
    for i in range(N):
        X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=.20, random_state = i*SEED)
        model.fit(X_train, y_train)
        preds = model.predict(X_cv)
        mae = metrics.mean_absolute_error(y_cv,preds)
        print "MAE (fold %d/%d): %f" % (i + 1, N, mae)
        MAEs += mae
    return MAEs/N

'''
Get the files to use
'''
def files_To_Use(files_to_use):
    if files_to_use == 'all':
            files_to_use = ['dswrf_sfc','dlwrf_sfc','uswrf_sfc','ulwrf_sfc','ulwrf_tatm','pwat_eatm','tcdc_eatm','apcp_sfc','pres_msl','spfh_2m','tcolc_eatm','tmax_2m','tmin_2m','tmp_2m','tmp_sfc']
    return files_to_use

# <codecell>

'''
Load training and testing data
'''
def load_Training_Data(data_dir='./data/',files_to_use='all'):
    files_to_use = files_To_Use(files_to_use)
    train_sub_str = '_latlon_subset_19940101_20071231.nc'

    print 'Loading training data...'
    trainT, trainX = load_GEFS_data(data_dir+'train/',files_to_use,train_sub_str)
    trainY = load_csv_data(os.path.join(data_dir,'train.csv'))
    print 'Training data shape',trainX.shape,trainY.shape
    
    # Load expected solar values for given days
    solarData = np.loadtxt("../../Data/TSiteSolars.csv",delimiter=',',dtype=float,skiprows=1)
    augmentX = solarData[:,1:]
    
    return trainT, trainX, trainY, augmentX

'''
Load testing data
'''
def load_Testing_Data(data_dir='./data/',files_to_use='all'):
    files_to_use = files_To_Use(files_to_use)
    test_sub_str = '_latlon_subset_20080101_20121130.nc'

    print 'Loading test data...'
    testT, testX = load_GEFS_data(data_dir+'test/',files_to_use,test_sub_str)
    print 'Test data shape',testX.shape

    # Load expected solar values for given days
    solarData = np.loadtxt("../../Data/TSiteSolarsTest.csv",delimiter=',',dtype=float,skiprows=1)
    augmentX = solarData[:,1:]
    
    return testT, testX, augmentX

'''
Get Times from data
'''
def load_Times(data_dir='./data/',file_to_use='dswrf_sfc',set='Train'):
    if(set=='Train'):
        sub_str = '_latlon_subset_20080101_20121130.nc'
    elif(set=='Test'):
        sub_str = '_latlon_subset_19940101_20071231.nc'

    print 'Loading test data...'
    data = load_GEFS_Object(data_dir+'test/',file_to_use,sub_str)
    return data.variables['intTime'][:]

'''
Fit the Model with training data
'''
def fit_Model(trainX, trainY, model, N):
    
    print 'Finding best regularization value for alpha...'
    alphas = np.logspace(-3,1,8,base=10) # List of alphas to check
    #alphas = np.array(( 0.1, 0.2, 0.3, 0.4, 0.5, 0.6 ))
    maes = []
    for alpha in alphas:
            model.alpha = alpha
            mae = cv_loop(trainX,trainY,model,N)
            maes.append(mae)
            print 'alpha %.4f mae %.4f' % (alpha,mae)
    best_alpha = alphas[np.argmin(maes)]
    print 'Best alpha of %s with mean average error of %s' % (best_alpha,np.min(maes))
    
    print 'Fitting model with best alpha...'
    model.alpha = best_alpha
    model.fit(trainX,trainY)
    
    return times, trainX, trainY

'''
Test the Model with testing data
'''
def test_Model(testX, model):
    
    print 'Predicting...'
    preds = model.predict(testX)
    
    return preds

def pcaStuff(inputX,inputY):
    pca = PCA(copy=True, n_components=50, whiten=False)
    pca.fit(inputX,inputY)
    outputX = pca.transform(inputX)
    return pca, outputX

def getSolarRadiation(lat, lon, time):
    altitude = ps.GetAltitude(lat, lon, time)
    return ps.radiation.GetRadiationDirect(time, altitude)

def getDailySolar(lat, lon, day):
    dailySolar = 0
    for minute in range(0, 24*60, 5):
        time = day + datetime.timedelta(minutes=minute)
        dailySolar += getSolarRadiation(lat, lon, time)
    dailySolar *= 300 #3600(joules per watt)*(5.0/60.0)(sample time)
    return dailySolar

def getMaxSolar(lat, lon, times):
    maxSolarEnergy = []
    for day in times:
        day = datetime.datetime.strptime(str(day),'%Y%m%d')
        maxSolarEnergy.append(getDailySolar(lat, lon, day))
    return maxSolarEnergy

def genSiteSolar(data_dir='./data/', saveFileName='SiteSolars.csv'):
    with open(os.path.join(data_dir,saveFileName), 'wb') as csvfile:
        spamwriter = csv.writer(csvfile)    
        times,trainY = load_csv_data(os.path.join(data_dir,'test.csv'))
        df = pd.read_csv(os.path.join(data_dir,'station_info.csv')).T
        dates = ['Date'] + list(times)
        spamwriter.writerow(dates)
        for i, station in df.iteritems():
            maxSolar = getMaxSolar(station['nlat'], station['elon'], times)
            stid = [station['stid']] + maxSolar
            spamwriter.writerow(stid)
            print('Job: ', i, ' Station: ', station['stid'])
    
    a = izip(*csv.reader(open(os.path.join(data_dir,saveFileName), "rb")))
    csv.writer(open(os.path.join(data_dir,'T'+saveFileName), "wb")).writerows(a)

