import csv
import os
import math
import netCDF4 as nc
import numpy as np
import pandas as pd
import cPickle as pickle
import datetime
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import ensemble
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
            T, X, infos = load_GEFS_file(directory,files_to_use[i],file_sub_str)
            X = np.expand_dims(X, axis=1)
        else:
            T, X_new, infos = load_GEFS_file(directory,files_to_use[i],file_sub_str)
            X_new = np.expand_dims(X_new, axis=1)
            X = np.hstack((X,X_new))
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],X.shape[3],X.shape[4]*X.shape[5])
    X = np.swapaxes(X,1,4)
    return T, X, infos

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
    infos = []
    lats = data.variables['lat'][:]
    lons = data.variables['lon'][:]-360
    for lat in lats:
        for lon in lons:
            info = {'lat':lat,'lon':lon}
            infos.append(info)
    #X = X.reshape(X.shape[0],55,4,10)                               # Reshape to merge sub_models and time_forcasts
    #X = np.mean(X,axis=1)                                            # Average models, but not hours
    #X = X.reshape(X.shape[0],np.prod(X.shape[1:]))                   # Reshape into (n_examples,n_features)
    return T, X, infos

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
    trainT, trainX, infos = load_GEFS_data(data_dir+'train/',files_to_use,train_sub_str)
    trainY = load_csv_data(os.path.join(data_dir,'train.csv'))
    print 'Training data shape',trainX.shape,trainY.shape
    
    # Load expected solar values for given days
    solarData = np.loadtxt("../../Data/TSiteSolars.csv",delimiter=',',dtype=float,skiprows=1)
    augmentX = solarData[:,1:]
    
    return trainT, trainX, trainY, infos, augmentX

'''
Load testing data
'''
def load_Testing_Data(data_dir='./data/',files_to_use='all'):
    files_to_use = files_To_Use(files_to_use)
    test_sub_str = '_latlon_subset_20080101_20121130.nc'

    print 'Loading test data...'
    testT, testX, infos = load_GEFS_data(data_dir+'test/',files_to_use,test_sub_str)
    print 'Test data shape',testX.shape
   
    # Load expected solar values for given days
    solarData = np.loadtxt("../../Data/TSiteSolarsTest.csv",delimiter=',',dtype=float,skiprows=1)
    augmentX = solarData[:,1:]
    
    return testT, testX, infos, augmentX

'''
Load station info
'''
def load_Station_Info(data_dir='./data/'):

    print 'Loading station info...'
    infos = []
    df = pd.read_csv(os.path.join(data_dir,'station_info.csv')).T
    for i, station in df.iteritems():
        temp = {'stid': station['stid'],
                'lat': station['nlat'],
                'lon': station['elon'],
                'elev': station['elev'],
                'index':i}
        infos.append(temp)
    return infos

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

def dayOfYear(T):
    '''Get the day of year time vector'''
    days = []
    for t in T:
        day = datetime.datetime.strptime(str(t),'%Y%m%d%H')
        day = day.timetuple().tm_yday
        days.append(day)
    days = np.array(days)
    return days

def compexDayOfYear(T):
    '''Get the compex day of year time vector'''
    T = dayOfYear(T)
    real = np.sin((T/365.0)*2*np.pi)
    imaginary = np.cos((T/365.0)*2*np.pi)
    compexDays = np.vstack((real,imaginary)).T
    return compexDays

def monthOfYear(T):
    '''Get the day of year time vector'''
    months = []
    for t in T:
        month = datetime.datetime.strptime(str(t),'%Y%m%d%H')
        month = month.timetuple().tm_mon
        months.append(month)
    months = np.array(months)
    return months

def compexMonthOfYear(T):
    '''Get the compex day of year time vector'''
    T = monthOfYear(T)
    real = np.sin((T/12.0)*2*np.pi)
    imaginary = np.cos((T/12.0)*2*np.pi)
    compexMonths = np.vstack((real,imaginary)).T
    return compexMonths

class stationClass:

    def __init__(self, info, model, config):
        '''Create a Station Model'''
        # Store the info about the station
        self.info   = info
        
        # Store the regression model used
        # for each weather prediction model
        self.models = [model]*11
        
        # Store the configuration of the station
        self.config = config
        
        # Store the GEFS points to be used
        self.GEFSs  = self.getGEFSs()
        
    
    def fit(self, X_train, y_train, wModels=None, mod_train=None):
        '''Fit the Station Model'''
        X_train = self.filterGEFSX(X_train)
        y_train = self.filterGEFSy(y_train)
        if (wModels == None):
            wModels = range(11)
        for i,index in enumerate(wModels):
            print "     Fitting Model:", i
            model = self.models[index]
            X = X_train[:,:,i]
            X = X.reshape(X.shape[0],np.prod(X.shape[1:]))
            if mod_train != None:
                X = np.hstack((X,mod_train))
            model.fit(X,y_train)
         
    def predict(self, X_test, wModels=None, mod_train=None):
        '''Predict with the Station Model'''
        X_test = self.filterGEFSX(X_test)
        if (wModels == None):
            wModels = range(11)
        for i,index in enumerate(wModels):
            print "     Predicting Model:", index
            X = X_test[:,:,index]
            X = X.reshape(X.shape[0],np.prod(X.shape[1:]))
            if mod_train != None:
                X = np.hstack((X,mod_train))
            model = self.models[index]
            if i == 0:
                prediction = model.predict(X)
            else:
                prediction = np.vstack((prediction,model.predict(X)))
        return prediction
    
    def getDistance(self,station, GEFS):
        '''Find the distance from the Station'''
        lat1, lon1 = [station['lat'], station['lon']]
        lat2, lon2 = [GEFS['lat'], GEFS['lon']]
        radius = 6371 # km
    
        dlat = math.radians(lat2-lat1)
        dlon = math.radians(lon2-lon1)
        a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
            * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        d = radius * c
    
        return d
    
    def getGEFSs(self):
        '''Get only the GEFS that are the N closest for config'''
        distances = []
        for index, GEFS in enumerate(self.info['GEFSs']):
            distance = self.getDistance(self.info['station'], GEFS)
            distances.append([index,distance])
        distances = sorted(distances,key=lambda l:l[1])
        GEFSs = []
        N = self.config['nGEFS']
        for index, GEFS in distances[0:N]:
            GEFSs.append(index)
        return GEFSs
    
    def filterGEFSX(self, X):
        '''Filter X data to just used GEFS points'''
        for i, GEFS in enumerate(self.GEFSs):
            if i == 0:
                X_new = X[:,GEFS]
                X_new = np.expand_dims(X_new, axis=1)
            else:
                X_temp = X[:,GEFS]
                X_temp = np.expand_dims(X_temp, axis=1)
                X_new = np.hstack((X_new,X_temp))
        return X_new
    
    def filterGEFSy(self, y):
        '''Filter y data to just used GEFS points'''
        y_new = y[:,self.info['station']['index']]
        return y_new

def makeStations(station_infos, GEFS_infos, model, config):
    stations = []
    for station_info in station_infos:
        info = {'station':station_info,
                'GEFSs':GEFS_infos}
        temp_station = stationClass(info, model, config)
        stations.append(temp_station)
    return stations
    
def fitStations(stations, X_train, y_train, wModels=None, mod_train=None):
    for i, station in enumerate(stations):
        print "Fitting Station", i
        station.fit(X_train, y_train, wModels, mod_train)
    return stations
    
def predictStations(stations, X_test, wModels=None, mod_train=None):
    for i, station in enumerate(stations):
        print "Predict Station", i
        prediction = station.predict(X_test,wModels,mod_train)
        if i == 0:
            prediction = np.expand_dims(prediction,0)
            predictions = prediction
        else:
            prediction = np.expand_dims(prediction,0)
            predictions = np.vstack((finalPrediction,prediction))
    return predictions

def savePickle(data, data_dir, file_name):
    pickle.dump(data, open(os.path.join(data_dir,file_name), "wb"))
    
def loadPickle(data_dir, file_name):
    data = pickle.load(open(os.path.join(data_dir,file_name), "rb"))
    return data

def saveModels(stations, data_dir, file_name):
    stationModels = []
    for station in stations:
        stationModels.append(station.models)
    savePickle(stationModels, data_dir, file_name)
    
def loadModels(stations, data_dir, file_name):
    stationModels = loadPickle(data_dir, file_name)
    for i, station in enumerate(stations):
        station.models = stationModels[i]
    return stations
