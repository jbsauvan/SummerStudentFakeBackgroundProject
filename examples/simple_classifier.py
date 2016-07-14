import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation

from rootpy.plotting import Hist2D, Hist3D
from rootpy.io import root_open
from root_numpy import root2array


def fit(filename1, treename1, filename2, treename2, inputsname):
    sample_info = pickle.load(open('/afs/cern.ch/user/j/jsauvan/public/Htautau/mc_info_76.pck', 'rb'))
    # sample weight = 1/(sample luminosity)
    weight_w = sample_info['W']['XSec']/sample_info['W']['SumWeights'] 
    weight_qcd = sample_info['QCD']['XSec']/sample_info['QCD']['SumWeights']
    print weight_w, weight_qcd
    # Reading inputs from ROOT tree
    data1 = root2array(filename1, treename=treename1, branches=inputsname)
    data2 = root2array(filename2, treename=treename2, branches=inputsname)
    data1 = data1.view((np.float64, len(data1.dtype.names)))
    data2 = data2.view((np.float64, len(data2.dtype.names)))
    # Creating target class arrays
    class1 = np.zeros((data1.shape[0],))
    class2 = np.ones((data2.shape[0],))
    # Merging datasets
    inputs = np.concatenate((data1, data2))
    classes = np.concatenate((class1, class2))
    # Split events in a training sample and a test sample (60% used for training)
    inputs_train, inputs_test, targets_train, targets_test = cross_validation.train_test_split(inputs, classes, test_size=0.4, random_state=0)
    # Fit and test classifier (BDT with gradient boosting)
    # Default training parameters are used
    clf = GradientBoostingClassifier()
    clf.fit(inputs_train, targets_train)
    print 'Accuracy on test sample:', clf.score(inputs_test, targets_test)


if __name__=='__main__':
    file_w = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/WJetsToLNu_LO/H2TauTauTreeProducerTauMu/tree.root'
    file_qcd = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/QCD_Mu15/H2TauTauTreeProducerTauMu/tree.root'
    inputs = ['mt','l2_decayMode']
    fit(file_w, 'tree', file_qcd, 'tree', inputs)
