import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation

from rootpy.plotting import Hist2D, Hist3D, Graph, Canvas
from rootpy.io import root_open
from root_numpy import root2array, fill_graph


def fit(filename1, treename1, filename2, treename2, inputsname, EventSelectionCuts):
    # Reading inputs from ROOT tree     
    ninputs = len(inputsname)
    #need to apply all cuts at once 
    data1 = root2array(filename1, treename=treename1, branches=inputsname, selection=EventSelectionCuts, include_weight=True)
    data2 = root2array(filename2, treename=treename2, branches=inputsname, selection=EventSelectionCuts, include_weight=True)
    # change to proper array display
    data1 = data1.view((np.float64, len(data1.dtype.names)))  
    data2 = data2.view((np.float64, len(data2.dtype.names)))
    # split the weights from the data
    weight1 = data1[:, [ninputs]].astype(np.float32).ravel() 
    data1 =  data1[:, range(ninputs)].astype(np.float32) 
    weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 
    data2 =  data2[:, range(ninputs)].astype(np.float32) 
    # Creating target class arrays
    class1 = np.zeros((data1.shape[0],))
    class2 = np.ones((data2.shape[0],))
    # Merging datasets
    inputs = np.concatenate((data1, data2))
    classes = np.concatenate((class1, class2))
    weights = np.concatenate((weight1,weight2))
    # Split events in a training sample and a test sample (60% used for training), 
    #40% is used for testing evaluating the classifier
    # add the weights as well
    inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test = cross_validation.train_test_split(inputs, classes, weights, test_size=0.4, random_state=0)
    # Fit and test classifier (BDT with gradient boosting)
    # Default training parameters are used
    clf = GradientBoostingClassifier()
    clf.fit(inputs_train, targets_train, sample_weight=weights_train)
    #print 'Accuracy on test sample:', clf.score(inputs_test, targets_test, sample_weight=weights_test)

    #get the class probabilities 
    class_proba = clf.predict_proba(inputs_test)
    #plotting
    print "plotting...."
    #create a graph
    graph = Graph(len(class_proba))
    #extract the probability for W+jet and the transverse mass     
    prob_W_jet = class_proba[:, 0].astype(np.float32).ravel() 
    mt = inputs_test[:, 0].astype(np.float32).ravel()
    #stack W+jet and mt together
    mt_vs_Wjet = np.column_stack((mt,prob_W_jet))
    #create a root file and fill it with the graph P(W+jet) vs mt
    root_open("plots/outputfile.root", 'recreate')
    fill_graph(graph,mt_vs_Wjet)
    graph.Write("AP")
    
    #Create Canvas make some cosmetics on the plot and save it as png
    c = Canvas()
    graph.SetTitle("This is the title of the Plot")
    graph.GetXaxis().SetTitle("mt")
    graph.GetXaxis().SetRangeUser(0.,250.)
    graph.GetYaxis().SetTitle("P(W+jet)")
    graph.SetMarkerColor(4)
    graph.Draw("AP")
    c.SaveAs("plots/firstplot.png")
   
if __name__=='__main__':
#   use 2 different classes, the W+jet class and the QCD class (add probably more later)
    file_w =   '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/WJetsToLNu_LO/H2TauTauTreeProducerTauMu/tree.root'
#               /afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/*            /H2TauTauTreeProducerTauMu/tree.root
    file_qcd = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/QCD_Mu15/H2TauTauTreeProducerTauMu/tree.root'
#started with 2 branches, 1) the transverse mass mt and 2) the decay mode with 2 leptons in
#the final state l2_decayMode.   
    inputs = ['mt','l2_decayMode']
#Defintion of different event cuts as proposed on the Twiki
    LeptonVetos = ['veto_dilepton<0.5','veto_thirdlepton<0.5','veto_otherlepton<0.5']
    METDataRemoval = ['!(met_pt < 0.15 && met_phi > 0. && met_phi < 1.8)']
    OppositeCharge = ['l1_charge*l2_charge<0']
    MuCuts = ['l1_pt>19', 'l1_reliso05<0.1', 'l1_muonid_medium>0.5']
    HadTauCuts = ['l2_againstMuon3>1.5', 'l2_againstElectronMVA6>0.5', 'l2_decayModeFinding', 'l2_pt>20']    
    AntiIsoTau = ['l2_byIsolationMVArun2v1DBoldDMwLT<3.5'] 
#merge all event cuts in one logic command of event selection cuts
    EventSelections = ' && '.join(np.concatenate((LeptonVetos,METDataRemoval,OppositeCharge,MuCuts,HadTauCuts,AntiIsoTau)))

    fit(file_w, 'tree', file_qcd, 'tree', inputs, EventSelections)
   