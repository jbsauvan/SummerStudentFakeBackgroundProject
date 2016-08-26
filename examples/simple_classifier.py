from ROOT import TH1D
import numpy as np
import pickle
import copy, time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation

from rootpy.plotting import Hist, Hist2D, Hist3D, Graph, Canvas
from rootpy.io import root_open
from root_numpy import root2array, fill_graph, fill_hist


def fit(filename1, treename1, filename2, treename2, inputsname, EventSelectionCuts, weight_name="weight", accuracy=False, eventdist=False) :#, plotflag=False, weight_name="weight"):
    #load the file to obtain the proper scaling of QCD and Wjet
    sample_info = pickle.load(open('/afs/cern.ch/user/j/jsauvan/public/Htautau/mc_info_76.pck', 'rb'))
    # sample weight = 1/(sample luminosity) output is a float
    weight_w = sample_info['W']['XSec']/sample_info['W']['SumWeights'] 
    weight_qcd = sample_info['QCD']['XSec']/sample_info['QCD']['SumWeights']

    #1/weight_qcd is the luminosity of the MC simulated QCD events
    rescale_Wjet = weight_w/weight_qcd 
    print "The overall weight is : ", rescale_Wjet    
    ninputs = len(inputsname)
    # merge inputsname with weight_name to branch_names
    branch_names = copy.copy(inputsname)
    branch_names.append(weight_name)
    # Reading inputs from ROOT tree     
    #need to apply all cuts at once, the data1/2 are of the matrix format 
    #with each row of the from (mt,decay channel, weight)    
    data1 = root2array(filename1, treename=treename1, branches=branch_names, selection=EventSelectionCuts[0]) #Wjet
    data2 = root2array(filename2, treename=treename2, branches=branch_names, selection=EventSelectionCuts[1]) #QCD
    # change to proper array display
    data1 = data1.view((np.float64, len(data1.dtype.names)))  
    data2 = data2.view((np.float64, len(data2.dtype.names)))
    #extracting weights
    weight1 = data1[:, [ninputs]].astype(np.float32).ravel() 
    weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 
    #proper rescaling of the QCD and Wjet event weights
    weight1 = np.multiply(weight1,weight_w/weight_qcd)    
    #weight in QCD (gives same result):   weight2 = np.multiply(weight2,weight_qcd/weight_w) 
    #the weight was extracted above data1/2_new is a matrix with each 
    #row of the form (mt, decay channel)
    data1_new = data1[:, range(ninputs)].astype(np.float32)
    data2_new = data2[:, range(ninputs)].astype(np.float32) 
    #define classes for training    
    class1 = np.zeros((data1_new.shape[0],))
    class2 = np.ones((data2_new.shape[0],))
    # Merging datasets
    inputs = np.concatenate((data1_new, data2_new))
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
    if (accuracy == True) :
        print 'Accuracy on test sample:', clf.score(inputs_test, targets_test, sample_weight=weights_test)
    if (eventdist == True) :
        simple_hist(data1[:, 0],"Wjet_weighted_EventDistribution_all_channels", data_weights = data1[:, 2], Nxbins=200, X_axis_range=[0.,200.])
        simple_hist(data2[:, 0],"QCD_weighted_EventDistribution_all_channels",  data_weights = data2[:, 2], Nxbins=200, X_axis_range=[0.,200.])
        decay_modes = [0.,1.,10.]
        for j in decay_modes :
            weight1_dec = np.array(filter(lambda x: x[1] == j, data1))
            #weight1_red = np.array(filter(lambda x: x[1] == j, data1[:,2]))
            weight2_dec = np.array(filter(lambda x: x[1] == j, data2))
            #weight2_red = np.array(filter(lambda x: x[1] == j, data2[:,2]))
            simple_hist(weight1_dec[:, 2],"Wjet_WeightDistribution_Channel_"+str(j), Nxbins=200, X_axis_range=[0.,3.5])
            simple_hist(weight2_dec[:, 2],"QCD_WeightDistribution_Channel_"+str(j),  Nxbins=200, X_axis_range=[0.,3.5])
            
    return clf, inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test







    
    
    
    
    
    
    

def simple_hist(array, name, data_weights = 1, Nxbins=100, X_axis_range=[0.,1.]) :
    if (type(data_weights) == int) :
        data_weights = np.ones(len(array))
    #h = Hist(30,40,200,title=name,markersize=0)
    h = TH1D("h1","simple_hist_"+name,Nxbins ,X_axis_range[0] ,X_axis_range[1])
    root_open("plots/ROOTfiles/H1_"+name+".root", 'recreate')
    fill_hist(h,array,weights=data_weights)
    h.Write()
    #create Canvas and save the plot as png
    c = Canvas()
    h.Draw("HIST")
    c.SaveAs("plots/H1_"+name+".png")

def txt_output(array, name) :
    text_file = open("txtOutput/"+name+".txt", "w")
    l = np.round(len(array),-1)/10
    per = 10  
    counter = 1
    for i in array :
        if (counter == 0 & l) :
            print per, " pecent is done"
        text_file.write(str(i)+"\n")
        counter = counter + 1
    text_file.close()
    print name, ".txt was created."

 
def simple_graph(x,y,name,axis_range=[0.,200.]) :
    #create a graph and save it as root file
    graph = Graph(len(x),name)
    #create a root file and fill it with the graph P(W+jet) vs mt
    root_open("plots/ROOTfiles/G_"+name+".root", 'recreate')
    fill_graph(graph,np.column_stack((x,y)))
    graph.Write()
    #create Canvas and save the plot as png
    c = Canvas()
    graph.SetTitle(name)
    graph.SetMarkerSize(0.3)
    graph.GetXaxis().SetRangeUser(axis_range[0],axis_range[1])
    graph.Draw("AP")
    c.SaveAs("plots/G_"+name+".png")


def TwoClassProba (classifier, inputs, name) :
    #get the class probabilities
    class_proba = classifier.predict_proba(inputs)     
    #extract the probablilty of the first class. The prob of the 2nd one is 1-Prob(1st class)
    #because we deal with only 2 classes
    prob = class_proba[:, 0].astype(np.float32).ravel() 
    #extract the mt
    mt = inputs[:, 0].astype(np.float32).ravel()
    #draw the proability distribution
    simple_graph(mt, prob, name)



def fit_QCD_data(filename1, treename1, filename2, treename2, inputsname, EventSelectionCuts, weight_name="weight", accuracy=False, eventdist=False) :#, plotflag=False, weight_name="weight"):
        
#==============================================================================
    #input variables for the subtraction
    QCDSubtrNames = ['WJetsToLNu_LO', 'DYJetsToLL_M50_LO_ext1', 'TT_pow_ext3', 'T_tWch', 'TBar_tWch', 'ZZTo4L', 'ZZTo2L2Q', 'WZTo2L2Q', 'WZTo1L3Nu', 'WZTo3LNu_amcatnlo', 'WZTo1L1Nu2Q', 'WWTo1L1Nu2Q', 'VVTo2L2Nu']
    XSecNames = ['W', 'ZJ', 'TT', 'T_tWch', 'TBar_tWch', 'ZZTo4L', 'ZZTo2L2Q', 'WZTo2L2Q', 'WZTo1L3Nu', 'WZTo3L', 'WZTo1L1Nu2Q', 'WWTo1L1Nu2Q', 'VVTo2L2Nu']
    
    #file_qcd_data = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/SingleMuon_Run2015D_16Dec/H2TauTauTreeProducerTauMu/tree.root'
#==============================================================================
    #load the file to obtain the proper scaling of QCD and Wjet
    sample_info = pickle.load(open('/afs/cern.ch/user/j/jsauvan/public/Htautau/mc_info_76.pck', 'rb'))
    
    #luminosity of the data set of QCD events      
    luminosity_QCD_data = 2260.0    
    RescaleWjet = luminosity_QCD_data*sample_info['W']['XSec']/sample_info['W']['SumWeights']
    
    print "need to rescale Wjet events by ", RescaleWjet
    
           
    ninputs = len(inputsname)
    # merge inputsname with weight_name to branch_names
    branch_names = copy.copy(inputsname)
    branch_names.append(weight_name)
    # Reading inputs from ROOT tree     
    #need to apply all cuts at once, the data1/2 are of the matrix format 
    #with each row of the from (mt,decay channel, weight)    
    data1 = root2array(filename1, treename=treename1, branches=branch_names, selection=EventSelectionCuts[0]) #Wjet
    data2 = SubEventsFromQCD(filename2, 'tree', EventSelectionCuts[1], QCDSubtrNames, 'tree', inputsname, XSecNames)#,drawflag=True)
    
    
    
#    data2 = copy.copy()#QCD data
    
    
    
    # change to proper array display
    data1 = data1.view((np.float64, len(data1.dtype.names)))  
    
    #data2 = data2.view((np.float64, len(data2.dtype.names)))
    #extracting weights
    weight1 = data1[:, [ninputs]].astype(np.float32).ravel() 
    weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 
    #proper rescaling of the QCD and Wjet event weights  
    
    
    weight1 = np.multiply(weight1,RescaleWjet)
    
    #the weight was extracted above data1/2_new is a matrix with each 
    #row of the form (mt, decay channel)
    data1_mt_dec = data1[:, range(ninputs)].astype(np.float32)
    data2_mt_dec = data2[:, range(ninputs)].astype(np.float32)
    #txt_output(data2_mt_dec,"data2_mt_dec")    
    
    #define classes for training    
    class1 = np.zeros((data1_mt_dec.shape[0],))
    class2 = np.ones((data2_mt_dec.shape[0],))
    # Merging datasets
    inputs = np.concatenate((data1_mt_dec, data2_mt_dec))
    classes = np.concatenate((class1, class2))
    weights = np.concatenate((weight1,weight2))
    # Split events in a training sample and a test sample (60% used for training), 
    #40% is used for testing evaluating the classifier
    # add the weights as well
    inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test = cross_validation.train_test_split(inputs, classes, weights, test_size=0.4, random_state=0)
    # Fit and test classifier (BDT with gradient boosting)
    # Default training parameters are used
    clf = GradientBoostingClassifier()
    """
    time test    
    """    
    # CPU time
    t0_CPU = time.clock()

    #wall time
    t0_wall = time.time()
    clf.fit(inputs_train, targets_train, sample_weight=weights_train)
    t_CPU = time.clock() - t0_CPU
    t_wall = time.time() - t0_wall
    txt_output(["The CPU time for 60 % training was " + str(t_CPU)],"CPU_time")
    txt_output(["The wall time for 60 % training was " + str(t_wall)],"wall_time")
    
    
    
    if (accuracy == True) :
        print 'Accuracy on test sample:', clf.score(inputs_test, targets_test, sample_weight=weights_test)
        #print the importance of the 2 variables mt and decay channel
        print "The importance of the variables (mt,decay channel) is: ", clf.feature_importances_
    if (eventdist == True) :
        simple_hist(data1_mt_dec[:, 0],"Wjet_all_channels", data_weights = weight1, Nxbins=200, X_axis_range=[0.,200.])
        simple_hist(data2_mt_dec[:, 0],"QCD_data_all_channels",  data_weights = weight2, Nxbins=200, X_axis_range=[0.,200.])
        decay_modes = [0.,1.,10.]
        for j in decay_modes :
            weight1_dec = np.array(filter(lambda x: x[1] == j, data1))
            weight2_dec = np.array(filter(lambda x: x[1] == j, data2))
            simple_hist(weight1_dec[:, 2],"Wjet_Channel_"+str(j), Nxbins=200, X_axis_range=[0.,3.5])
            simple_hist(weight2_dec[:, 2],"QCD_data_subtracted_Channel_"+str(j),  Nxbins=200, X_axis_range=[-10.,0.0])
            simple_hist(weight2_dec[:, 2],"QCD_data_Channel_"+str(j),  Nxbins=200, X_axis_range=[0.0,3.0])
            
    return clf, inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test





def SubEventsFromQCD (filename, treename, EventSelectionCuts, SubFileNames, SubTreeNames, inputsname, XSectionNames, weight_name="weight",drawflag=False) :
    ninputs = len(inputsname)
    # merge inputsname with weight_name to branch_names
    branch_names = copy.copy(inputsname)
    branch_names.append(weight_name)
    # Reading inputs from ROOT tree     
    #need to apply all cuts at once, the data1/2 are of the matrix format 
    #with each row of the from (mt,decay channel, weight)    
    data1 = root2array(filename, treename=treename, branches=branch_names, selection=EventSelectionCuts) #QCD data
    # change to proper array display
    data1 = data1.view((np.float64, len(data1.dtype.names)))  
    #extracting weights
    weight1 = data1[:, [ninputs]].astype(np.float32).ravel() 
    

    data = copy.copy(data1[:,0])  
    decayChannel = copy.copy(data1[:,1])    
    weight = copy.copy(weight1)

    if (drawflag==True) :
        simple_hist(data1[:, 0],"QCD_weighted_EventDistribution_all_channels", data_weights = weight1, Nxbins=200, X_axis_range=[0.,200.])


    for j in range(len(XSectionNames)) :  
        sub_file =  '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/'+ SubFileNames[j] +'/H2TauTauTreeProducerTauMu/tree.root'        
        #luminosity of the data set of QCD events      
        luminosity_QCD_data = 2260.0
        #Extrapolationfaktor for going from same sign region to opposite sign region
        SS_to_OS = 1.06
        
        #load the file to obtain the proper scaling of QCD and Wjet
        sample_info = pickle.load(open('/afs/cern.ch/user/j/jsauvan/public/Htautau/mc_info_76.pck', 'rb'))
        # sample weight = 1/(sample luminosity) output is a float
        RescaleFactor = SS_to_OS*luminosity_QCD_data*sample_info[XSectionNames[j]]['XSec']/sample_info[XSectionNames[j]]['SumWeights'] 
        print "The rescale factor of the ", SubFileNames[j], " events is ", RescaleFactor 
        
        data2 = root2array(sub_file, treename=SubTreeNames, branches=branch_names, selection=EventSelectionCuts) #QCD subtraction
        data2 = data2.view((np.float64, len(data2.dtype.names)))    
        weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 
        weight2 = np.multiply(weight2,RescaleFactor)
                
        #subtract
        data = np.concatenate((data,data2[:,0]))
        decayChannel = np.concatenate((decayChannel, data2[:,1]))
        weight = np.concatenate((weight, np.multiply(weight2,-1.))) 
        
        if (drawflag==True) :
            #event distribution of the subtracted sample
            simple_hist(data2[:, 0],SubFileNames[j]+"_weighted_EventDistribution_all_channels", data_weights = weight2, Nxbins=200, X_axis_range=[0.,200.])
            #event distribution of QCD after the subtraction            
            simple_hist(data,"Subtracted_"+str(j+1)+"_Events", data_weights = weight, Nxbins=200, X_axis_range=[0.,200.])

    return np.transpose(np.vstack((data, decayChannel, weight)))

   
if __name__=='__main__':
    #use 2 different classes, the W+jet class and the QCD class (add probably more later)
    file_w =   '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/WJetsToLNu_LO/H2TauTauTreeProducerTauMu/tree.root'
              # /afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/*            /H2TauTauTreeProducerTauMu/tree.root
    file_qcd = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/QCD_Mu15/H2TauTauTreeProducerTauMu/tree.root'
    #QCD data
    file_qcd_data = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/SingleMuon_Run2015D_16Dec/H2TauTauTreeProducerTauMu/tree.root'
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


    #same sign region for QCD. This control region is enriched in QCD
    SameCharge = ['l1_charge*l2_charge<0']
    EventSelections_QCD = ' && '.join(np.concatenate((LeptonVetos,METDataRemoval,SameCharge,MuCuts,HadTauCuts,AntiIsoTau)))



    """
    QCD subtraction samples
    """
    QCDSubtrNames = ['WJetsToLNu_LO']#, 'DYJetsToLL_M50_LO_ext1', 'TT_pow_ext3', 'T_tWch', 'TBar_tWch', 'ZZTo4L', 'ZZTo2L2Q', 'WZTo2L2Q', 'WZTo1L3Nu', 'WZTo3LNu_amcatnlo', 'WZTo1L1Nu2Q', 'WWTo1L1Nu2Q', 'VVTo2L2Nu']
    XSecNames = ['W']#, 'ZJ', 'TT', 'T_tWch', 'TBar_tWch', 'ZZTo4L', 'ZZTo2L2Q', 'WZTo2L2Q', 'WZTo1L3Nu', 'WZTo3L', 'WZTo1L1Nu2Q', 'WWTo1L1Nu2Q', 'VVTo2L2Nu']


    #SubEventsFromQCD(file_qcd_data, 'tree', EventSelections_QCD, QCDSubtrNames, 'tree', inputs, XSecNames)    
    f = fit_QCD_data(file_w, 'tree', file_qcd_data, 'tree', inputs, [EventSelections,EventSelections_QCD])#,accuracy=True,eventdist=True)#,accuracy=True)


#==============================================================================
#    decay_modes = [0.,1.,10.]
#    for j in decay_modes :
#        data_reduced = np.array(filter(lambda x: x[1] == j, f[2]))
#        TwoClassProba(f[0],data_reduced,"Wjet_proba_vs_mt_DecMode_"+str(j)+"_QCD_data_SUBTRACTED")
#        #txt_output(data_reduced, "inputs_test_DecMode_"+str(j))
#==============================================================================





#    f = fit(file_w, 'tree', file_qcd, 'tree', inputs, [EventSelections,EventSelections])#,accuracy=True)
 
#==============================================================================
#    decay_modes = [0.,1.,10.]
#    for j in decay_modes :
#        data_reduced = np.array(filter(lambda x: x[1] == j, f[2]))
#        TwoClassProba(f[0],data_reduced,"Wjet_proba_vs_mt_DecMode_"+str(j))
#        #txt_output(data_reduced, "inputs_test_DecMode_"+str(j))
#==============================================================================


#============================================================================== 
#    step = 0.01
#    decay_mode = 10.
#    mt_generated = np.arange(0.,200.,step)
#    dmode = np.add(np.zeros(len(mt_generated)),decay_mode).astype(np.float32).ravel()
#    GBC_input = np.column_stack((mt_generated,dmode))
#    TwoClassProba(f[0],GBC_input,"generated_mt_DecMode_"+str(decay_mode)+"_stepsize_"+str(step))
#==============================================================================

        
#==============================================================================
#     txt_output(f[1], "inputs_train")
#     txt_output(f[2], "inputs_test")
#     txt_output(f[3], "targets_train")
#     txt_output(f[4], "targets_test")
#     txt_output(f[5], "weights_train")
#     txt_output(f[6], "weights_test")
#==============================================================================
    
    #f = fit(file_w, 'tree', file_qcd, 'tree', inputs, EventSelections_total)#,accuracy=True)
#============================================================================== 
#    step = 0.001
#    decay_mode = 10.
#    mt_generated = np.arange(0.,200.,step)
#    dmode = np.add(np.zeros(len(mt_generated)),decay_mode).astype(np.float32).ravel()
#    GBC_input = np.column_stack((mt_generated,dmode))
#    TwoClassProba(f[0],GBC_input,"QCD_SS_generated_mt_DecMode_"+str(decay_mode)+"_stepsize_"+str(step))
#==============================================================================
