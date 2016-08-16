# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:33:45 2016

@author: janik
"""

from ROOT import TH1D, gPad
import numpy as np
import pickle
import os
import copy, time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
import ast 

from rootpy.plotting import Hist, Hist2D, Hist3D, Graph, Canvas, Legend, HistStack
from rootpy.io import root_open
from root_numpy import root2array, fill_graph, fill_hist

    
def simple_hist(array, name, data_weights = 1, Nxbins=100, X_axis_range=[0.,1.]) :
    if (type(data_weights) == int) :
        data_weights = np.ones(len(array))
    #h = Hist(30,40,200,title=name,markersize=0)
    h = TH1D("h1","simple_hist_"+name,Nxbins ,X_axis_range[0] ,X_axis_range[1])
    root_open("H1_"+name+".root", 'recreate')
    fill_hist(h,array,weights=data_weights)
    h.Write()
    #create Canvas and save the plot as png
    c = Canvas()
    h.Draw("HIST")
    c.SaveAs("H1_"+name+".png")


def txt_output(array, path, name) :
    text_file = open(path+"/"+name+".txt", "w")
    for i in array :
        text_file.write(str(i)+"\n")
    text_file.close()
    print name, ".txt was created."


def MultiClass_fit (filenames, treenames, inputsname, path, weight_name="weight",training=True,Cls_quality=False) :
        
    #load the file to obtain the proper scaling of QCD and Wjet
    sample_info = pickle.load(open('/afs/cern.ch/user/j/jsauvan/public/Htautau/mc_info_76.pck', 'rb'))
    
    #luminosity of the data set of QCD events      
    #! EVERYTHING IS RESCALED TO THE LUMINOSITY OF THE QCD DATA SAMPLE !
    luminosity_QCD_data = 2260.0    
    Rescale_Wjet = luminosity_QCD_data*sample_info['W']['XSec']/sample_info['W']['SumWeights']
    Rescale_Z    = luminosity_QCD_data*sample_info['ZJ']['XSec']/sample_info['ZJ']['SumWeights']
    Rescale_tt   = luminosity_QCD_data*sample_info['TT']['XSec']/sample_info['TT']['SumWeights']
        
    print "need to rescale Wjet events by ", Rescale_Wjet
    print "need to rescale Z events by ", Rescale_Z
    print "need to rescale TT events by ", Rescale_tt
    
           
    ninputs = len(inputsname)
    # merge inputsname with weight_name to branch_names
    branch_names = copy.copy(inputsname)
    branch_names.append(weight_name)
    
    
    # Reading inputs from ROOT tree     
    #need to apply all cuts at once, the data variables are of the matrix format 
    #with each row of the from (mt,decay channel,...other discrimination variables..., weight) 
#============================================================================== 
    #Wjet
    data1 = root2array(filenames[0], treename=treenames[0], branches=branch_names, selection=OppositeChargeEventCuts())    
    #QCD data (with subtracted events)
    # CPU time test
    t0_CPU = time.clock()
    data2 = SubEventsFromQCD(filename=filenames[1], treename=treenames[1], EventSelectionCuts=SameChargeEventCuts(), SubFileNames=QCD_dataSubrtaction()[0], SubTreeNames='tree', inputsname=inputsname, XSectionNames= QCD_dataSubrtaction()[1], path = path)
    t_CPU = time.clock() - t0_CPU
    txt_output(["The CPU time for obtaining the subtracted QCD event samples " + str(t_CPU)], path, "CPU_QCD_DATA_sub")   
   
    #selection cuts for fake and non-fake MC events
    fake    = ' && '.join((OppositeChargeEventCuts(),'l2_gen_match==6'))   
    nonFake = ' && '.join((OppositeChargeEventCuts(),'l2_gen_match!=6'))    
        
    #Z fake events 
    data3 = root2array(filenames[2], treename=treenames[2], branches=branch_names, selection=fake)
    #Z non fake events 
    data4 = root2array(filenames[2], treename=treenames[2], branches=branch_names, selection=nonFake)
    #tt fake events 
    data5 = root2array(filenames[3], treename=treenames[3], branches=branch_names, selection=fake)
    #tt non fake events 
    data6 = root2array(filenames[3], treename=treenames[3], branches=branch_names, selection=nonFake)   
#==============================================================================

        
    # change to proper array display, data 2 is already properly displayed
    data1 = data1.view((np.float64, len(data1.dtype.names)))  
    data3 = data3.view((np.float64, len(data3.dtype.names)))  
    data4 = data4.view((np.float64, len(data4.dtype.names)))  
    data5 = data5.view((np.float64, len(data5.dtype.names)))  
    data6 = data6.view((np.float64, len(data6.dtype.names)))  
    
    #extracting weights
    weight1 = data1[:, [ninputs]].astype(np.float32).ravel() 
    weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 
    weight3 = data3[:, [ninputs]].astype(np.float32).ravel() 
    weight4 = data4[:, [ninputs]].astype(np.float32).ravel() 
    weight5 = data5[:, [ninputs]].astype(np.float32).ravel() 
    weight6 = data6[:, [ninputs]].astype(np.float32).ravel() 


    #proper rescaling of the QCD and Wjet event weights      
    weight1 = np.multiply(weight1,Rescale_Wjet) #Wjet
    weight3 = np.multiply(weight3,Rescale_Z) #Z fake
    weight4 = np.multiply(weight4,Rescale_Z) #Z non fake
    weight5 = np.multiply(weight5,Rescale_tt) #tt fake
    weight6 = np.multiply(weight6,Rescale_tt) #tt non fake


    
    #the weight was extracted above data.._disc_var is a matrix with each 
    #row of the form (mt, decay channel,...other discriminating variales...)
    data1_disc_var = data1[:, range(ninputs)].astype(np.float32)
    data2_disc_var = data2[:, range(ninputs)].astype(np.float32)
    data3_disc_var = data3[:, range(ninputs)].astype(np.float32)
    data4_disc_var = data4[:, range(ninputs)].astype(np.float32)
    data5_disc_var = data5[:, range(ninputs)].astype(np.float32)
    data6_disc_var = data6[:, range(ninputs)].astype(np.float32)
    
    
    
    #define classes for training    
    class1 = np.zeros((data1_disc_var.shape[0],))         #Wjet
    class2 = np.zeros((data2_disc_var.shape[0],))+1.      #QCD
    class3 = np.zeros((data3_disc_var.shape[0],))+2.      #Zfake
    class4 = np.zeros((data4_disc_var.shape[0],))+3.      #Znonfake
    class5 = np.zeros((data5_disc_var.shape[0],))+4.      #TTfake
    class6 = np.zeros((data6_disc_var.shape[0],))+5.      #TTnonfake
    
    
       
    # Merging datasets
#==============================================================================     
    inputs = np.concatenate((data1_disc_var, data2_disc_var, data3_disc_var, data4_disc_var, data5_disc_var, data6_disc_var))
    classes = np.concatenate((class1, class2, class3, class4, class5, class6))
    weights = np.concatenate((weight1, weight2, weight3, weight4, weight5, weight6))
     
    #save the above variables in pickle files
    if (ninputs == 10) :
        pickle.dump( inputs, open( path+"/"+"inputs.pck", "wb" ) )
        pickle.dump( classes, open( path+"/"+"classes.pck", "wb" ) )
        pickle.dump( weights, open( path+"/"+"weights.pck", "wb" ) )
#==============================================================================
    
    
    
    # Split events in a training sample and a test sample (60% used for training), 
    #40% is used for testing evaluating the classifier
    # add the weights as well
    inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test = cross_validation.train_test_split(inputs, classes, weights, test_size=0.4, random_state=0)
    #save the splitting in train and test samples
#    pickle.dump( inputs_train, open( path+"/"+"inputs_train.pck", "wb" ) )
#    pickle.dump( inputs_test, open( path+"/"+"inputs_test.pck", "wb" ) )
#    pickle.dump( targets_train, open( path+"/"+"targets_train.pck", "wb" ) )
#    pickle.dump( targets_test, open( path+"/"+"targets_test.pck", "wb" ) )
#    pickle.dump( weights_train, open( path+"/"+"weights_train.pck", "wb" ) )
#    pickle.dump( weights_test, open( path+"/"+"weights_test.pck", "wb" ) )
        
    
        
    if (training == True) :
        # Fit and test classifier (BDT with gradient boosting)
        # Default training parameters are used
        clf = GradientBoostingClassifier()
        
        print "classifying started"
        # CPU time test
        t0_CPU = time.clock()
        clf.fit(inputs_train, targets_train, sample_weight=weights_train)
        t_CPU = time.clock() - t0_CPU
    
        txt_output(["The CPU time for 60 % training was " + str(t_CPU)], path, "CPU_Training")
        #save the classifier    
        pickle.dump( clf, open( path+"/"+"classifier.pck", "wb" ) )

    if (Cls_quality == True) :
        ClassifierQuality(path,inputs_test,targets_test, weights_test)

def ClassifierQuality (path, inp_test, tar_test, weig_test) :        

    classifier = pickle.load( open( path+"/"+"classifier.pck", "rb" ) )     
    inputs_test = inp_test
    targets_test = tar_test
    weights_test = weig_test

#    inputs_test = pickle.load( open( path+"/"+"inputs_test.pck", "rb" ) ) 
#    targets_test = pickle.load( open( path+"/"+"targets_test.pck", "rb" ) ) 
#    weights_test = pickle.load( open( path+"/"+"weights_test.pck", "rb" ) ) 
    
    
    accuracy = classifier.score(inputs_test, targets_test, sample_weight=weights_test)
    VariableImportance = classifier.feature_importances_
    
    targets_pred = classifier.predict(inputs_test)
    
    #pickle.dump( targets_pred, open( path+"/"+"targets_pred.pck", "wb" ) )

    ConfMatrix = confusion_matrix(targets_test,targets_pred,labels=[0,1,2,3,4,5])
    
    pickle.dump( ConfMatrix, open( path+"/"+"confusion_matrix.pck", "wb" ) )
    pickle.dump( accuracy, open( path+"/"+"cls_accuracy.pck", "wb" ) )
    pickle.dump( VariableImportance, open( path+"/"+"variable_importance.pck", "wb" ) )

    txt_output([accuracy,VariableImportance,ConfMatrix],path,"ClassifierQuality")
    WjetMissidentification(path,inputs_test, targets_test, targets_pred)

#def WjetMissidentification(inputs, trueClass, predClass) :
def WjetMissidentification(path, inputs, trueClass, predClass) :
        
#    inputs = pickle.load( open( path+"/"+"inputs_test.pck", "rb" ) ) 
#    trueClass = pickle.load( open( path+"/"+"targets_test.pck", "rb" ) ) 
#    predClass = pickle.load( open( path+"/"+"targets_pred.pck", "rb" ) ) 
    ninputs = len(inputs[0,:])    

    #loop over all the discriminating variables
    for variable_index in xrange(ninputs) :
        #extract input variable
        variable = inputs[:,variable_index]
        #put the variable together with the true and predicted class in order
        #to make filtering possible. The form is (variable, true class, pred class)
        data = np.vstack((variable,trueClass,predClass)).T
        #filter for Wjet missidentification x[2] == 0 corresponds to predicted Wjet 
        #x[1] != x[2] corresponds to the condition of missidentification    
        #--> the combination filters for mispredicted Wjet events
        variable_missID = np.array(filter(lambda x: (x[2] == 0 and x[2] != x[1]), data))[:,0]
        
        #gives num of bins with min and max in a 3-tuple
        a = VariableBinning(GetVariableName(variable_index))
        h = Hist(a[2],a[0],a[1])
        h.SetStats(0)
        c = Canvas()
        h.fill_array(variable_missID)
    
        h.GetXaxis().SetTitle(GetVariableName(variable_index))
        h.GetYaxis().SetTitle("number of events")
        h.SetTitle("W+jet missidentification in " + GetVariableName(variable_index))
        
        #color and style settings
        h.SetLineColor(0)
        h.SetMarkerStyle(21) #square
        h.SetMarkerColor(1) #black
        h.SetMarkerSize(0.9)
        h.Draw("P")
        c.SaveAs(path+"/"+"WjetMissID_"+GetVariableName(variable_index)+".png")
    
    return 0

def GetClass (index) :
    if (index == 1) :
        return 'Wjet'
    if (index == 2) :
        return 'QCD'
    if (index == 3) :
        return 'Zfake'
    if (index == 4) :
        return 'Znonfake'
    if (index == 5) :
        return 'TTfake'
    if (index == 6) :
        return 'TTnonfake'    


def Eventdist (path,variables) :
    inputs = pickle.load( open( path+"/"+"inputs.pck", "rb" ) ) 
    classes = pickle.load( open( path+"/"+"classes.pck", "rb" ) ) 
    weights = pickle.load( open( path+"/"+"weights.pck", "rb" ) ) 

#    class_list = [0,1,2,3,4,5]    
    variables_index = 0

    
    for variable_name in variables :
        data = np.transpose(np.vstack((inputs[:,variables_index], classes, weights)))
        variables_index = variables_index + 1
        
        
#...........................................................................        
        minimum = VariableBinning(variable_name)[0]
        maximum = VariableBinning(variable_name)[1]
        Nbins = VariableBinning(variable_name)[2]
    
        #create histograms        
        h0 = Hist(Nbins,minimum,maximum)
        h1 = Hist(Nbins,minimum,maximum)
        h2 = Hist(Nbins,minimum,maximum)
        h3 = Hist(Nbins,minimum,maximum)
        h4 = Hist(Nbins,minimum,maximum)    
        h5 = Hist(Nbins,minimum,maximum)
        
        #extract data and fill the histograms
        class_index = 0
        for h in [h0,h1,h2,h3,h4,h5] :
            var = np.array(filter(lambda x: x[1] == class_index, data))[:,0]
            weight = np.array(filter(lambda x: x[1] == class_index, data))[:,2]
            h.fill_array(var, weights=weight)
            
            #normalize all the integrals to 1
            scale = 1./h.Integral()
            #do not normalize
            #scale = 1.0            
            
            h.Scale(scale)
            h.SetStats(0)
            h.SetTitle(variable_name+" distribution")
            class_index = class_index + 1
        #color and marker settings
        h0.SetLineColor(0)
        h0.SetMarkerStyle(21) #square
        h0.SetMarkerColor(1) #black
        h0.SetMarkerSize(0.9)
        
        h1.SetLineColor(0)
        h1.SetMarkerStyle(22) #triangle up
        h1.SetMarkerColor(2) #red
        h1.SetMarkerSize(0.9)
        
        h2.SetLineColor(0)
        h2.SetMarkerStyle(23) #triangle down
        h2.SetMarkerColor(4) #blue
        h2.SetMarkerSize(0.9)
        
        h3.SetLineColor(0)
        h3.SetMarkerStyle(33) #raute
        h3.SetMarkerColor(6) #magenta
        h3.SetMarkerSize(1.0)
        
        h4.SetLineColor(0)
        h4.SetMarkerStyle(34) #cross
        h4.SetMarkerColor(7) #cyan
        h4.SetMarkerSize(0.9)
        
        h5.SetLineColor(0)
        h5.SetMarkerStyle(20) #circle
        h5.SetMarkerColor(9) #violett
        h5.SetMarkerSize(0.9)
        
        c = Canvas()
        
        stack = HistStack()
        stack.Add(h0)
        stack.Add(h1)
        stack.Add(h2)
        stack.Add(h3)
        stack.Add(h4)
        stack.Add(h5)
        
        stack.Draw("P nostack")
    
        stack.xaxis.SetTitle(variable_name)
        stack.yaxis.SetTitle("number of events (normalized)")
#        stack.yaxis.SetTitle("number of events")
        stack.SetTitle("Event distribution for " +variable_name)
    
        
        legend = Legend(6, rightmargin=0.1, leftmargin=0.45, margin=0.3)
        legend.AddEntry(h0, "Wjet", style='P')
        legend.AddEntry(h1, "QCD", style='P')
        legend.AddEntry(h2, "Zfake", style='P')
        legend.AddEntry(h3, "Znonfake", style='P')
        legend.AddEntry(h4, "TTfake", style='P')
        legend.AddEntry(h5, "TTnonfake", style='P')
        
        legend.Draw()
        c.SaveAs(path+"/"+"EventDist_"+variable_name+".png")    
#        c.SaveAs(path+"/"+"NotNormalizedEventDist_"+variable_name+".png")    
 

def Variable_Correlation (path) :
    inputs = pickle.load( open( path+"/"+"inputs.pck", "rb" ) ) 
    
#    for i in xrange(0,10,1) :
#        for j in xrange(i+1,10,1) :
    i = 0
    j = 8

    #extract the 2 variables to compare and put them in 2d array
    x = inputs[:,i]
    y = inputs[:,j]
    array = np.vstack((x,y)).T
    
    
    #gives num of bins with min and max in a 3-tuple
    a = VariableBinning(GetVariableName(i))
    b = VariableBinning(GetVariableName(j))
    #create 2d histogram with appropriate binning and ranges
    h2d = Hist2D(a[2],a[0],a[1],b[2],b[0],b[1])
    h2d.SetStats(0)

#    h2d.SetMarkerStyle(20)
#    h2d.SetMarkerSize(0.3)
    
    #axis labels and title
    h2d.GetXaxis().SetTitle(GetVariableName(i))
    h2d.GetYaxis().SetTitle(GetVariableName(j))
    h2d.SetTitle("Two variable correlation")
    
   
    c = Canvas()   
    h2d.fill_array(array)
    gPad.SetLogz();
    h2d.Draw("COLZ")
    c.SaveAs("variable_correlation/"+GetVariableName(i)+"_vs_"+GetVariableName(j)+"_Corr"+".png")
    

 
 
def GetVariableName (index) :
    if (index == 0) :
        return "mt"
    if (index == 1) :
        return "l2_decayMode"
    if (index == 2) :
        return "mvis"
    if (index == 3) :
        return "n_bjets"
    if (index == 4) :
        return "n_jets"
    if (index == 5) :
        return "l1_reliso05"
    if (index == 6) :
        return "delta_phi_l1_l2"
    if (index == 7) :
        return "delta_eta_l1_l2"
    if (index == 8) :
        return "delta_phi_l1_met"
    if (index == 9) :
        return "delta_phi_l2_met"
  
def VarImportance () :
    var1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    var2 = np.array([0.92896082, 0.07103918, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    var3 = np.array([0.47708751, 0.05206651, 0.47084598, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    var4 = np.array([0.36941843, 0.0364033, 0.45997332, 0.13420496, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    var5 = np.array([0.35006318, 0.04330156, 0.34119783, 0.05113996, 0.21429747, 0.0, 0.0, 0.0, 0.0, 0.0])
    var6 = np.array([0.27423583, 0.0391834, 0.18402275, 0.0618217, 0.38364066, 0.05709566, 0.0, 0.0, 0.0, 0.0])
    var7 = np.array([0.26920434, 0.04043931, 0.29218903, 0.03162261, 0.0825477, 0.05955056, 0.2244464, 0.0, 0.0, 0.0])
    var8 = np.array([0.1074763, 0.02234695, 0.02389117, 0.02825535, 0.03163124, 0.02164265, 0.0522094, 0.71254696, 0.0, 0.0])
    var9 = np.array([0.10552508, 0.02073505, 0.03322193, 0.0255536, 0.03110894, 0.01854066, 0.04647704, 0.71503332, 0.0038043, 0.0])
    var10 = np.array([0.10533593, 0.03331511, 0.03700573, 0.02290451, 0.02808899, 0.01885084, 0.05842051, 0.65205103, 0.01031483, 0.03371251])
    
    
    
    
    stack = HistStack()
    l = len(var1)
    c = Canvas()
    legend = Legend(10, rightmargin=0.02, leftmargin=0.55, margin=0.3, topmargin = 0.01,  textsize=0.03)
    for j in xrange(0,l,1) :
        h = Hist(29,0.75,15.25)
        counter = 0
        for variable in [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10] :
            h.SetBinContent((2*counter+1),variable[j])
            
            counter = counter + 1
        h.fillstyle = 'solid'
        h.SetFillColor(j+1)
        stack.Add(h)
        legend.AddEntry(h, GetVariableName(j), style='F')        


    stack.Draw("HIST")
    stack.xaxis.SetTitle("Number of variables")
    stack.yaxis.SetTitle("Importance")
    stack.SetTitle("Variables importance")
    legend.Draw()
    c.SaveAs("VariableImportance.png")     
        

def VariableBinning (var) :
    if (var == "mt") :
        minimum = 0.0
        maximum = 200.0
        Nbins = 100
        return minimum, maximum, Nbins
    if (var == "l2_decayMode") :
        minimum = -0.5
        maximum = 10.5
        Nbins = 11
        return minimum, maximum, Nbins    
    if (var == "mvis") :
        minimum = 0.0
        maximum = 200.0
        Nbins = 100
        return minimum, maximum, Nbins
    if (var == "n_bjets") :
        minimum = -0.5
        maximum = 5.5
        Nbins = 6
        return minimum, maximum, Nbins
    if (var == "n_jets") :
        minimum = -0.5
        maximum = 10.5
        Nbins = 11
        return minimum, maximum, Nbins
    if (var == "l1_reliso05") :
        minimum = 0.0
        maximum = 0.5
        Nbins = 20
        return minimum, maximum, Nbins
    if (var == "delta_phi_l1_l2") :
        minimum = -4.0
        maximum = 4.0
        Nbins = 100
        return minimum, maximum, Nbins
    if (var == "delta_eta_l1_l2") :
        minimum = 0.0
        maximum = 5.0
        Nbins = 100
        return minimum, maximum, Nbins
    if (var == "delta_phi_l1_met") :
        minimum = -3.5
        maximum = 4.5
        Nbins = 100
        return minimum, maximum, Nbins
    if (var == "delta_phi_l2_met") :
        minimum = -4.0
        maximum = 4.0
        Nbins = 100
        return minimum, maximum, Nbins
    else :
        return "no info"


def ConfMatrixAnalsis() : 
    InputVariables = ['mt','l2_decayMode','mvis','n_bjets','n_jets','l1_reliso05', 'delta_phi_l1_l2', 'delta_eta_l1_l2', 'delta_phi_l1_met', 'delta_phi_l2_met']
    var_names = []
    conf_matrices = []    
    
    for i in xrange(len(InputVariables)) :
        var_names.append(InputVariables[i])
        path = PathGenerator(var_names)
        conf_matrices.append(pickle.load( open( path+"/"+"confusion_matrix.pck", "rb" ) ) )
    
    l = len(conf_matrices[0][0,:]) 
 
    #loop over classes    
    for j in xrange(0,l,1) :
        
        stack = HistStack()
        c = Canvas()
        legend = Legend(6, rightmargin=0.02, leftmargin=0.55, margin=0.3, topmargin = 0.01,  textsize=0.03)
                
        #loop over the row elements in the confusion matrix
        for k in xrange(0,l,1) :
            h = Hist(29,0.75,15.25)
      
            counter = 0
            #loop over the number of inputvariables in matrix form
            for variable in conf_matrices :
                h.SetBinContent((2*counter+1),variable[j,k])   
                counter = counter + 1

            h.fillstyle = 'solid'
            h.SetFillColor(k+1)
            stack.Add(h)
            legend.AddEntry(h, GetClass(k+1), style='F')        


        stack.Draw("HIST")
        stack.xaxis.SetTitle("Number of variables")
        stack.yaxis.SetTitle("Predicted classes")
        stack.SetTitle("Predicted classes for true class "+GetClass(j+1))
        legend.Draw()
        c.SaveAs("ConfMatrixEvolution_"+GetClass(j+1)+".png")      
     
    
def ColorPlotConfMatrix() :

    InputVariables = ['mt','l2_decayMode','mvis','n_bjets','n_jets','l1_reliso05', 'delta_phi_l1_l2', 'delta_eta_l1_l2', 'delta_phi_l1_met', 'delta_phi_l2_met']
    var_names = []
    conf_matrices = []    
    
    #reading the confusion matrix
    for i in xrange(len(InputVariables)) :
        var_names.append(InputVariables[i])
        path = PathGenerator(var_names)
        conf_matrices.append(pickle.load( open( path+"/"+"confusion_matrix.pck", "rb" ) ) )

    
    
    variable_counter = 1
    
    for mat in conf_matrices :     
        #create 2d histogram
        h2d = Hist2D(6,0,6,6,0,6)
        h2d.SetStats(0)
        #get the dimensions of the confusion matrix
        row_num = h2d.GetNbinsX()
        column_num = h2d.GetNbinsX()
    
        #filling the histogram
        for i in xrange(row_num) :
            h2d.GetXaxis().SetBinLabel((i+1),GetClass(i+1))
            h2d.GetYaxis().SetBinLabel((i+1),GetClass(i+1))
            for j in xrange(column_num) :
                h2d.SetBinContent((j+1),(i+1),mat[i,j])
       
        
        #axis labels and title
        h2d.GetXaxis().SetTitle("predicted")
        h2d.GetYaxis().SetTitle("true")
        h2d.SetTitle("Confusion matrix for "+str(variable_counter)+ " input variable(s)" )
        
       
        c = Canvas()   
        gPad.SetLogz();
        h2d.Draw("COLZ")
        c.SaveAs("confusion_matrix/confusion_matrix_NoOfVariables_"+str(variable_counter)+".png")
        variable_counter = variable_counter + 1
        
 


    variable_counter = 1

    #relative fraction 
    for mat in conf_matrices :     
#    mat = conf_matrices[9]

        #create 2d histogram
        h2d = Hist2D(6,0,6,6,0,6)
        h2d.SetStats(0)
        #get the dimensions of the confusion matrix
        row_num = h2d.GetNbinsX()
        column_num = h2d.GetNbinsX()
    
        #filling the histogram
        for i in xrange(row_num) :
            h2d.GetXaxis().SetBinLabel((i+1),GetClass(i+1))
            h2d.GetYaxis().SetBinLabel((i+1),GetClass(i+1))
            sum_ConfMat = sum(mat[i,:])
            print sum_ConfMat
#            print "-----------------"
            for j in xrange(column_num) :              
                h2d.SetBinContent((j+1),(i+1),np.divide(mat[i,j],1.*sum_ConfMat))
#                print np.divide(mat[i,j],1.*sum_ConfMat)
        
        
        #axis labels and title
        h2d.GetXaxis().SetTitle("predicted")
        h2d.GetYaxis().SetTitle("true")
        h2d.SetTitle("Confusion matrix for "+str(variable_counter)+ " input variable(s)" )
        h2d.SetMinimum(1.e-2)
        h2d.SetMaximum(1.)
       
        c = Canvas()   
        gPad.SetLogz();
        h2d.Draw("COLZ")
        c.SaveAs("confusion_matrix/confusion_matrix_REL_NoOfVariables_"+str(variable_counter)+".png")
        variable_counter = variable_counter + 1
        
    
    
    
    

def SubEventsFromQCD (filename, treename, EventSelectionCuts, SubFileNames, SubTreeNames, inputsname, XSectionNames, path, weight_name="weight") :
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

    data = copy.copy(data1)
    
    #simple_hist(data1[:, 0],"Start", data_weights = data1[:,ninputs], Nxbins=200, X_axis_range=[0.,200.])
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
        #rescale the weights, the minus sign is for the subtraction
        data2[:, ninputs] = np.multiply(data2[:, ninputs],(-1)*RescaleFactor)  

        #subtract
        data = np.concatenate((data,data2))
        
#        #some plots
#        simple_hist(data2[:, 0], SubFileNames[j], data_weights = data2[:,ninputs], Nxbins=200, X_axis_range=[0.,200.])        
#        simple_hist(data[:, 0],"Subtraction"+str(j), data_weights = data[:,ninputs], Nxbins=200, X_axis_range=[0.,200.])


#    pickle.dump( data , open( path+"/"+"QCD_subtracted.pck", "wb" ) )
    return data



def OppositeChargeEventCuts () :
    OppositeCharge = ['l1_charge*l2_charge<0']
    LeptonVetos = ['veto_dilepton<0.5','veto_thirdlepton<0.5','veto_otherlepton<0.5']
    METDataRemoval = ['!(met_pt < 0.15 && met_phi > 0. && met_phi < 1.8)']
    MuCuts = ['l1_pt>19', 'l1_reliso05<0.1', 'l1_muonid_medium>0.5']
    HadTauCuts = ['l2_againstMuon3>1.5', 'l2_againstElectronMVA6>0.5', 'l2_decayModeFinding', 'l2_pt>20']    
    AntiIsoTau = ['l2_byIsolationMVArun2v1DBoldDMwLT<3.5'] 
    #merge all event cuts in one logic command of event selection cuts
    EventSelections = ' && '.join(np.concatenate((LeptonVetos,METDataRemoval,OppositeCharge,MuCuts,HadTauCuts,AntiIsoTau)))
    return EventSelections

def SameChargeEventCuts() :
    SameCharge = ['l1_charge*l2_charge<0']
    LeptonVetos = ['veto_dilepton<0.5','veto_thirdlepton<0.5','veto_otherlepton<0.5']
    METDataRemoval = ['!(met_pt < 0.15 && met_phi > 0. && met_phi < 1.8)']
    MuCuts = ['l1_pt>19', 'l1_reliso05<0.1', 'l1_muonid_medium>0.5']
    HadTauCuts = ['l2_againstMuon3>1.5', 'l2_againstElectronMVA6>0.5', 'l2_decayModeFinding', 'l2_pt>20']    
    AntiIsoTau = ['l2_byIsolationMVArun2v1DBoldDMwLT<3.5'] 
    EventSelections = ' && '.join(np.concatenate((LeptonVetos,METDataRemoval,SameCharge,MuCuts,HadTauCuts,AntiIsoTau)))
    return EventSelections
   
def QCD_dataSubrtaction() :
    QCDSubtrNames = ['WJetsToLNu_LO', 'DYJetsToLL_M50_LO_ext1', 'TT_pow_ext3', 'T_tWch', 'TBar_tWch', 'ZZTo4L', 'ZZTo2L2Q', 'WZTo2L2Q', 'WZTo1L3Nu', 'WZTo3LNu_amcatnlo', 'WZTo1L1Nu2Q', 'WWTo1L1Nu2Q', 'VVTo2L2Nu']
    XSecNames = ['W', 'ZJ', 'TT', 'T_tWch', 'TBar_tWch', 'ZZTo4L', 'ZZTo2L2Q', 'WZTo2L2Q', 'WZTo1L3Nu', 'WZTo3L', 'WZTo1L1Nu2Q', 'WWTo1L1Nu2Q', 'VVTo2L2Nu']
    return QCDSubtrNames, XSecNames   
  
 
def SixClassProba (classifier, inputs, name) :
    #get the class probabilities
    class_proba = classifier.predict_proba(inputs) 
    
    pickle.dump( class_proba, open( "class_proba.pck", "wb" ) )



def PathGenerator(input_variables) :
    path = "Var"+str(len(input_variables))
    for i in input_variables :
        path = path+"_"+i
    newpath = r'/afs/cern.ch/user/j/jandrejk/work/Project/examples/AdditionalVariableTesting/'+path 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath
  

