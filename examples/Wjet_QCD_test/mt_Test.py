# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 12:07:13 2016

@author: janik
"""



from ROOT import TH1D, gPad
import numpy as np
import matplotlib.pyplot as plt
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
#from sklearn.tree import DecisionTreeClassifier





def Classifier_Regularization (filenames, treenames, inputsname, path, weight_name="weight",QCDfromPickle = False) :
        
    #load the file to obtain the proper scaling of QCD and Wjet
    sample_info = pickle.load(open('/afs/cern.ch/user/j/jsauvan/public/Htautau/mc_info_76.pck', 'rb'))    
    #luminosity of the data set of QCD events      
    #! EVERYTHING IS RESCALED TO THE LUMINOSITY OF THE QCD DATA SAMPLE !
    luminosity_QCD_data = 2260.0    
    Rescale_Wjet = luminosity_QCD_data*sample_info['W']['XSec']/sample_info['W']['SumWeights']        
    print "need to rescale Wjet events by ", Rescale_Wjet 
               
    ninputs = len(inputsname)
    # merge inputsname with weight_name to branch_names
    branch_names = copy.copy(inputsname)
    branch_names.append(weight_name)    
    # Reading inputs from ROOT tree     
    #need to apply all cuts at once, the data variables are of the matrix format 
    #with each row of the from (mt,decay channel,...other discrimination variables..., weight) 
#============================================================================== 
    print "test 1"
    #Wjet
    data1 = root2array(filenames[0], treename=treenames[0], branches=branch_names, selection=OppositeChargeEventCuts())       
    #QCD data (with subtracted events)
    print "test 2"    
    if (QCDfromPickle == True) :
        data2 = pickle.load( open( path+"/"+"QCD_subtracted.pck", "rb" ) ) 
        data2 = pickle.load( open( path+"/"+"QCD_MC.pck", "rb" ) ) 
        
    else :
        # CPU time test
        t0_CPU = time.clock()
        print "test 3"
        data2 = SubEventsFromQCD(filename=filenames[1], treename=treenames[1], EventSelectionCuts=SameChargeEventCuts(), SubFileNames=QCD_dataSubrtaction()[0], SubTreeNames='tree', inputsname=inputsname, XSectionNames= QCD_dataSubrtaction()[1], path = path)
        t_CPU = time.clock() - t0_CPU
        txt_output(["The CPU time for obtaining the subtracted QCD event samples " + str(t_CPU)], path, "CPU_QCD_DATA_sub")   

    print "data is read out"
    # change to proper array display, data 2 is already properly displayed
    data1 = data1.view((np.float64, len(data1.dtype.names)))  
    #extracting weights
    weight1 = data1[:, [ninputs]].astype(np.float32).ravel() 
    weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 
    #proper rescaling of the QCD and Wjet event weights      
    weight1 = np.multiply(weight1,Rescale_Wjet) #Wjet    
    #the weight was extracted above data.._disc_var is a matrix with each 
    #row of the form (mt, decay channel,...other discriminating variales...)
    data1_disc_var = data1[:, range(ninputs)].astype(np.float32)
    data2_disc_var = data2[:, range(ninputs)].astype(np.float32)    
    #define classes for training    
    class1 = np.zeros((data1_disc_var.shape[0],))         #Wjet
    class2 = np.zeros((data2_disc_var.shape[0],))+1.      #QCD    
       
    # Merging datasets
#==============================================================================     
    inputs = np.concatenate((data1_disc_var, data2_disc_var))
    classes = np.concatenate((class1, class2))
    weights = np.concatenate((weight1, weight2))     
    
    print np.shape(inputs)
    print np.shape(classes)
    print np.shape(weights)
#==============================================================================


    print "data sets devided and merged for classification"      
    # Split events in a training sample and a test sample (60% used for training), 
    #40% is used for testing evaluating the classifier
    # add the weights as well
    inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test = cross_validation.train_test_split(inputs, classes, weights, test_size=0.4, random_state=0)
    print "split in train and test samples successful"
   
#==============================================================================
#     pickle.dump( inputs_train, open( path+"/"+"inputs_train.pck", "wb" ) )
#     pickle.dump( targets_train, open( path+"/"+"targets_train.pck", "wb" ) )
#     pickle.dump( weights_train, open( path+"/"+"weights_train.pck", "wb" ) )
         
     
#    pickle.dump( inputs_test, open( path+"/"+"inputs_test_MC.pck", "wb" ) )
#    pickle.dump( targets_test, open( path+"/"+"targets_test_MC.pck", "wb" ) )
#    pickle.dump( weights_test, open( path+"/"+"weights_test_MC.pck", "wb" ) )
#==============================================================================
#    
#    num_estimators = 1000
#    learn_rate = [0.1,0.05,0.01]
#    
#    for learn in learn_rate :
#   
#        # Fit and test classifier (BDT with gradient boosting)
#        clf = GradientBoostingClassifier(n_estimators=num_estimators,learning_rate=learn,subsample=0.5)
#        
#        print "classifying started"
#        # CPU time test
#        t0_CPU = time.clock()
#        clf.fit(inputs_train, targets_train, sample_weight=weights_train)
#        t_CPU = time.clock() - t0_CPU
#    
#        txt_output(["The CPU time for 60 % training was " + str(t_CPU)], path, "CPU_Training_Subsample0.5_Lrate"+str(learn))
#        #save the classifier    
#        pickle.dump( clf, open( path+"/"+"classifier_Subsample0.5_Lrate"+str(learn)+".pck", "wb" ) )
    

def Gaussian_Deviance(test, pred, weights) :
#    l = len(test)
#    sum_deviance = 0.
#    for i in xrange(l) :
#        sum_deviance += weights[i]*np.abs(test[i]-pred[i])
#    
#    return sum_deviance
    return np.sum(np.multiply(weights,(test - pred)**2)) / np.sum(weights)


def Regularization_Plots(path) :
    num_estimators = 1000
    
    cls_names = ['Lrate0.01', 'Lrate0.05', 'Lrate0.1']#, 'Subsample0.5_Lrate0.01', 'Subsample0.5_Lrate0.05', 'Subsample0.5_Lrate0.1']
    marker = ['o-','v-','^-']#,'<-','>-','*-']
    color = ['blue','red','green']#,'black','magenta','cyan']
    labl = ['learning_rate: 0.01','learning_rate: 0.05','learning_rate: 0.1']#,'learning_rate: 0.01, subsampling : 0.5', 'learning_rate: 0.05, subsampling : 0.5', 'learning_rate: 0.1, subsampling : 0.5']
    index = 0
    
    X_test = pickle.load( open( path+"/"+"inputs_test.pck", "rb" ) )
    y_test = pickle.load( open( path+"/"+"targets_test.pck", "rb" ) )  
    weights_test = pickle.load( open( path+"/"+"weights_test.pck", "rb" ) )  
    
    
    for cl in cls_names :
        clf = pickle.load( open( path+"/"+"classifier_"+cl+".pck", "rb" ) )
#        targets_pred = clf.predict(X_test)    
#        ConfMatrix = confusion_matrix(y_test,targets_pred,labels=[0,1])

        print "done loading"
        #compute test set deviance
        test_deviance = np.zeros((num_estimators,), dtype=np.float64)

#        w1 = weights_test[:40000]
#        w2 = weights_test[-31429:]
#
#        s1 = np.dot(np.multiply(weights_test,y_test),y_test)
#
#        normal = np.sum(weights_test)
            
        #for i, y_pred in enumerate(clf.staged_decision_function(X_test)) :
        for i, y_pred in enumerate(clf.staged_predict(X_test)) :
            
            # clf.loss_ assumes that y_test[i] in {0, 1}        
            #generic loss function leading to weird behaviour            
            #test_deviance[i] = clf.loss_(y_test, y_pred,sample_weight=weights_test)
            test_deviance[i] = Gaussian_Deviance(y_test, y_pred,weights=weights_test)

#            
#            ones = 0
#            zeros = 0
#              
#            if (i%100 == 0) :                
#                print y_pred
#                if (y_pred[i] == 1) :
#                    ones += 1
#                else :                  
#                    if (y_pred[i] == 0) :
#                        zeros += 1
#                    else :
#                        print "weird class label", y_pred[i]
#            print "number of ones QCD", ones
#            print "number of zeros Wjet", zeros
#                neg_numbers = 0
#                for k in y_pred :
#                    if (k < -100.) :
#                        neg_numbers += 1.
#                print "Number of neg. numbers below -100:", neg_numbers
#            print max(y_pred)
#            print min(y_pred)
        
        plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            marker[index], color=color[index], label=labl[index])
        
        index += 1
        
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Gaussian Deviance')
    plt.xlim((0.,300.))
    #plt.ylim((0.4,1.))
    plt.savefig(path+"/GaussianDeviance/"+"regul_class_label_LrateOnly_300.png")
    plt.show()



def MultiClass_fit (filenames, treenames, inputsname, path, weight_name="weight",QCDfromPickle = False,training=True,Cls_quality=False) :
        
    #load the file to obtain the proper scaling of QCD and Wjet
    sample_info = pickle.load(open('/afs/cern.ch/user/j/jsauvan/public/Htautau/mc_info_76.pck', 'rb'))
    
    #luminosity of the data set of QCD events      
    #! EVERYTHING IS RESCALED TO THE LUMINOSITY OF THE QCD DATA SAMPLE !
    luminosity_QCD_data = 2260.0    
    Rescale_Wjet = luminosity_QCD_data*sample_info['W']['XSec']/sample_info['W']['SumWeights']
        
    print "need to rescale Wjet events by ", Rescale_Wjet 
    
           
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

    print "okay"

#    #little operation with QCD MC    
#    data2 = root2array(filenames[1], treename=treenames[1], branches=branch_names, selection=OppositeChargeEventCuts())   
#    data2 = data2.view((np.float64, len(data2.dtype.names))) 
#    Rescale_QCD_MC = luminosity_QCD_data*sample_info['QCD']['XSec']/sample_info['QCD']['SumWeights']
#    print "need to rescale QCD MC events by ", Rescale_QCD_MC    
#    weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 
#    weight2 = np.multiply(weight2,Rescale_QCD_MC)
#    data2_disc_var = data2[:, range(ninputs)].astype(np.float32)
#    
#    print np.shape(data2)
#    print np.shape(data2_disc_var)  
#    print np.shape(weight2)    
#    
#    data = np.vstack((data2_disc_var[:,0],weight2)).T
#    print np.shape(data)    
#    pickle.dump( data, open( path+"/"+"QCD_MC.pck", "wb" ) )

    
    #QCD data (with subtracted events)
    if (QCDfromPickle == True) :
        data2 = pickle.load( open( path+"/"+"QCD_subtracted.pck", "rb" ) ) 
    else :
        # CPU time test
        t0_CPU = time.clock()
        data2 = SubEventsFromQCD(filename=filenames[1], treename=treenames[1], EventSelectionCuts=SameChargeEventCuts(), SubFileNames=QCD_dataSubrtaction()[0], SubTreeNames='tree', inputsname=inputsname, XSectionNames= QCD_dataSubrtaction()[1], path = path)
        t_CPU = time.clock() - t0_CPU
        txt_output(["The CPU time for obtaining the subtracted QCD event samples " + str(t_CPU)], path, "CPU_QCD_DATA_sub")   
   

        
    # change to proper array display, data 2 is already properly displayed
    data1 = data1.view((np.float64, len(data1.dtype.names)))  

    #extracting weights
    weight1 = data1[:, [ninputs]].astype(np.float32).ravel() 
    weight2 = data2[:, [ninputs]].astype(np.float32).ravel() 

    #proper rescaling of the QCD and Wjet event weights      
    weight1 = np.multiply(weight1,Rescale_Wjet) #Wjet

    
    #the weight was extracted above data.._disc_var is a matrix with each 
    #row of the form (mt, decay channel,...other discriminating variales...)
    data1_disc_var = data1[:, range(ninputs)].astype(np.float32)
    data2_disc_var = data2[:, range(ninputs)].astype(np.float32)
    
    
    #define classes for training    
    class1 = np.zeros((data1_disc_var.shape[0],))         #Wjet
    class2 = np.zeros((data2_disc_var.shape[0],))+1.      #QCD
    
    
       
    # Merging datasets
#==============================================================================     
    inputs = np.concatenate((data1_disc_var, data2_disc_var))
    classes = np.concatenate((class1, class2))
    weights = np.concatenate((weight1, weight2))
     
#==============================================================================
    
    
    
    # Split events in a training sample and a test sample (60% used for training), 
    #40% is used for testing evaluating the classifier
    # add the weights as well
    inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test = cross_validation.train_test_split(inputs, classes, weights, test_size=0.4, random_state=0)

#    pickle.dump( inputs_train, open( path+"/"+"inputs_train_MC.pck", "wb" ) )
#    pickle.dump( inputs_test, open( path+"/"+"inputs_test_MC.pck", "wb" ) )
#    pickle.dump( targets_train, open( path+"/"+"targets_train_MC.pck", "wb" ) )
#    pickle.dump( targets_test, open( path+"/"+"targets_test_MC.pck", "wb" ) )
#    pickle.dump( weights_train, open( path+"/"+"weights_train_MC.pck", "wb" ) )
#    pickle.dump( weights_test, open( path+"/"+"weights_test_MC.pck", "wb" ) )
   
   
        
    if (training == True) :
        # Fit and test classifier (BDT with gradient boosting)
        # Default training parameters are used
 
       #clf = DecisionTreeClassifier()       
       #clf = GradientBoostingClassifier(learning_rate=0.0001)
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


def SubEventsFromQCD (filename, treename, EventSelectionCuts, SubFileNames, SubTreeNames, inputsname, XSectionNames, path, weight_name="weight", differencePlot=True) :
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

    
    if (differencePlot == True) :    
        mt_original = data1[:,0]
        weight_original = data1[:,1]
        
        mt_subtracted = data[:,0]
        weight_subtracted = data[:,1]
        
        h_orig = Hist(200,0.,200.)
        h_sub = Hist(200,0.,200.)
        
        h_orig.fill_array(mt_original, weights=weight_original)
        h_sub.fill_array(mt_subtracted, weights=weight_subtracted)

        hist = h_orig.Clone()
        hist.Add(h_sub,-1.)
        
        c = Canvas()
        
        
        hist.SetLineColor(0)
        hist.SetMarkerStyle(21) #square
        hist.SetMarkerColor(1) #black
        hist.SetMarkerSize(0.9)
        
        hist.Draw("P")
        
        c.SaveAs(path+"/"+"QCD_sub_difference.png")
    
    pickle.dump( data , open( path+"/"+"QCD_subtracted.pck", "wb" ) )
    return data

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

    targets_proba = classifier.predict_proba(inputs_test)    
    
    #pickle.dump( targets_pred, open( path+"/"+"targets_pred.pck", "wb" ) )

    ConfMatrix = confusion_matrix(targets_test,targets_pred,labels=[0,1])
    
    pickle.dump( ConfMatrix, open( path+"/"+"confusion_matrix.pck", "wb" ) )
    pickle.dump( accuracy, open( path+"/"+"cls_accuracy.pck", "wb" ) )
    pickle.dump( VariableImportance, open( path+"/"+"variable_importance.pck", "wb" ) )
    pickle.dump( targets_proba, open( path+"/"+"class_proba.pck", "wb" ) )

    txt_output([accuracy,VariableImportance,ConfMatrix, targets_proba],path,"ClassifierQuality")
    
    WjetMissidentification(path,inputs_test, targets_test, targets_pred, weights_test)


def ClassProba(path) :
    inputs_test = pickle.load( open( path+"/"+"inputs_test.pck", "rb" ) )
    Wjet_proba = pickle.load( open( path+"/"+"class_proba.pck", "rb" ) )[:,0]

    c = Canvas()
    graph = Graph(len(inputs_test),"WjetProba")
    print len(inputs_test)
    print len(Wjet_proba)


    root_open("WjetProba.root", 'recreate')

    fill_graph(graph,np.column_stack((inputs_test,Wjet_proba)))    
    
    graph.SetMarkerSize(0.3)
    graph.GetXaxis().SetRangeUser(0.,250.)
    graph.Draw("AP")
  
    c.SaveAs(path+"/"+"WjetProba_mt.png")

    
    
def WjetMissidentification(path, inputs, trueClass, predClass, weights) :
        
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
        data = np.transpose(np.vstack((variable,trueClass,predClass,weights)))
        #filter for Wjet missidentification x[2] == 0 corresponds to predicted Wjet 
        #x[1] != x[2] corresponds to the condition of missidentification    
        #--> the combination filters for mispredicted Wjet events
        variable_missID = np.array(filter(lambda x: (x[2] == 0 and x[2] != x[1]), data))[:,0]
        weights_missID = np.array(filter(lambda x: (x[2] == 0 and x[2] != x[1]), data))[:,3]
        #filter the true Wjet distribution
        variable_TrueWjet = np.array(filter(lambda x: (x[1] == 0), data))[:,0]        
        weights_TrueWjet = np.array(filter(lambda x: (x[1] == 0), data))[:,3]        
        
        #gives num of bins with min and max in a 3-tuple
        a = VariableBinning(GetVariableName(variable_index))
        h = Hist(a[2],a[0],a[1])
        h_true = Hist(a[2],a[0],a[1])
        h.SetStats(0)
        h_true.SetStats(0)
        c = Canvas()
        h.fill_array(variable_missID, weights=weights_missID)
        h_true.fill_array(variable_TrueWjet, weights= weights_TrueWjet)
            
        #color and style settings
        h.SetLineColor(0)
        h.SetMarkerStyle(21) #square
        h.SetMarkerColor(1) #black
        h.SetMarkerSize(0.9)
        h_true.SetLineColor(0)
        h_true.SetMarkerStyle(22) #triangle up
        h_true.SetMarkerColor(2) #red
        h_true.SetMarkerSize(0.9)        
         


        stack = HistStack()
        stack.Add(h)
        stack.Add(h_true)
        
        stack.Draw("P nostack")
    
        stack.xaxis.SetTitle(GetVariableName(variable_index))
        stack.yaxis.SetTitle("number of events")
        stack.SetTitle("W+jet missidentification in " + GetVariableName(variable_index))
    
        
        legend = Legend(2, rightmargin=0.1, leftmargin=0.45, margin=0.3)
        legend.AddEntry(h, "Wjet missID", style='P')
        legend.AddEntry(h_true, "Wjet true", style='P') 
        legend.Draw()

        c.SaveAs(path+"/"+"WjetMissID_"+GetVariableName(variable_index)+".png")
    
    return 0
 

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
   
def txt_output(array, path, name) :
    text_file = open(path+"/"+name+".txt", "w")
    for i in array :
        text_file.write(str(i)+"\n")
    text_file.close()
    print name, ".txt was created."

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
    newpath = r'/afs/cern.ch/user/j/jandrejk/work/Project/examples/Wjet_QCD_test/'+path 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath