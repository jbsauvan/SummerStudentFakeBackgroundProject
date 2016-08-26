# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:33:45 2016

@author: janik
"""

from ROOT import TH1D, gPad, gStyle, TLegend
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

    

def MultiClass_fit (filenames, treenames, inputsname, path, weight_name="weight",QCD_MC = False, training=True,Cls_quality=False) :
        
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
    if (QCD_MC == True ) : #this needs to be improved 
        file_qcd = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/QCD_Mu15/H2TauTauTreeProducerTauMu/tree.root'
        data2 = root2array(file_qcd, treename='tree', branches=branch_names, selection=OppositeChargeEventCuts()) 
        data2 = data2.view((np.float64, len(data2.dtype.names)))  
        #Rescale_QCD = luminosity_QCD_data*sample_info['QCD']['XSec']/sample_info['QCD']['SumWeights']
        
        
        print "QCD MC extracted"
    else :
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


#    print "sum of weights:"
#    print np.sum(weight1)
#    print np.sum(weight2)
#    print np.sum(weight3)
#    print np.sum(weight4)
#    print np.sum(weight5)
#    print np.sum(weight6)
    
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


    if (len(inputs[0,:]) == 2) :
        print "fill proba histograms"
        Fill_mt_decMode_ProbHistos(path,inputs,classes,weights)
     
    #save the above variables in pickle files
#    if (ninputs == 10) :
#        pickle.dump( inputs, open( path+"/"+"inputs.pck", "wb" ) )
#        pickle.dump( classes, open( path+"/"+"classes.pck", "wb" ) )
#        pickle.dump( weights, open( path+"/"+"weights.pck", "wb" ) )
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
        #generate the test samples using the QCD MC data
        #file to QCD MC
        file_qcd_MC = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/QCD_Mu15/H2TauTauTreeProducerTauMu/tree.root'
        
        data_QCD_MC = root2array(file_qcd_MC, treename='tree', branches=branch_names, selection=OppositeChargeEventCuts())   
        data_QCD_MC = data_QCD_MC.view((np.float64, len(data_QCD_MC.dtype.names))) 
        Rescale_QCD_MC = luminosity_QCD_data*sample_info['QCD']['XSec']/sample_info['QCD']['SumWeights']
        print "need to rescale QCD MC events by ", Rescale_QCD_MC    
        weight_QCD_MC = data_QCD_MC[:, [ninputs]].astype(np.float32).ravel() 
        weight_QCD_MC = np.multiply(weight_QCD_MC,Rescale_QCD_MC)
        data_QCD_MC_disc_var = data_QCD_MC[:, range(ninputs)].astype(np.float32)
              
        class_QCD_MC = np.zeros((data_QCD_MC_disc_var.shape[0],))+1.      #QCD
        
        inputs = np.concatenate((data1_disc_var, data_QCD_MC_disc_var, data3_disc_var, data4_disc_var, data5_disc_var, data6_disc_var))
        classes = np.concatenate((class1, class_QCD_MC, class3, class4, class5, class6))
        weights = np.concatenate((weight1, weight_QCD_MC, weight3, weight4, weight5, weight6))    
           
        inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test = cross_validation.train_test_split(inputs, classes, weights, test_size=0.4, random_state=0)   
        
        #ClassifierQuality(path,inputs_test,targets_test, weights_test)
        #Regularization_Plots(path,inputs_test,targets_test, weights_test)
        if (len(inputs_test[0,:]) == 10) :        
            ClassProba(path,inputs_test,targets_test)
        if (len(inputs_test[0,:]) == 2) :        
            ClassifierQuality(path,inputs_test,targets_test, weights_test)

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
    #WjetMissidentification(path,inputs_test, targets_test, targets_pred, weights_test)



def Fill_mt_decMode_ProbHistos (path, inp_var, classes, weight) :
    data = np.vstack((inp_var[:,0],inp_var[:,1],classes,weight)).T
        
    h_Wjet = Hist2D(25,0.,250.,2,0,12)
    h_QCD = Hist2D(25,0.,250.,2,0,12)
    h_Zf = Hist2D(25,0.,250.,2,0,12)
    h_Znf = Hist2D(25,0.,250.,2,0,12)
    h_TTf = Hist2D(25,0.,250.,2,0,12)
    h_TTnf = Hist2D(25,0.,250.,2,0,12)

    class_index = 0
    for h in [h_Wjet,h_QCD,h_Zf,h_Znf,h_TTf,h_TTnf] :
        h.SetStats(0) 
        h.GetYaxis().SetBinLabel(1,"1 track")
        h.GetYaxis().SetBinLabel(2,"3 tracks")
        h.GetXaxis().SetTitle("mt")

        #need the filter because only the events of a certain class
        #get filled in the histogram
        data_mt_dec = np.array(filter(lambda x: (x[2] == class_index), data))
        

        h.fill_array(data_mt_dec[:,[0,1]],weights=data_mt_dec[:,3])
        class_index += 1
    
    
    SumHist = h_Wjet.Clone()
    SumHist.Add(h_QCD,1.)
    SumHist.Add(h_Zf,1.)
    SumHist.Add(h_Znf,1.)
    SumHist.Add(h_TTf,1.)
    SumHist.Add(h_TTnf,1.)
    
    class_index = 0
    
    for h in [h_Wjet,h_QCD,h_Zf,h_Znf,h_TTf,h_TTnf] :
        
        root_open(path+"/"+"ProbasByRatio"+GetClass(class_index+1)+".root", 'recreate')

        hist = h.Clone()
        hist = hist / SumHist
        hist.Write()    
         
        c = Canvas()        
        gStyle.SetPaintTextFormat("4.2f");
        hist.SetMarkerSize(1.2);        
        hist.Draw("COLZ TEXT90")

        c.SaveAs(path+"/"+GetClass(class_index+1)+".png")
        
        class_index += 1
    
    h_Wjet.Divide(SumHist)
    h_QCD.Divide(SumHist)
    h_Zf.Divide(SumHist)
    h_Znf.Divide(SumHist)
    h_TTf.Divide(SumHist)
    h_TTnf.Divide(SumHist)
    histos = [h_Wjet, h_QCD, h_Zf, h_Znf, h_TTf, h_TTnf]
    
    Hist_comp_ratios(path,histos,inp_var,classes, weight)
    Hist_comp_ratios(path,histos,inp_var,classes, weight,Full_collision_data=True)

    return 0    

def Hist_comp_ratios (path, histos, inputs, classes, weights, Full_collision_data=False) :

    if (Full_collision_data == True) :
        classifier = pickle.load( open( path+"/"+"classifier.pck", "rb" ) ) 
        InputVariables = ['mt','l2_decayMode']
        ninputs = len(InputVariables)
        weight_name = "weight"
        branch_names = copy.copy(InputVariables)
        branch_names.append(weight_name)
        #full collision data OS region
        file_data = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/SingleMuon_Run2015D_16Dec/H2TauTauTreeProducerTauMu/tree.root'

        data_full = root2array(file_data, treename='tree', branches=branch_names, selection=OppositeChargeEventCuts())  
        # change to proper array display, data 2 is already properly displayed
        data_full = data_full.view((np.float64, len(data_full.dtype.names)))  
        #extracting weights
        weights = data_full[:, [ninputs]].astype(np.float32).ravel() 
        print "All weights 1? : ", weights.all()
        #the weight was extracted above data.._disc_var is a matrix with each 
        #row of the form (mt, decay channel,...other discriminating variales...)
        mt_dec = data_full[:, range(ninputs)].astype(np.float32)
        #define classes     
        classes = np.zeros((mt_dec.shape[0],))         #Wjet
 
        probas = classifier.predict_proba(mt_dec) 
    
    else :
        classifier = pickle.load( open( path+"/"+"classifier.pck", "rb" ) )  
        inputs_train, inputs_test, targets_train, targets_test, weights_train, weights_test = cross_validation.train_test_split(inputs, classes, weights, test_size=0.4, random_state=0)   
    
        probas = classifier.predict_proba(inputs_test) # pickle.load( open( "class_probaWholeSample.pck", "rb" ) ) 
        mt_dec = inputs_test
        classes = copy.copy(targets_test)
        weights = copy.copy(weights_test) 

    decayChannel = [0.0,10.0]
    background = ["Wjet", "QCD", "Zfake", "TTfake"] #equivalent to class
    
    for dec in decayChannel :
        for bkg in background :
    
            #put mt_dec weights and the RELEVANT class probability together
            class_num = (GetClassIndex(bkg)-1)
            all_data = np.transpose(np.vstack((mt_dec[:,0],mt_dec[:,1], classes, weights, probas[:,class_num])))
            
            
#check this !!! why selecting only for the true class???   
#selecting for specific class not necessairy, have also proba when
#other class is the true class.       
#==============================================================================
#             all_data = np.array(filter(lambda x: x[2] == class_num, all_data))
#==============================================================================
    
    
            #extract the relevant values for the specific decay channel
            Filter = np.array(filter(lambda x: x[1] == dec, all_data))
            MT = Filter[:,0]    
            Weight = Filter[:,3]
            Prob_bkg = Filter[:,4] 

           
    
            SumProbXweight = np.multiply(Prob_bkg,Weight)
        
            h1 = Hist(25,0.,250.)
            h1.fill_array(MT, weights=SumProbXweight)            
            
            h2 = Hist(25,0.,250.)
            h2.fill_array(MT, weights=Weight)            
            
            h3 = h1.Clone("h3")
            h3.Divide(h2)
            
            if (dec == 0.0) :
                Filter2 = np.array(filter(lambda x: x[1] == 1.0, all_data))
                MT2 = Filter2[:,0]    
                Weight2 = Filter2[:,3]
                Prob_bkg2 = Filter2[:,4]
                SumProbXweight2 = np.multiply(Prob_bkg2,Weight2)
               
                h12 = Hist(25,0.,250.)
                h12.fill_array(MT2, weights=SumProbXweight2) 
                h22 = Hist(25,0.,250.)
                h22.fill_array(MT2, weights=Weight2) 
                                
                h32 = h12.Clone("h32")
                h32.Divide(h22)
                
                h12.SetStats(0)
                h22.SetStats(0)
                h32.SetStats(0)
                h12.SetLineColor(2)
                h22.SetLineColor(2)
               
                h32.SetLineColor(0)
                h32.SetMarkerStyle(23)
                h32.SetMarkerColor(2)
                h32.SetMarkerSize(1.2)
                
                
            #create Canvas and save the plot as png
            c = Canvas()
            c.Divide(2,2)
            c.cd(1)
            h1.SetStats(0)
            s1 = HistStack()

            s1.Add(h1)        
            if (dec == 0.0) :
                s1.Add(h12)
            
                        
            s1.Draw("HIST nostack")
            s1.GetXaxis().SetTitle("mt")
            s1.GetYaxis().SetTitle("#sum_{i} prob_{i} #times weight_{i}")            
            s1.GetYaxis().SetTitleOffset(1.6)           

            legend = TLegend(0.6,0.7,0.9,0.9)
            if (dec == 0.0) :
                legend.AddEntry(h1, "no #pi^{0}", 'F')
                legend.AddEntry(h12, "with #pi^{0}", 'F') 
            else :
                legend.AddEntry(h1, "3 tracks", 'F')
            legend.Draw()
            
        
            c.cd(2)
            h2.SetStats(0)         
            s2 = HistStack()

            s2.Add(h2)
            if (dec == 0.0) :
                s2.Add(h22)

            s2.Draw("HIST nostack")
            s2.xaxis.SetTitle("mt")
            s2.yaxis.SetTitle("#sum_{i} weight_{i}")
            s2.GetYaxis().SetTitleOffset(1.6)            

            legend = TLegend(0.6,0.7,0.9,0.9)
            if (dec == 0.0) :
                legend.AddEntry(h1, "no #pi^{0}", 'F')
                legend.AddEntry(h12, "with #pi^{0}", 'F') 
            else :
                legend.AddEntry(h1, "3 tracks", 'F')                
            legend.Draw()


            c.cd(3)
                 
            f1 = root_open(GetClassProbaPath(bkg))
            #get the 2 histograms for the 2 decay channels: 1 track & 3 tracks
            H1 = f1.Get("h_w_2d")
            if (dec == 10.0) :
                #3 tracks
                h_data = Hist(list(H1.xedges()))
                h_data[:] = H1[:,2]
            else :
                #1 track
                h_data = Hist(list(H1.xedges()))
                h_data[:] = H1[:,1]
             
            h_data.GetXaxis().SetRangeUser(0.,250.)
            h_data.GetYaxis().SetRangeUser(0.,1.)
            h_data.fillstyle = '/'
            h_data.fillcolor = (255,255,0) #yellow
            h_data.SetStats(0)    
            h_data.Draw("HIST")
            h_data.GetXaxis().SetTitle("mt")
            h_data.GetYaxis().SetTitle(bkg + " probability")   
             
        #    h3.SetFillColor(4) #blue
        #    h3.SetFillStyle(3005)
            h3.SetLineColor(0)
            h3.SetMarkerStyle(21)
            h3.SetMarkerColor(4)
            h3.SetMarkerSize(1.2)
            h3.SetStats(0)    
            h3.SetTitle(bkg+str(dec))
            h3.Draw("HIST P SAME")
                 
                             
            if (dec == 0.0) :
                h32.Draw("HIST P SAME")
#            c.Update()    



            legend = TLegend(0.6,0.7,0.9,0.9)
            if (dec == 0.0) :
                legend.AddEntry(h3, "BDT, no #pi^{0}", 'P')
                legend.AddEntry(h32, "BDT, with #pi^{0}", 'P') 
                legend.AddEntry(h_data, "data", 'F')
            else :
                legend.AddEntry(h3, "BDT",'P')
                legend.AddEntry(h_data, "data",'F')
            legend.Draw()

            c.cd(4)

            if (dec == 10.0) :
                #3 tracks
                print "three tracks"
                h_data = Hist(25,0.,250.)
                for k in xrange(25) :
                    h_data.SetBinContent((k+1),histos[GetClassIndex(bkg)-1].GetBinContent((k+1),2))
            else :
                #1 track
                h_data = Hist(25,0.,250.)
                print "one track"
                for k in xrange(25) :
                    h_data.SetBinContent((k+1),histos[GetClassIndex(bkg)-1].GetBinContent((k+1),1))
             
            h_data.GetXaxis().SetRangeUser(0.,250.)
            h_data.GetYaxis().SetRangeUser(0.,1.)
            h_data.fillstyle = '/'
            h_data.fillcolor = (255,0,0) #red
            h_data.SetStats(0)    
            h_data.Draw("HIST")
            h_data.SetTitle(bkg+" background")
            h_data.GetXaxis().SetTitle("mt")
            h_data.GetYaxis().SetTitle(bkg + " probability")   
            
            h3.SetLineColor(0)
            h3.SetMarkerStyle(21)
            h3.SetMarkerColor(4)
            h3.SetMarkerSize(1.2)
            h3.SetStats(0)    
            h3.SetTitle(bkg+str(dec))
            h3.Draw("HIST P SAME")
                 
                             
            if (dec == 0.0) :
                h32.Draw("HIST P SAME")
            c.Update()    



            legend = TLegend(0.6,0.7,0.9,0.9)
            if (dec == 0.0) :
                legend.AddEntry(h3, "BDT, no #pi^{0}", 'P')
                legend.AddEntry(h32, "BDT, with #pi^{0}", 'P') 
                legend.AddEntry(h_data, "histogram", 'F')
            else :
                legend.AddEntry(h3, "BDT",'P')
                legend.AddEntry(h_data, "histogram",'F')
            legend.Draw()

            if (Full_collision_data == True) :
                c.SaveAs(path+"/"+bkg+str(dec)+"_RatioCompSamptot_FullDataSample.png")
            else :                           
                c.SaveAs(path+"/"+bkg+str(dec)+"_RatioCompSamptot.png")

    return 0

def WjetMissidentification(path, inputs, trueClass, predClass,weights) :
        
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
        data = np.vstack((variable,trueClass,predClass,weights)).T
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
        h.fill_array(variable_missID,weights=weights_missID)
        h_true.fill_array(variable_TrueWjet,weights=weights_TrueWjet)
            
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
#            scale = 1./h.Integral()
            #do not normalize
            scale = 1.0            
            
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
#        stack.yaxis.SetTitle("number of events (normalized)")
        stack.yaxis.SetTitle("number of events")
        stack.yaxis.SetTitleOffset(1.6)
        stack.SetTitle("Event distribution for " +variable_name)
    
        
        legend = Legend(6, rightmargin=0.1, leftmargin=0.45, margin=0.3)
        legend.AddEntry(h0, "Wjet", style='P')
        legend.AddEntry(h1, "QCD", style='P')
        legend.AddEntry(h2, "Zfake", style='P')
        legend.AddEntry(h3, "Znonfake", style='P')
        legend.AddEntry(h4, "TTfake", style='P')
        legend.AddEntry(h5, "TTnonfake", style='P')
        
        legend.Draw()
#        c.SaveAs(path+"/"+"EventDist_"+variable_name+".png")    
        c.SaveAs(path+"/"+"NotNormalizedEventDist_"+variable_name+".png")    
 

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

def GetClassIndex (name) :
    if (name == 'Wjet') :
        return 1
    if (name == 'QCD') :
        return 2
    if (name == 'Zfake') :
        return 3
    if (name == 'Znonfake') :
        return 4
    if (name == 'TTfake') :
        return 5
    if (name == 'TTnonfake') :
        return 6
 
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
  
def GetClassProbaPath (name) :
    if (name == 'Wjet') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_wjets.root"
    if (name == 'QCD') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_qcd.root"
    if (name == 'Zfake') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_dy.root"
    if (name == 'TTfake') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_tt.root"
 
        

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
        c.SaveAs("ConfMat_QCD_MC/ConfMatrixEvolution_"+GetClass(j+1)+".png")      
     
    
def ColorPlotConfMatrix() :

    InputVariables = ['mt','l2_decayMode','mvis','n_bjets','n_jets','l1_reliso05', 'delta_phi_l1_l2', 'delta_eta_l1_l2', 'delta_phi_l1_met', 'delta_phi_l2_met']
    var_names = []
    conf_matrices = []    
    
    #reading the confusion matrix
    for i in xrange(len(InputVariables)) :
        var_names.append(InputVariables[i])
        path = PathGenerator(var_names)
        conf_matrices.append(pickle.load( open( path+"/"+"confusion_matrix.pck", "rb" ) ) )

    
    
#    variable_counter = 1
#    
#    for mat in conf_matrices :     
#        #create 2d histogram
#        h2d = Hist2D(6,0,6,6,0,6)
#        h2d.SetStats(0)
#        #get the dimensions of the confusion matrix
#        row_num = h2d.GetNbinsX()
#        column_num = h2d.GetNbinsX()
#    
#        #filling the histogram
#        for i in xrange(row_num) :
#            h2d.GetXaxis().SetBinLabel((i+1),GetClass(i+1))
#            h2d.GetYaxis().SetBinLabel((i+1),GetClass(i+1))
#            for j in xrange(column_num) :
#                h2d.SetBinContent((j+1),(i+1),mat[i,j])
#       
#        
#        #axis labels and title
#        h2d.GetXaxis().SetTitle("predicted")
#        h2d.GetYaxis().SetTitle("true")
#        h2d.SetTitle("Confusion matrix for "+str(variable_counter)+ " input variable(s)" )
#        
#       
#        c = Canvas()   
#        gPad.SetLogz();
#        h2d.Draw("COLZ")
#        c.SaveAs("ConfMat_QCD_MC/confusion_matrix_NoOfVariables_"+str(variable_counter)+".png")
#        variable_counter = variable_counter + 1
        
 


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
            #print sum_ConfMat
#            print "-----------------"
            for j in xrange(column_num) :              
                matrix_value = np.divide(mat[i,j],1.*sum_ConfMat)
                                
                if (matrix_value > 0.01) :
                    
                    h2d.SetBinContent((j+1),(i+1),matrix_value)
                else :
                    h2d.SetBinContent((j+1),(i+1),0)
#                    h2d.SetCellContent((j+1),(i+1),1.)#matrix_value)
#                print np.divide(mat[i,j],1.*sum_ConfMat)
        
        gStyle.SetPaintTextFormat("4.2f");
        h2d.SetMarkerSize(1.8);        
        #axis labels and title
        h2d.GetXaxis().SetTitle("predicted")
        h2d.GetYaxis().SetTitle("true")
        h2d.SetTitle("Confusion matrix for "+str(variable_counter)+ " input variable(s)" )
        h2d.SetMinimum(1.e-2)
        h2d.SetMaximum(1.)
       
        c = Canvas()   
        gPad.SetLogz();
        h2d.Draw("TEXT0 COLZ")
        c.SaveAs("ConfMat_QCD_MC/confusion_matrix_REL_NoOfVariables_"+str(variable_counter)+".png")
        variable_counter = variable_counter + 1
        
    
    
def Gaussian_Deviance(test, pred, weights) :
    normalisation = np.sum(weights)
    weighted_missID_sum = 0.
    for i in xrange(len(test)) :
        if (test[i] != pred[i]) :
            weighted_missID_sum += weights[i]
    return weighted_missID_sum / normalisation


def Regularization_Plots(path, inputs_test, targets_test, weights_test) :
    num_estimators = 100
    
    cls_names = ['Lrate0.1']
    marker = ['o-']
    color = ['blue']
    labl = ['learning_rate: 0.1']
    index = 0
    
    X_test = inputs_test 
    y_test = targets_test   
      
    for cl in cls_names :
        clf = pickle.load( open( path+"/"+"classifier.pck", "rb" ) )

        #compute test set deviance
        test_deviance = np.zeros((num_estimators,), dtype=np.float64)
            
        #for i, y_pred in enumerate(clf.staged_decision_function(X_test)) :
        for i, y_pred in enumerate(clf.staged_predict(X_test)) :
            
            # clf.loss_ assumes that y_test[i] in {0, 1}        
            #generic loss function leading to weird behaviour            
            #test_deviance[i] = clf.loss_(y_test, y_pred,sample_weight=weights_test)
            test_deviance[i] = Gaussian_Deviance(y_test, y_pred,weights=weights_test)

        plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            marker[index], color=color[index], label=labl[index])
        
        index += 1
        
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Adopted Gaussian Deviance')
    #plt.xlim((0.,300.))
    #plt.ylim((0.4,1.))
    plt.savefig(path+"/"+"regul_6classes.png")
    plt.show()   
   
   
def ClassProba(path, inputs_test, targets_test) :
#    inputs_test = pickle.load( open( path+"/"+"inputs_test.pck", "rb" ) )
    clf = pickle.load( open( path+"/"+"classifier.pck", "rb" ) ) 
    class_probas = clf.predict_proba(inputs_test) #pickle.load( open( path+"/"+"class_proba.pck", "rb" ) )[:,0]


    mvis = inputs_test[:,2]
    y_test = targets_test
    y_pred = clf.predict(inputs_test)
    
    
    data = np.vstack((mvis,y_test,y_pred,class_probas[:,0],class_probas[:,1],class_probas[:,2],class_probas[:,3],class_probas[:,4],class_probas[:,5])).T
    
    #filter the array for true Znonfake : y_test == 3
    data_Znonfake_pred = np.array(filter(lambda x: x[1] == 3, data))


#    Znonfake_predata = np.array(filter(lambda x: x[2] == 3, data_Znonfake_pred))    
#    QCD_predata = np.array(filter(lambda x: x[2] == 1, data_Znonfake_pred))    
#    Wjet_predata = np.array(filter(lambda x: x[2] == 0, data_Znonfake_pred))    
        
    c = Canvas()
    print len(data_Znonfake_pred[:,0])
    graph_Z = Graph(len(data_Znonfake_pred[:,0]))
    graph_QCD = Graph(len(data_Znonfake_pred[:,0]))
    graph_Wjet = Graph(len(data_Znonfake_pred[:,0]))

    root_open("ProbasForZnonfakePred.root", 'recreate')

    fill_graph(graph_Z,np.column_stack((data_Znonfake_pred[:,0],data_Znonfake_pred[:,6])))    
    fill_graph(graph_QCD,np.column_stack((data_Znonfake_pred[:,0],data_Znonfake_pred[:,4])))    
    fill_graph(graph_Wjet,np.column_stack((data_Znonfake_pred[:,0],data_Znonfake_pred[:,3])))    
    
    
    #color and marker settings
    graph_Wjet.SetLineColor(0)
    graph_Wjet.SetMarkerStyle(1) #dot
    graph_Wjet.SetMarkerColor(1) #black
    graph_Wjet.SetMarkerSize(0.5)
    
    graph_QCD.SetLineColor(0)
    graph_QCD.SetMarkerStyle(1) #dot
    graph_QCD.SetMarkerColor(2) #red
    graph_QCD.SetMarkerSize(0.5)
        
    graph_Z.SetLineColor(0)
    graph_Z.SetMarkerStyle(1) #dot
    graph_Z.SetMarkerColor(6) #magenta
    graph_Z.SetMarkerSize(0.5)


    graph_Wjet.GetXaxis().SetRangeUser(0.,200.)
    graph_QCD.GetXaxis().SetRangeUser(0.,200.)
    graph_Z.GetXaxis().SetRangeUser(0.,200.)

    graph_Wjet.GetYaxis().SetRangeUser(0.,1.)
    graph_QCD.GetYaxis().SetRangeUser(0.,1.)
    graph_Z.GetYaxis().SetRangeUser(0.,1.)
   
    graph_Wjet.Draw("AP")
    graph_QCD.Draw("P SAME")
    graph_Z.Draw("P SAME")
    
    
    legend = TLegend(0.6,0.7,0.9,0.9)

    legend.AddEntry(graph_Wjet, "Wjet", 'P')
    legend.AddEntry(graph_QCD, "QCD", 'P') 
    legend.AddEntry(graph_Z, "Z nonfake", 'P') 

    legend.Draw()
  
    c.SaveAs(path+"/"+"ZnonfakePredProbas.png")   



   
#def ClassProbaOLD(path, inputs_test, targets_test) :
##    inputs_test = pickle.load( open( path+"/"+"inputs_test.pck", "rb" ) )
#    clf = pickle.load( open( path+"/"+"classifier.pck", "rb" ) ) 
#    class_probas = clf.predict_proba(inputs_test) #pickle.load( open( path+"/"+"class_proba.pck", "rb" ) )[:,0]
#
#
#    mvis = inputs_test[:,2]
#    y_test = targets_test
#    y_pred = clf.predict(inputs_test)
#    
#    
#    data = np.vstack((mvis,y_test,y_pred,class_probas[:,0],class_probas[:,1],class_probas[:,2],class_probas[:,3],class_probas[:,4],class_probas[:,5])).T
#    
#    #filter the array for true Znonfake : y_test == 3
#    data_Znonfake_pred = np.array(filter(lambda x: x[1] == 3, data))
#
#    Znonfake_predata = np.array(filter(lambda x: x[2] == 3, data_Znonfake_pred))    
#    QCD_predata = np.array(filter(lambda x: x[2] == 1, data_Znonfake_pred))    
#    Wjet_predata = np.array(filter(lambda x: x[2] == 0, data_Znonfake_pred))    
#        
#    c = Canvas()
#    graph_Z = Graph(len(Znonfake_predata[:,0]))
#    graph_QCD = Graph(len(QCD_predata[:,0]))
#    graph_Wjet = Graph(len(Wjet_predata[:,0]))
#
#    root_open("ProbasForZnonfakePred.root", 'recreate')
#
#    fill_graph(graph_Z,np.column_stack((Znonfake_predata[:,0],Znonfake_predata[:,6])))    
#    fill_graph(graph_QCD,np.column_stack((QCD_predata[:,0],QCD_predata[:,4])))    
#    fill_graph(graph_Wjet,np.column_stack((Wjet_predata[:,0],Wjet_predata[:,3])))    
#    
#    
#    #color and marker settings
#    graph_Wjet.SetLineColor(0)
#    graph_Wjet.SetMarkerStyle(21) #square
#    graph_Wjet.SetMarkerColor(1) #black
#    graph_Wjet.SetMarkerSize(0.9)
#    
#    graph_QCD.SetLineColor(0)
#    graph_QCD.SetMarkerStyle(22) #triangle up
#    graph_QCD.SetMarkerColor(2) #red
#    graph_QCD.SetMarkerSize(0.9)
#        
#    graph_Z.SetLineColor(0)
#    graph_Z.SetMarkerStyle(33) #raute
#    graph_Z.SetMarkerColor(6) #magenta
#    graph_Z.SetMarkerSize(1.0)
#
#
#    graph_Wjet.GetXaxis().SetRangeUser(0.,200.)
#    graph_QCD.GetXaxis().SetRangeUser(0.,200.)
#    graph_Z.GetXaxis().SetRangeUser(0.,200.)
#
#    graph_Wjet.GetYaxis().SetRangeUser(0.,1.)
#    graph_QCD.GetYaxis().SetRangeUser(0.,1.)
#    graph_Z.GetYaxis().SetRangeUser(0.,1.)
#   
#    graph_Wjet.Draw("AP")
#    graph_QCD.Draw("P SAME")
#    graph_Z.Draw("P SAME")
#    
#    
#    legend = TLegend(0.1,0.7,0.48,0.9)
#
#    legend.AddEntry(graph_Wjet, "Wjet", 'P')
#    legend.AddEntry(graph_QCD, "QCD", 'P') 
#    legend.AddEntry(graph_Z, "Z nonfake", 'P') 
#
##    legend = Legend(3, topmargin=0.5 ,rightmargin=0.03, leftmargin=0.45, margin=0.3)
##    legend.AddEntry(graph_Wjet, "Wjet", style='P')
##    legend.AddEntry(graph_QCD, "QCD", style='P') 
##    legend.AddEntry(graph_Z, "Z nonfake", style='P') 
#    legend.Draw()
#  
#    c.SaveAs(path+"/"+"ProbasForZnonfakePredW.png")


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
  

