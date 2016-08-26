# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:33:45 2016

@author: janik
"""

import mt_Test as mt
       

#use 2 different classes, the W+jet class and the QCD class (add probably more later)
file_w =   '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/WJetsToLNu_LO/H2TauTauTreeProducerTauMu/tree.root'
          # /afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/*            /H2TauTauTreeProducerTauMu/tree.root
#QCD data
file_qcd_data = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/SingleMuon_Run2015D_16Dec/H2TauTauTreeProducerTauMu/tree.root'

#QCD MC
file_qcd = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/QCD_Mu15/H2TauTauTreeProducerTauMu/tree.root'

#2 branches, 1) the transverse mass mt and 2) the decay mode with 2 leptons in
#the final state l2_decayMode.   


InputVariables = ['mt']
path = mt.PathGenerator(InputVariables+["_regularisation_"])

#mt.Regularization_Plots(path)
#mt.Classifier_Regularization(filenames=[file_w,file_qcd_data],treenames=['tree','tree'],inputsname=InputVariables, path = path, QCDfromPickle=True)    
mt.Regularization_Plots(path)


#path = mt.PathGenerator(InputVariables)

#QCD data
#mt.MultiClass_fit(filenames=[file_w,file_qcd_data],treenames=['tree','tree'],inputsname=InputVariables, path = path, QCDfromPickle=False, training = False, Cls_quality=False)    
#mt.ClassProba(path)
 

#QCD MC
#mt.MultiClass_fit(filenames=[file_w,file_qcd],treenames=['tree','tree'],inputsname=InputVariables, path = path, QCDfromPickle=False, training = False, Cls_quality=False)    
#mt.ClassProba(path)