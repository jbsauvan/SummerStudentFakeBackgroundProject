# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 14:33:45 2016

@author: janik
"""

import AdditionalVariableClassification as AVC
       

#use 2 different classes, the W+jet class and the QCD class (add probably more later)
file_w =   '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/WJetsToLNu_LO/H2TauTauTreeProducerTauMu/tree.root'
          # /afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/*            /H2TauTauTreeProducerTauMu/tree.root
#QCD data
file_qcd_data = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/SingleMuon_Run2015D_16Dec/H2TauTauTreeProducerTauMu/tree.root'
file_Z_data   = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/DYJetsToLL_M50_LO_ext1/H2TauTauTreeProducerTauMu/tree.root'
file_tt_data  = '/afs/cern.ch/work/s/steggema/public/mt/070416/TauMuSVFitMC/TT_pow_ext3/H2TauTauTreeProducerTauMu/tree.root'
#2 branches, 1) the transverse mass mt and 2) the decay mode with 2 leptons in
#the final state l2_decayMode.   


InputVariables = ['mt','l2_decayMode']#,'mvis','n_bjets','n_jets','l1_reliso05', 'delta_phi_l1_l2', 'delta_eta_l1_l2', 'delta_phi_l1_met', 'delta_phi_l2_met']

path = AVC.PathGenerator(InputVariables)
#AVC.Eventdist(path, InputVariables)
AVC.MultiClass_fit(filenames=[file_w,file_qcd_data,file_Z_data,file_tt_data],treenames=['tree','tree','tree','tree'],inputsname=InputVariables, path = path, QCD_MC=False, training =False, Cls_quality=False)    
