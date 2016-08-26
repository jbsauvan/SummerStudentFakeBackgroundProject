# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:59:39 2016

@author: janik
"""

import numpy as np


from ROOT import TH1F, TFile, TCanvas, THStack

from rootpy.plotting import Hist2D, Hist3D, Graph, Canvas, Legend, HistStack, Hist
from rootpy.io import root_open
from root_numpy import root2array, fill_graph


 
if __name__=='__main__':



    name1 = "plots/ROOTfiles/H1_Wjet_all_channels.root"
    name2 = "plots/ROOTfiles/H1_QCD_data_all_channels.root"
   
    f1 = TFile(name1)
    f2 = TFile(name2)
   
#    h1 = Hist(30, 40, 200, title='Background', markersize=0)
#    # fill the histograms with our distributions
#    h1.FillRandom('landau', 1000)   
#    h2 = Hist(30, 40, 200, title='Signal', markersize=0)
#    # fill the histograms with our distributions
#    h2.FillRandom('landau', 1500)   
#    print type(h1)

 
    print "root files are read"
    #check this entry maybe type not recognized properly
    h1 = f1.Get("h1")
    h2 = f2.Get("h1")
    print type(h1)
    print type(h2)
    
    h1.GetXaxis().SetRangeUser(0.,200.)
    h2.GetXaxis().SetRangeUser(0.,200.)
   
    name="Wjet_proba_combined"
    c = Canvas()
    
    h1.SetStats(0)
    h2.SetStats(0)

    h1.SetFillColor(4)
    #h1.SetFillStyle(3001)

    h1.GetXaxis().SetTitle("mt [GeV]")
    h1.GetYaxis().SetTitle("Number of events")
    h1.GetYaxis().SetTitleOffset(1.5)
    
    h1.SetTitle("Event distribution for W+jet MC and QCD data")    
    
    
    h2.SetFillColor(2)
    #h2.SetFillStyle(3001)
    

#    h1.Draw("HIST")
#    h2.Draw("HIST same")
#==============================================================================
#    
    stack = THStack()
    stack.Add(h1)
    stack.Add(h2)
     
    stack.Draw("HIST")
#==============================================================================
   
   
    leg = Legend(2,header="Sample Events",leftmargin=0.4, topmargin=0.05, rightmargin=0.03)
    #p = Pad(0.,0.5,1.,1.)
#    leg.SetHeader("The decay modes")
    leg.AddEntry(h1,"W+jet MC","F")    
    leg.AddEntry(h2,"QCD data subtracted","F")    
    #leg.SetTextFont(2)
    leg.SetTextSize(0.04)
    leg.Draw()
    c.SaveAs("plots/ProcH1_"+name+".png")

