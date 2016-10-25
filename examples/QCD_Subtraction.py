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



    name1 = "plots/ROOTfiles/H1_QCD_weighted_EventDistribution_all_channels.root"
    name2 = "plots/ROOTfiles/H1_WJetsToLNu_LO_weighted_EventDistribution_all_channels.root"
    name3 = "plots/ROOTfiles/H1_DYJetsToLL_M50_LO_ext1_weighted_EventDistribution_all_channels.root"
    name4 = "plots/ROOTfiles/H1_TT_pow_ext3_weighted_EventDistribution_all_channels.root"
   
   
   
  
   
    f1 = TFile(name1)
    f2 = TFile(name2)
    f3 = TFile(name3)
    f4 = TFile(name4)
   
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
    h3 = f3.Get("h1")
    h4 = f4.Get("h1")
    
    h1.GetXaxis().SetRangeUser(0.,200.)
    h2.GetXaxis().SetRangeUser(0.,200.)
    h3.GetXaxis().SetRangeUser(0.,200.)
    h4.GetXaxis().SetRangeUser(0.,200.)
   
    name="QCD_SUB"
    c = Canvas()
    
    h1.SetStats(0)
    h2.SetStats(0)
    h3.SetStats(0)
    h4.SetStats(0)

    h1.SetFillColor(2)
    h1.SetFillStyle(3001)

    h1.GetXaxis().SetTitle("mt [GeV]")
    h1.GetYaxis().SetTitle("Number of events")
    h1.GetYaxis().SetTitleOffset(1.5)
    
    h1.SetTitle("Event distribution for W+jet MC and QCD data")    
    
    
    h2.SetFillColor(4)
    h2.SetFillStyle(3001)
    h3.SetFillColor(3)
    h3.SetFillStyle(3001)
    h4.SetFillColor(5)
    h4.SetFillStyle(3001)
    

    h1.Draw("HIST")
    h2.Draw("HIST same")
    h3.Draw("HIST same")
    h4.Draw("HIST same")
#==============================================================================
#    
#    stack = THStack()
#    stack.Add(h1)
#    stack.Add(h2)
#     
#    stack.Draw("HIST")
#==============================================================================
   
   
    leg = Legend(4,header="Sample Events",leftmargin=0.38, topmargin=0.05, rightmargin=0.01)
    #p = Pad(0.,0.5,1.,1.)
#    leg.SetHeader("The decay modes")
    leg.AddEntry(h1,"QCD data","F")    
    leg.AddEntry(h2,"WJetsToLNu_LO sub.","F")    
    leg.AddEntry(h3,"DYJetsToLL_M50_LO_ext1 sub.","F")    
    leg.AddEntry(h4,"TT_pow_ext3 sub.","F")    
    #leg.SetTextFont(2)
    leg.SetTextSize(0.03)
    leg.Draw()
    c.SaveAs("plots/ProcH1_"+name+".png")

