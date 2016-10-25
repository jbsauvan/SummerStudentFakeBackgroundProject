# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:59:39 2016

@author: janik
"""

import numpy as np

import ROOT

from rootpy.plotting import Hist2D, Hist3D, Graph, Canvas, Legend
from rootpy.io import root_open
from root_numpy import root2array, fill_graph



if __name__=='__main__':
    
    decaymode = ['0.0','1.0','10.0']
    for j in decaymode :
        nameData = "plots/ROOTfiles/G_Wjet_proba_vs_mt_DecMode_"+j+".root"
        nameQCD = "plots/ROOTfiles/G_Wjet_proba_vs_mt_DecMode_"+j+"_QCD_data_SUBTRACTED.root"
        
        f1 = ROOT.TFile(nameData)
        f2 = ROOT.TFile(nameQCD)
        print "root files are read"
        g1 = f1.Get("Wjet_proba_vs_mt_DecMode_"+j)
        g2 = f2.Get("Wjet_proba_vs_mt_DecMode_"+j+"_QCD_data_SUBTRACTED")
        
        print type(g1)
        print type(g2)
        
        g1.GetXaxis().SetRangeUser(0.,200.)
        g2.GetXaxis().SetRangeUser(0.,200.)
        
        c = Canvas()
    
        g1.SetMarkerSize(0.7)
        g1.SetMarkerColor(1) #black
        g1.SetMarkerStyle(20) #full circle
    
        g1.GetXaxis().SetTitle("#m_{T} [GeV]")
        g1.GetYaxis().SetTitle("W+jet probability")

        if (j == "0.0") :
            g1.SetTitle("W+jet probability for the decay mode with track without #pi^{0} (QCD from data)")    
            name="Wjet_proba_1track_QCD_data_subtracted"
        if (j == "1.0") :
            g1.SetTitle("W+jet probability for the decay mode with track with #pi^{0} (QCD from data)")    
            name="Wjet_proba_1trackPi0_QCD_data_subtracted"
        if (j == "10.0") :
            g1.SetTitle("W+jet probability for the decay mode with 3 tracks (QCD from data)")    
            name="Wjet_proba_3tracks_QCD_data_subtracted"
                
        g1.Draw("SAME AP")
    
        g2.SetMarkerSize(0.7)
        g2.SetMarkerColor(2) #red
        g2.SetMarkerStyle(2) #plus
        g2.Draw("SAME P")
        
    
        leg = Legend(3,header="The samples",leftmargin=0.5, topmargin=0.3, rightmargin=0.03)
        #p = Pad(0.,0.5,1.,1.)
    #    leg.SetHeader("The decay modes")
        leg.AddEntry(g1,"QCD MC","P")    
        leg.AddEntry(g2,"QCD data","P")    
        
        leg.Draw()
        c.SaveAs("plots/ProcG_"+name+".png")
        