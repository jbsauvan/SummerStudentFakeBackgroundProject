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


def compare (name1, name2, j) :
    c = Canvas()
    QCD = ROOT.TFile.Open(name1+".root")
    Wjet = ROOT.TFile.Open(name2+".root")
    c.Divide(2,1)
    c.cd(1)
    QCD.Draw("AP")
    Wjet.Draw("AP")
    c.SaveAs("plots/WeightComparison/WjetProbaDecayChannel_"+str(j)+"_WeightComparison.png")    
    
    
    
if __name__=='__main__':
    name1 = "plots/ROOTfiles/G_Wjet_proba_vs_mt_DecMode_10.0.root"
    name2 = "plots/ROOTfiles/G_Wjet_proba_vs_mt_DecMode_1.0.root"
    name3 = "plots/ROOTfiles/G_Wjet_proba_vs_mt_DecMode_0.0.root"

    f1 = ROOT.TFile(name1)
    f2 = ROOT.TFile(name2)
    f3 = ROOT.TFile(name3)
    print "root files are read"
    g1 = f1.Get("Wjet_proba_vs_mt_DecMode_10.0")
    g2 = f2.Get("Wjet_proba_vs_mt_DecMode_1.0")
    g3 = f3.Get("Wjet_proba_vs_mt_DecMode_0.0")

    g1.GetXaxis().SetRangeUser(0.,200.)
    g2.GetXaxis().SetRangeUser(0.,200.)
    g3.GetXaxis().SetRangeUser(0.,200.)
    
    name="Wjet_proba_combined"
    c = Canvas()

    g1.SetMarkerSize(0.7)
    g1.SetMarkerColor(1) #black
    g1.SetMarkerStyle(20) #full circle

    g1.GetXaxis().SetTitle("mt [GeV]")
    g1.GetYaxis().SetTitle("W+jet probability")
    g1.SetTitle("W+jet probability for 3 decay modes")    
    
    g1.Draw("SAME AP")

    g2.SetMarkerSize(0.7)
    g2.SetMarkerColor(2) #red
    g2.SetMarkerStyle(2) #plus
    g2.Draw("SAME P")
    
    g3.SetMarkerSize(0.7)
    g3.SetMarkerColor(4) #blue 
    g3.SetMarkerStyle(5) #cross         

        
    g3.Draw("SAME P")

    leg = Legend(3,header="The decay modes",leftmargin=0.4, topmargin=0.1, rightmargin=0.03)
    #p = Pad(0.,0.5,1.,1.)
#    leg.SetHeader("The decay modes")
    leg.AddEntry("g1","3 tracks","P")    
    leg.AddEntry("g2","1 track without #pi^{0}","P")    
    leg.AddEntry("g3","1 track with #pi^{0}","P")    
    
    leg.Draw()
    c.SaveAs("plots/ProcG_"+name+".png")

"""    
    root2 = root_open(name2,'recreate')
    root3 = root_open(name3,'recreate')
    root1.Write()
    root2.Write()
    root3.Write()

    c = Canvas()
    root1.Draw("AP same")
    root2.Draw("AP same")
    root3.Draw("AP same")
    c.SaveAs("plots/ProcG_"+name+".png")
"""