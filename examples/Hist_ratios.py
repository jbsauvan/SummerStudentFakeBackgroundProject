# -*- coding: utf-8 -*-
"""
Created on Tue Aug 09 11:57:55 2016

@author: janik
"""

import numpy as np
import pickle

import ROOT
from ROOT import TH1D

from rootpy.plotting import Hist2D, Hist3D, Graph, Canvas, Legend, HistStack, Hist
from rootpy.io import root_open
from root_numpy import root2array, fill_graph, fill_hist


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
    

def GetClassProbaPath (name) :
    if (name == 'Wjet') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_wjets.root"
    if (name == 'QCD') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_qcd.root"
    if (name == 'Zfake') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_dy.root"
    if (name == 'TTfake') :
        return "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_tt.root"
    

def Hist_comp_ratios (dec, bkg) :
#    weights = pickle.load( open( "weights_train.pck", "rb" ) ) 
#    probas = pickle.load( open( "class_proba.pck", "rb" ) ) 
#    mt_dec = pickle.load( open( "inputs_train.pck", "rb" ) )    
#    classes = pickle.load( open( "targets_train.pck", "rb" ) )

    weights = pickle.load( open( "weights.pck", "rb" ) ) 
    probas = pickle.load( open( "class_probaWholeSample.pck", "rb" ) ) 
    mt_dec = pickle.load( open( "inputs.pck", "rb" ) )    
    classes = pickle.load( open( "classes.pck", "rb" ) )
 

   
    
    #put mt_dec weights and the RELEVANT class probability together
    class_num = (GetClassIndex(bkg)-1)
    all_data = np.transpose(np.vstack((mt_dec[:,0],mt_dec[:,1], classes, weights, probas[:,class_num])))
    all_data = np.array(filter(lambda x: x[2] == class_num, all_data))
    
    
    #extract the relevant values for the specific decay channel
    Filter = np.array(filter(lambda x: x[1] == dec, all_data))
    MT = Filter[:,0]    
    Weight = Filter[:,3]
    Prob_bkg = Filter[:,4] 

           
    
    SumProbXweight = np.multiply(Prob_bkg,Weight)
    
   
    h1 = TH1D("h1","SumProbXweight_"+bkg+str(dec), 25 , 0.0 ,250.)
    root_open("plots/ROOTfiles/H1_SumProbXweight_"+bkg+str(dec)+"Samptot.root", 'recreate')
    fill_hist(h1,MT, weights=SumProbXweight)
    h1.Write()
    h2 = TH1D("h2","SumWeight_"+bkg+str(dec), 25 , 0.0 ,250.)
    root_open("plots/ROOTfiles/H1_SumWeight_"+bkg+str(dec)+"Samptot.root", 'recreate')
    fill_hist(h2,MT,weights=Weight)
    h2.Write()
    
    h3 = h1.Clone("h3")
    h3.Divide(h2)
    
    if (dec == 0.0) :
        Filter2 = np.array(filter(lambda x: x[1] == 1.0, all_data))
        MT2 = Filter2[:,0]    
        Weight2 = Filter2[:,3]
        Prob_bkg2 = Filter2[:,4]
        SumProbXweight2 = np.multiply(Prob_bkg2,Weight2)
        h12 = TH1D("h12","SumProbXweight_"+bkg+str(1.0), 25 , 0.0 ,250.)
        root_open("plots/ROOTfiles/H1_SumProbXweight_"+bkg+str(1.0)+"Samptot.root", 'recreate')
        fill_hist(h12,MT2, weights=SumProbXweight2)
        h12.Write()
        h22 = TH1D("h22","SumWeight_"+bkg+str(1.0), 25 , 0.0 ,250.)
        root_open("plots/ROOTfiles/H1_SumWeight_"+bkg+str(1.0)+"Samptot.root", 'recreate')
        fill_hist(h22,MT2,weights=Weight2)
        h22.Write()
        
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
    if (dec == 0.0) :
        if (h1.GetMaximum() > h12.GetMaximum()) :
            print "h1"
            h12.GetYaxis().SetRangeUser(0.,h1.GetMaximum())
        else :
            print "h12"
            h12.GetYaxis().SetRangeUser(0.,h12.GetMaximum())
        h12.Draw("HIST")
    h1.Draw("HIST SAME")

    c.cd(2)
    h2.SetStats(0)         
    if (dec == 0.0) :
        if (h2.GetMaximum() > h22.GetMaximum()) :
            h22.GetYaxis().SetRangeUser(0.,h2.GetMaximum())
        else :
            h22.GetYaxis().SetRangeUser(0.,h22.GetMaximum())
        h22.Draw("HIST")
    h2.Draw("HIST SAME")
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
     
#    h3.SetFillColor(4) #blue
#    h3.SetFillStyle(3005)
    h3.SetLineColor(0)
    h3.SetMarkerStyle(21)
    h3.SetMarkerColor(4)
    h3.SetMarkerSize(1.2)
    h3.SetStats(0)    
    h3.SetTitle(bkg+str(dec))
    h3.GetXaxis().SetTitle("m_{T}")
    h3.GetYaxis().SetTitle("Class probability")        
         
         
    h3.Draw("HIST P SAME")
    if (dec == 0.0) :
        h32.Draw("HIST P SAME")
    c.Update()    
    
    
    if (dec == 0.0) :        
        legend = Legend(3, leftmargin=0.45, margin=0.3)
        legend.AddEntry(h3, "training, no #pi^{0}", style='P')
        legend.AddEntry(h32, "training, with #pi^{0}", style='P')
        legend.AddEntry(h_data, "data", style='F')
        legend.Draw()
    else :    
        legend = Legend(2, leftmargin=0.45, margin=0.3)
        legend.AddEntry(h3, "training", style='P')
        legend.AddEntry(h_data, "data", style='F')
        legend.Draw()
    
    c.SaveAs("plots/H1_"+bkg+str(dec)+"_RatioCompSamptot.png")


if __name__=='__main__':
    print "Hello world"
    
    decayChannel = [0.0,10.0]
    background = ["Wjet", "QCD", "Zfake", "TTfake"] #equivalent to class
    
    for dec in decayChannel :
        for bkg in background :
            Hist_comp_ratios(dec,bkg)    
        
#    Hist_comp_ratios(0.0,"Wjet")
    
    
        