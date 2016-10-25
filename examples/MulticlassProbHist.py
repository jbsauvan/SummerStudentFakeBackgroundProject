# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:59:39 2016

@author: janik
"""

import numpy as np
import pickle

import ROOT

from rootpy.plotting import Hist2D, Hist3D, Graph, Canvas, Legend, HistStack, Hist
from rootpy.io import root_open
from root_numpy import root2array, fill_graph



def DataClassHistos () :
    name1 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_wjets.root"
    name2 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_qcd.root"
    name3 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_dy.root"
    name4 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_tt.root"
    
    f1 = root_open(name1)
    f2 = root_open(name2)
    f3 = root_open(name3)
    f4 = root_open(name4)

    H1 = f1.Get("h_w_2d")
    H2 = f2.Get("h_w_2d")
    H3 = f3.Get("h_w_2d")
    H4 = f4.Get("h_w_2d")
    
    
    decChannelSlice = [1,2]

    for b in decChannelSlice :    
    
        h1 = Hist(list(H1.xedges()))
        h1[:] = H1[:,b]    
        h2 = Hist(list(H2.xedges()))
        h2[:] = H2[:,b]    
        h3 = Hist(list(H3.xedges()))
        h3[:] = H3[:,b]    
        h4 = Hist(list(H4.xedges()))
        h4[:] = H4[:,b]    
    

        h1.GetXaxis().SetRangeUser(0.,250.)
        h2.GetXaxis().SetRangeUser(0.,250.)
        h3.GetXaxis().SetRangeUser(0.,250.)
        h4.GetXaxis().SetRangeUser(0.,250.)
        h1.GetYaxis().SetRangeUser(0.,1.)
        h2.GetYaxis().SetRangeUser(0.,1.)
        h3.GetYaxis().SetRangeUser(0.,1.)
        h4.GetYaxis().SetRangeUser(0.,1.)
        
    
        h1.fillstyle = 'solid'
        h2.fillstyle = 'solid'
        h3.fillstyle = 'solid'
        h4.fillstyle = 'solid'
          
        h1.fillcolor = (255,0,0)
        h2.fillcolor = (0,255,255)
        h3.fillcolor = (127,0,255)
        h4.fillcolor = (0,204,0)
    
        
    
    
    
        c = Canvas()
        stack = HistStack()
        stack.Add(h1)
        stack.Add(h2)
        stack.Add(h3)
        stack.Add(h4)
        stack.Draw("HIST")
    
        stack.xaxis.SetTitle("m_{T} [GeV]")
        stack.yaxis.SetTitle("Class probability")
        stack.SetTitle("Class probabilities for different event samples")
    
    
        legend = Legend(4, leftmargin=0.45, margin=0.3)
        if (b==1) :           
             legend.SetHeader("1 track")
        else :
            legend.SetHeader("3 tracks")
        legend.AddEntry(h1, "Wjet", style='F')
        legend.AddEntry(h2, "QCD", style='F')
        legend.AddEntry(h3, "ZJ", style='F')
        legend.AddEntry(h4, "ttJ", style='F')
        
        legend.Draw()    
        
        if (b==1) :           
             c.SaveAs("plots/ProcH1_ClassProba_CMS_1track.png")
        else :
            c.SaveAs("plots/ProcH1_ClassProba_CMS_3tracks.png")
        
        


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
    


def multiclass_graph(mt_dec,probas,name,axis_range=[0.,250.]) :
    #create a graph and save it as root file
    
    dec_modes = [0.0,1.0,10.0]

    print np.shape(mt_dec)
    print np.shape(probas)    
    
    num_of_classes = len(probas[0])
        
    all_input = np.transpose(np.vstack((mt_dec[:,0],mt_dec[:,1], probas[:,0], probas[:,1], probas[:,2], probas[:,3], probas[:,4], probas[:,5])))
    
    
    for j in dec_modes :    
        x = np.array(filter(lambda x: x[1] == j, all_input))[:,0]
        
        for k in xrange(0,num_of_classes,1) :
            y = np.array(filter(lambda x: x[1] == j, all_input))[:,k+2]        
            graph = Graph(len(x),"g1")
            #create a root file and fill it with the graph P(W+jet) vs mt
            root_open("plots/ROOTfiles/G_"+name+GetClass(k+1)+'_dec_'+str(j)+".root", 'recreate')
            fill_graph(graph,np.column_stack((x,y)))
            graph.Write()
            #create Canvas and save the plot as png
            c = Canvas()
            graph.SetTitle(name+GetClass(k+1)+'_dec_'+str(j))
            graph.SetMarkerSize(0.3)
            graph.GetXaxis().SetRangeUser(axis_range[0],axis_range[1])
            graph.Draw("AP")
            c.SaveAs("plots/G_"+name+GetClass(k+1)+'_dec_'+str(j)+".png")


def training_vs_data (name1,name2, name3, name4, name) :
    f1 = root_open(name1)
    f2 = root_open(name2)
    f3 = root_open(name3)
    f4 = root_open(name4)

    #get the 2 histograms for the 2 decay channels: 1 track & 3 tracks
    H1 = f1.Get("h_w_2d")
    #1 track
    h1 = Hist(list(H1.xedges()))
    h1[:] = H1[:,1]
    #3 tracks
    h3 = Hist(list(H1.xedges()))
    h3[:] = H1[:,2]    

    #axis range of the histograms & color settings
    h1.GetXaxis().SetRangeUser(0.,250.)
    h3.GetXaxis().SetRangeUser(0.,250.)
    h1.GetYaxis().SetRangeUser(0.,1.)
    h3.GetYaxis().SetRangeUser(0.,1.)
    h1.fillstyle = 'solid'
    h3.fillstyle = 'solid'   
    h1.fillcolor = (255,255,0) #yellow
    h3.fillcolor = (255,255,0)
    
    #get the graphs
    g1 = f2.Get("g1")
    g2 = f3.Get("g1")
    g3 = f4.Get("g1")
    
    #axis range of the histograms & color + style settings
    g1.GetXaxis().SetRangeUser(0.,250.)
    g2.GetXaxis().SetRangeUser(0.,250.)
    g3.GetXaxis().SetRangeUser(0.,250.)
    g1.GetYaxis().SetRangeUser(0.,1.)
    g2.GetYaxis().SetRangeUser(0.,1.)
    g3.GetYaxis().SetRangeUser(0.,1.)
    g1.SetMarkerSize(0.7)
    g1.SetMarkerColor(1) #black
    g1.SetMarkerStyle(20) #full circle
    g2.SetMarkerSize(0.7)
    g2.SetMarkerColor(2) #red
    g2.SetMarkerStyle(2) #plus
    g3.SetMarkerSize(0.7)
    g3.SetMarkerColor(1) #black
    g3.SetMarkerStyle(20) #full circle


    #1 track comparison
    c = Canvas()
    h1.GetXaxis().SetTitle("m_{T} [GeV]")
    h1.GetYaxis().SetTitle("Class probability")
    h1.SetTitle("Class probabilities for the " + name + " event sample")
    h1.SetStats(0)
    h1.Draw("HIST")
    g1.Draw("P SAME")
    g2.Draw("P SAME")
    legend = Legend(3, leftmargin=0.45, margin=0.3)
    legend.AddEntry(h1, "1 track data", style='F')
    legend.AddEntry(g1, "training, with #pi^{0}", style='P')
    legend.AddEntry(g2, "training, no #pi^{0}", style='P')
    legend.Draw()
    c.SaveAs("plots/Proc_"+name+"_1track_DataTrainingComparison.png")
    c.Close()    
    
    #3 tracks comparison
    d = Canvas()
    h3.GetXaxis().SetTitle("m_{T} [GeV]")
    h3.GetYaxis().SetTitle("Class probability")
    h3.SetTitle("Class probabilities for the " + name + " event sample")
    h3.SetStats(0)
    h3.Draw("HIST")
    g3.Draw("P SAME")
    legend = Legend(2, leftmargin=0.45, margin=0.3)
    legend.AddEntry(h3, "3 tracks data", style='F')
    legend.AddEntry(g3, "training, 3 tracks", style='P')
    legend.Draw()    
    d.SaveAs("plots/Proc_"+name+"_3tracks_DataTrainingComparison.png")    
    
    
if __name__=='__main__':
    
    #DataClassHistos()
    

#    input_train = pickle.load( open( "inputs_train.pck", "rb" ) )    
#    class_probas = pickle.load( open( "class_proba.pck", "rb" ) )
#    
#    multiclass_graph(input_train,class_probas, "ClProbas")
    
    
    
#    name = "Wjet"
#    name1 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_wjets.root"
#    name2 = "plots/ROOTfiles/G_ClProbasWjet_dec_0.0.root"
#    name3 = "plots/ROOTfiles/G_ClProbasWjet_dec_1.0.root"
#    name4 = "plots/ROOTfiles/G_ClProbasWjet_dec_10.0.root"
#    training_vs_data(name1, name2, name3, name4, name)
#    
#    
    name = "QCD"
    name1 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_qcd.root"
    name2 = "plots/ROOTfiles/G_ClProbasQCD_dec_0.0.root"
    name3 = "plots/ROOTfiles/G_ClProbasQCD_dec_1.0.root"
    name4 = "plots/ROOTfiles/G_ClProbasQCD_dec_10.0.root"
    training_vs_data(name1, name2, name3, name4, name)
    
    name = "Z"
    name1 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_dy.root"
    name2 = "plots/ROOTfiles/G_ClProbasZfake_dec_0.0.root"
    name3 = "plots/ROOTfiles/G_ClProbasZfake_dec_1.0.root"
    name4 = "plots/ROOTfiles/G_ClProbasZfake_dec_10.0.root"
    training_vs_data(name1, name2, name3, name4, name)
        
    name = "TT"
    name1 = "/afs/cern.ch/user/m/mflechl/public/Htautau/FakeRate/20160511/pieces/frac_tt.root"
    name2 = "plots/ROOTfiles/G_ClProbasTTfake_dec_0.0.root"
    name3 = "plots/ROOTfiles/G_ClProbasTTfake_dec_1.0.root"
    name4 = "plots/ROOTfiles/G_ClProbasTTfake_dec_10.0.root"
    training_vs_data(name1, name2, name3, name4, name)


    