import numpy as np
import matplotlib.pyplot as plt
class simulatedSpectrumEstimations:
    def __init__(self, energySpectrum,materialMus, phantomLengths, initialWs, ordering=0,transmissionMeasurement=None):
        
        self.materialMus = materialMus # numMaterials x numEnergyBins
        self.numMaterials,self.numEnergyBins = np.shape(self.materialMus)
        self.energySpectrum = energySpectrum
        self.numMeasurements = self.numMaterials*len(phantomLengths)
        self.phantomLengths = phantomLengths
        self.numLengths = len(self.phantomLengths)
        self.ordering = ordering

        if transmissionMeasurements is None:
            transmissionMeasurements = self.simulateTransmissionMeasurements()



        self.transmissionMeasurements = transmissionMeasurements; ## numMeasurements x 1 (correspond to one angle's projection, averaged over detectors)

        assert(self.numMeasurements==len(self.transmissionMeasurements))
        assert(np.shape(self.materialMus)[1] == len(energySpectrum))
        assert(np.shape(self.materialMus)[1] == len(energySpectrum))
        
        self.A = np.zeros(self.numMeasurements,self.energySpectrum)
        self.Ws = initialWs

        self.summedOverMeasurements = np.transpose(sum(self.A,axis=0)) ## numEnergyBins x 1
        self.summedOverEnergies = np.transpose(sum(self.A,axis=1)) ## numMeasurements x 1




    
    '''------------------------------------------ Setup & Simulation Functions ------------------------------------------'''

    def setup(self):
        if self.ordering==0:
            for m in range(self.numMaterials):
                for n in range(self.numLengths):
                    self.A[m*self.numMaterials+n,:] =  exp(self.materialMus[m]*self.numLengths[n])
        else: 
            for n in range(self.numLengths):
                for m in range(self.numMaterials):
                    self.A[m*self.numMaterials+n,:] =  exp(self.materialMus[m]*self.numLengths[n])


    def simulateTransmissionMeasurements(self):
        ## TODO
        return [] 


    '''------------------------------------------ Update Steps ------------------------------------------'''
    def getSpectrum(self,count=0):
        if self.numMeasurements == self.numMeasurements:
            return self.Ws
        ## TODO 
        return self.getSpectrum(count+1)





        
            
            
            
        
        


    

            
            


        




    
        