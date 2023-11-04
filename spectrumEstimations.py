import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale

class simulatedSpectrumEstimations:
    def __init__(self,materialMus, phantomLengths, initialWs,spectrum,ordering=0,theta=0,I_0_NormalizedImageData=None,transmissionMeasurement=None):
        
        self.materialMus = materialMus # numMaterials x numEnergyBins
        self.numMaterials,self.numEnergyBins = np.shape(self.materialMus)
        self.energySpectrum = energySpectrum
        self.Esteps, self.energySpectrum = spectrum
        self.numMeasurements = self.numMaterials*len(phantomLengths)
        self.phantomLengths = phantomLengths
        self.numLengths = len(self.phantomLengths)
        self.ordering = ordering

        assert(self.numEnergyBins == len(self.energySpectrum))

        self.A = np.zeros(self.numMeasurements,self.energySpectrum)
        self.Ws = initialWs
        self.summedOverMeasurements = np.transpose(sum(self.A,axis=0)) ## numEnergyBins x 1

        self.setup()

        if transmissionMeasurements is None and imageData is None:
            self.transmissionMeasurements = self.simulateTransmissionMeasurements()  ## numMeasurements x 1 
        else if imageData is not None:
            self.transmissionMeasurements = self.getTMeasurementsFromImageData(angle,I_0_NormalizedImageData) ## numMeasurements x 1 (correspond to one angle's projection, averaged over detectors)
        else:
            self.transmissionMeasurements = transmissionMeasurements; ## numMeasurements x 1
        




    
    '''------------------------------------------ Setup & Simulation Functions ------------------------------------------'''

    def setup(self):
        if self.ordering==0:
            for m in range(self.numMaterials):
                for n in range(self.numLengths):
                    self.A[m*self.numMaterials+n,:] =  exp(-self.materialMus[m]*self.phantomLengths[n])
        else: 
            for n in range(self.numLengths):
                for m in range(self.numMaterials):
                    self.A[n*self.numLengths+m,:] =  exp(-self.materialMus[m]*self.phantomLengths[n])


    def simulateTransmissionMeasurements(self):
        return np.multiply(self.A*self.energySpectrum, self.Esteps)

    def getTMeasurementsFromImageData(self,angle, imageData):
        ## imageData Shape = (M,N, numMeasurements)
        imageData = np.array(imageData)
        assert(imageData.ndim == 3 and np.shape(imageData)[2] == self.numMeasurements)

        transmissionMeasurements = np.zeros(self.numMeasurements,1)
        thetas =np.linspace(0., 180., max(imageData[:,:,0].shape))

        for i in range(self.numMeasurements):
            scaledImage = rescale(imageData[:,:,i], scale=0.4, mode='reflect', channel_axis=None)
            sinogram = radon(scaledImage[:,:,i], theta=thetas)
            transmissionMeasurements[i] =  np.mean(sinogram[:,np.where(thetas == angle)])
        return transmissionMeasurements


    '''------------------------------------------ Update Steps ------------------------------------------'''
    def getSpectrum(self,count=0):

        '''
        G_j = [w_j^[n]/sumOverMeasurements(Aij)] 
        H_i = sumOverEnergies(Aij*w_j^[n])
        R_j = sumOverMeasurements(Aij*tData_i/H_i)
        w_j^[n+1] =  G_j* R_j

         '''
        if count == self.numMeasurements:
            return self.Ws

        G = np.divide(self.Ws,self.summedOverMeasurements) ## numEnergyBins x1 
        H = self.A*self.Ws ## numMeasurements x 1
        R = np.transpose(np.sum([self.A[i,:]*self.transmissionMeasurements[i]/H[i] for i in range(self.numMeasurements)],axis=0)) ## numEnergyBins x 1
        self.Ws = np.multiply(G,R) 
        return self.getSpectrum(count+1)





        
            
            
            
        
        


    

            
            


        




    
        