import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale

class SpekEstimations:
    def __init__(self,materialMus, phantomLengths, initialWs,spectrum,theta=0,I_0_NormalizedImageData=None,transmissionMeasurements=None):
        
        self.materialMus = np.array(materialMus) # numMaterials x numEnergyBins
        if self.materialMus.ndim == 1:
            self.materialMus = np.reshape(self.materialMus,(1, len(self.materialMus)))
        self.numMaterials,self.numEnergyBins = np.shape(self.materialMus)
        self.Esteps, self.energySpectrum = spectrum
        self.numMeasurements = self.numMaterials*len(phantomLengths)
        self.phantomLengths = phantomLengths
        self.numLengths = len(self.phantomLengths)


        assert(self.numEnergyBins == len(self.energySpectrum))


        self.A = np.zeros((self.numMeasurements,self.numEnergyBins))
        self.Ws = initialWs
        self.initialWs = initialWs
        

        self.setup()



        if transmissionMeasurements is None and I_0_NormalizedImageData is None:
            self.transmissionMeasurements = self.simulateTransmissionMeasurements()  ## numMeasurements x 1 
        elif imageData is not None:
            self.transmissionMeasurements = self.getTMeasurementsFromImageData(angle,I_0_NormalizedImageData) ## numMeasurements x 1 (correspond to one angle's projection, averaged over detectors)
        else:
            self.transmissionMeasurements = transmissionMeasurements; ## numMeasurements x 1
        




    
    '''------------------------------------------ Setup & Simulation Functions ------------------------------------------'''

    def setup(self):

        for m in range(self.numMaterials):
            for n in range(self.numLengths):
                for j in range(self.numEnergyBins):
                    self.A[m*self.numMaterials+n,j] =  np.exp(-self.materialMus[m,j]*self.phantomLengths[n])



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


        for j in range(self.numEnergyBins):



            G_j = self.Ws[j]/np.sum(self.A[:,j])
            R_j = 0
            for i in range(self.numMeasurements):
                H_i = 0
                for jhat in range(self.numEnergyBins):
                    H_i += self.A[i,jhat]*self.Ws[jhat]
                if H_i == 0:
                    print("H_i")
                    print(count)
                R_j += (self.A[i,j]*self.transmissionMeasurements[i])/H_i


            self.Ws[j] = G_j*R_j[j]




        # G = np.divide(self.Ws,self.summedOverMeasurements) ## numEnergyBins x1 
        # H = self.A*self.Ws ## numMeasurements x 1
        # R = np.transpose(np.sum([self.A[i,:]*self.transmissionMeasurements[i]/H[i] for i in range(self.numMeasurements)],axis=0)) ## numEnergyBins x 1
        # self.Ws = np.multiply(G,R) 
        #plt.figure()
        #plt.plot(self.Esteps, self.Ws, label="Estimate")
        #plt.plot(self.Esteps, self.energySpectrum, label="Ground Truth", alpha=0.5)
        #plt.plot(self.Esteps,self.initialWs,label="Initial Guess",linestyle="--",alpha=0.8)
        
        #plt.title("Count {0}".format(count))
        return self.getSpectrum(count+1)


    '''------------------------------------------ Visualizing ------------------------------------------'''
    def plotSpectrum(self):
        plt.figure()
        plt.plot(self.Esteps, self.energySpectrum, label="Ground Truth", alpha=0.5)
        plt.plot(self.Esteps,self.initialWs,label="Initial Guess",linestyle="--",alpha=0.8)
        plt.plot(self.Esteps,self.Ws,label="Estimate")
        plt.title("Ground Truth (Total Num Photons: {})".format(np.sum(self.energySpectrum)))
        plt.ylabel("Num Photons")
        plt.xlabel("Energy (keV)")
        plt.legend()
        return 





        
            
            
            
        
        


    

            
            


        




    
        