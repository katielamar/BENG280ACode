import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, rescale
from xraydb import material_mu

class SpekEstimations:
    def __init__(self,materials, phantomLengths, initialWs,Es, groundTruth=None,theta=0,I_0_NormalizedImageData=None,transmissionMeasurements=None,plot=False):
        
        self.materials = np.array(materials) # numMaterials x 1
        self.numMaterials = len(materials)
        self.Esteps = Es
        self.energySpectrum = groundTruth
        self.numEnergyBins = len(self.energySpectrum)
        self.phantomLengths = phantomLengths
        self.numLengths = len(self.phantomLengths)
        self.numMeasurements = self.numMaterials*self.numLengths


        assert(self.numEnergyBins == len(self.energySpectrum))


        self.A = np.zeros((self.numMeasurements,self.numEnergyBins))
        self.Ws = initialWs
        self.initialWs = np.array(initialWs)
        

        self.setup(plot)



        if transmissionMeasurements is None and I_0_NormalizedImageData is None:
            self.transmissionMeasurements = self.simulateTransmissionMeasurements()  ## numMeasurements x 1 
        elif imageData is not None:
            self.transmissionMeasurements = self.getTMeasurementsFromImageData(angle,I_0_NormalizedImageData) ## numMeasurements x 1 (correspond to one angle's projection, averaged over detectors)
        else:
            self.transmissionMeasurements = transmissionMeasurements; ## numMeasurements x 1
        




    
    '''------------------------------------------ Setup & Simulation Functions ------------------------------------------'''

    def setup(self,plot):
        for m in range(self.numMaterials):
            for n in range(self.numLengths):
                mu = material_mu(self.materials[m], self.Esteps*10**3)
                for j in range(self.numEnergyBins):
                    self.A[m*self.numMaterials+n,:] = -self.phantomLengths[n] * mu
            if plot:
                plt.figure()
                helper = plt.plot(self.Esteps,mu)
                plt.xlabel("Energy (keV)")
                plt.ylabel("Mass Attenuation Coefficient [1/cm]")
                plt.title(self.materials[m])
                plt.show()

        self.A = np.abs(np.expm1(self.A))


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
    def getSpectrum(self,count=0,plot=False, plotFactor=10):

        '''
        G_j = [w_j^[n]/sumOverMeasurements(Aij)] 
        H_i = sumOverEnergies(Aij*w_j^[n])
        R_j = sumOverMeasurements(Aij*tData_i/H_i)
        w_j^[n+1] =  G_j* R_j

         '''



        if count == self.numMeasurements:
            return self.Ws
        elif count == 0:
            plt.figure()
            helper = plt.plot(self.Esteps,self.energySpectrum, label="Ground Truth")
            helper = plt.plot(self.Esteps,self.initialWs, label="Initial Guess")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Photon Count")
            plt.title("Spectrums")
            plt.legend()
            plt.show()




        for j in range(self.numEnergyBins):
            sumOverMeasurements = np.sum(self.A[:,j])
            if sumOverMeasurements == 0 :
                print("G_j: Zero Division energyBin: j = {0}, Iteration count={1}".format(j,count))
            G_j = self.Ws[j]/np.sum(self.A[:,j])
            R_j = 0
            for i in range(self.numMeasurements):
                H_i = 0
                for jhat in range(self.numEnergyBins):
                    H_i += self.A[i,jhat]*self.Ws[jhat]


                R_j += (self.A[i,j]*self.transmissionMeasurements[i])/max(H_i,0.1)


            self.Ws[j] = G_j*R_j[j]

        if plot and count%plotFactor == 0:
            plt.figure()
            helper = plt.plot(self.Esteps,self.energySpectrum, label="Ground Truth")
            helper = plt.plot(self.Esteps,self.Ws, label="Estimate")

            plt.xlabel("Energy (keV)")
            plt.ylabel("Photon Count")
            plt.title("Iteration {}".format(count))
            plt.legend()
            plt.show()
        return self.getSpectrum(count=count+1,plot=plot, plotFactor=plotFactor)


    '''------------------------------------------ Visualizing ------------------------------------------'''
    def plotSpectrum(self):
        plt.figure()
        helper = plt.plot(self.Esteps, self.energySpectrum, label="Ground Truth", alpha=0.5)
        helper = plt.plot(self.Esteps,self.initialWs,label="Initial Guess",linestyle="--",alpha=0.8)
        helper = plt.plot(self.Esteps,self.Ws,label="Estimate")
        plt.title("Ground Truth (Total Num Photons: {})".format(np.sum(self.energySpectrum)))
        plt.ylabel("Num Photons")
        plt.xlabel("Energy (keV)")
        plt.legend()
        return 





        
            
            
            
        
        


    

            
            


        




    
        