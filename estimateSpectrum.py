import numpy as np
import matplotlib.pyplot as plt
class estimateSpectrum:
    def __init__(self,factor, muHelper, deltaXvec,energyStart, energyStop, numEnergySteps,Izero, sinogramData, initialWs):
        assert (np.shape(muHelper) == (len(muHelper),numEnergySteps) and 1<= len(muHelper), "muHelper needs shape (numMeasurements,numEnergySteps)")
        assert (np.shape(deltaXvec) == (len(muHelper),), "len(deltaXvec) should be (numMeasurements,)")
        assert(energyStop-energyStart > 0, "energyStop > energyStart")
        assert(numEnergySteps > 0, "numEnergySteps > 0")
        assert(np.shape(sinogramData)[0] == numEnergySteps, "Data should have shape (numEnergySteps,M,N)")
        assert(np.shape(sinogramData)[1] == np.shape(Izero)[0] and np.shape(sinogramData)[2] == np.shape(Izero)[1], "Izero should have shape (numAngles, numDetectors)")
        assert(np.shape(initialWs)==(numEnergySteps, np.shape(sinogramData)[1],np.shape(sinogramData)[2]), "Initial Ws must have shape (numEnergySteps, numAngles, numDetectors)")

        self.mu_vec = muHelper
        self.deltaXs = deltaXvec
        self.numMeasurements = len(muHelper)
        self.numEnergySteps = numEnergySteps
        self.energyStepSize = (energyStop-energyStart)/numEnergySteps
        self.energyStart = energyStart
        self.energyStop = energyStop
        self.numAngles = np.shape(sinogramData)[1]
        self.numDetectors = np.shape(sinogramData)[2]

        self.data = sinogramData  ## numMeasurements x numAngles x numDetectors
        self.Izero = Izero ##numAngles x numDetectors
        self.raw = -factor * [np.divide(np.log(self.data[0]), Izero) for i in range(numEnergySteps)] ## numMeasurements x numAngles x numDetectors
        self.exp =np.exp(np.multiply(np.transpose([deltaXvec for i in range(numEnergySteps)]),muHelper)) ## numMeasurements x numEnergySteps
        self.sumAlongMeasurements = np.sum(self.exp, axis=0) ##numEnergySteps x 1
        self.sumAlongEnergySteps = np.sum(self.exp, axis=1) ##numMeasurements x 1
        self.Ws = initialWs ## numEnergySteps x numAngles x numDetectors

    def getEstimate(self,numSteps):
        for i in range(numSteps):
            for (theta, d) in zip(range(self.numAngles), range(self.numDetectors)):
                self.Ws[:,theta,d] = self.update(self.Ws[:,theta,d], self.raw[:,theta,d])
        return self.Ws

    def update(self,lastWs,raw):
        scaleHelper = sum(self.exp * np.transpose(lastWs)) # numMeasurements x 1
        scale = np.divide(self.exp, np.transpose([scaleHelper for j in range(self.numEnergySteps)])) # numMeasurements x numEnergySteps
        nextWs = np.multiply(np.divide(lastWs*self.numMeasurements,self.sumAlongMeasurements), np.transpose(np.transpose(raw)*scale)) ## numEnergySteps x 1
        return nextWs

    def compressDims(self):
        ## TODO: How do I handle multiple thetas + detectors?
        return

    def plotEnergySpectrum(self,title, theta, detector):
        plt.figure()

        Es = np.arange(self.energyStart,self.energyStop,self.numEnergySteps)
        plt.plot(Es, self.Ws[:, theta,detector],markersize=12)
        plt.xlabel("Energy (keV)")
        plt.ylabel("Counts")
        plt.title(title)
        plt.show()





