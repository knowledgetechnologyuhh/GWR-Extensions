# -*- coding: utf-8 -*-
"""
Gamma-GWR example with IRIS dataset (Python 3)

@last-modified: 13 August 2018

@author: German I. Parisi (german.parisi@gmail.com)

Please cite this paper: Parisi, G.I., Tani, J., Weber, C., Wermter, S. (2017) Lifelong Learning of Human Actions with Deep Neural Network Self-Organization. Neural Networks 96:137-149.
"""

import csv
from gammagwr_plus import GammaGWR
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Main ########################################################################

if __name__ == "__main__":
    # Set working path
    #os.getcwd()
    dataFlag = 1         # Import dataset from file
    importFlag = 0       # Import saved network
    trainFlag = 1        # Train AGWR with imported dataset
    saveFlag = 0         # Save trained network to file
    testFlag = 1         # Compute classification accuracy
    plotFlag = 1         # Plot 2D map
        
    if (dataFlag):
       # Load data set
        reader = csv.reader(open("iris.csv","r"),delimiter=',')
        x = list(reader)
        dataSet = np.array(x).astype('float')
        size = dataSet.shape
        
        # Pre-process samples and labels
        labelSet = dataSet[:,size[1]-1]
        dataSet = dataSet[:,0:size[1]-1]
        dimension = len(dataSet[1,:])
    
        # Data normalization
        oDataSet = np.copy(dataSet)
        for i in range(0, size[1]-1):
            maxColumn = max(dataSet[:,i])
            minColumn = min(dataSet[:,i])
            for j in range(0, size[0]):
                oDataSet[j,i] = ( dataSet[j,i] - minColumn ) / ( maxColumn - minColumn )

    if (importFlag):
        
        file = open('myGammaGWR.network','br')
        dataPickle = file.read()
        file.close()
        myGammaGWR = GammaGWR()
        myGammaGWR.__dict__ = pickle.loads(dataPickle)

    if (trainFlag):
        
        trainWithReplay = 0
        batchSize = 3
        numWeights = 3
        
        myGammaGWR = GammaGWR()
        myGammaGWR.initNetwork(dimension, numWeights, numClasses=3, regularized=0)
       
        for i in range(0, len(oDataSet), batchSize):
            print ("Incremental training with mini-batch:", i, "-", i+batchSize)
            
            myGammaGWR.train(oDataSet[i:i+batchSize], labelSet[i:i+batchSize], maxEpochs=2, insertionT=0.85, beta=0.5, epsilon_b=0.2, epsilon_n=0.001, context=1)
            
            if (trainWithReplay):
                myGammaGWR.replay(maxEpochs=2, insertionT=0.85, beta=0.5, epsilon_b=0.2, epsilon_n=0.001, context=0)
    
    if (saveFlag):
        
        file = open('myGammaGWR.network','wb')
        file.write(pickle.dumps(myGammaGWR.__dict__))
        file.close()

    if (testFlag):
        
        bmuWeights, bmuActivation, bmuLabel = myGammaGWR.predict(oDataSet)
        predictedLabels, accuracy = myGammaGWR.predictLabels(bmuLabel, labelSet)
        print ("Classification accuracy: ", accuracy)
 
    if (plotFlag):           
        # Plot network
        # This just plots the first two dimensions of the weight vectors.
        # For better visualization, PCA over weight vectors must be performed.
        ccc = ['black','blue','red'] #'green','yellow','cyan','magenta','0.75','0.15','1'
        fig = plt.figure()
        #sns.set(style="darkgrid")
        classLabels = 1
        ccc = ['black','blue','red'] #'green','yellow','cyan','magenta','0.75','0.15','1'
        for ni in range(len(myGammaGWR.recurrentWeights)):
            plindex = np.argmax(myGammaGWR.alabels[ni])
            if (classLabels):
                plt.scatter(myGammaGWR.recurrentWeights[ni,0,0], myGammaGWR.recurrentWeights[ni,0,1], color=ccc[plindex], alpha=.5)
            else:
                plt.scatter(myGammaGWR.recurrentWeights[ni,0,0], myGammaGWR.recurrentWeights[ni,0,1])
            for nj in range(len(myGammaGWR.recurrentWeights)):
                if (myGammaGWR.edges[ni,nj]>0):
                    plt.plot([myGammaGWR.recurrentWeights[ni,0,0], myGammaGWR.recurrentWeights[nj,0,0]], [myGammaGWR.recurrentWeights[ni,0,1], myGammaGWR.recurrentWeights[nj,0,1]], 'gray', alpha=.3)
        plt.show()