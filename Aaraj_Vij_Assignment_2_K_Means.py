import pandas as pd
import random
import timeit

flowerDF = pd.read_csv(r"C:\Users\fllbm\Desktop\Data Mining Assignments\Assignment 2\K_means\K_means_train.csv") # Create dataframe from Iris data set

# These lists contain centroid data
# K = 3 per project specifications
# [RowNum, SLC, SWC, PLC, PWC]
centroid1List = [0, 0, 0, 0, 0]
centroid2List = [0, 0, 0, 0, 0]
centroid3List = [0, 0, 0, 0, 0]

## This function updates the centroid data each run
def updateCentroidAttributes(centroidList):
    rowNum = centroidList[0]
    return [rowNum, flowerDF.iat[rowNum, 1], flowerDF.iat[rowNum, 2], flowerDF.iat[rowNum, 3], flowerDF.iat[rowNum, 4]]

## This function calculates the Euclidean distance of the current point in the dataframe to the specific centroid
def calcDistanceToCentroids(centroidList, dfRow):
    centroidSLC = centroidList[1]
    centroidSWC = centroidList[2]
    centroidPLC = centroidList[3]
    centroidPWC = centroidList[4]

    # Calculate Euclidean distance
    insideEq = (centroidSLC - flowerDF.iat[dfRow, 1])**2 + (centroidSWC - flowerDF.iat[dfRow, 2])**2 + (centroidPLC - flowerDF.iat[dfRow, 3])**2 + (centroidPWC - flowerDF.iat[dfRow, 4])**2
    return insideEq**.5

## This function calculates new centroids of each cluster by averaging every entry in the cluster
def findNewCentroids(centroidNum):
    clusterDF = flowerDF[flowerDF["Labels"] == centroidNum]
    return [0, clusterDF["SepalLengthCm"].mean(), clusterDF["SepalWidthCm"].mean(), clusterDF["PetalLengthCm"].mean(), clusterDF["PetalWidthCm"].mean()]
    
## This function randomly picks three starting centroids
def pickThreeRandom(dfLength):
    return random.sample(range(1, dfLength-1), 3)

if __name__ == "__main__":
    print("Starting program.")

    print("Full DataFrame is:")
    print(flowerDF)

    #### Begin K-Means Clustering ####
    # Start timing; I used guidance from this link: https://stackoverflow.com/questions/5622976/how-do-you-calculate-program-run-time-in-python
    startTime = timeit.default_timer()

    ### Randomly select three starting centroids
    print('Randomly selecting starting centroids...')
    startingCentroids = pickThreeRandom(len(flowerDF))
    print("Starting centroids have IDs:", startingCentroids)

    # Update rows of each centroid to account for indexing
    centroid1List[0] = startingCentroids[0]-1
    centroid2List[0] = startingCentroids[1]-1
    centroid3List[0] = startingCentroids[2]-1

    # Update global centroid values
    centroid1List = updateCentroidAttributes(centroid1List)
    centroid2List = updateCentroidAttributes(centroid2List)
    centroid3List = updateCentroidAttributes(centroid3List)

    ### Using Euclidean distance, cluster all data points to nearest centroid
    print("Calculating distances to centroids...")

    ## Iterate through each entry in flowerDF and assign entry to the cluster of closest centroid
    for i in range(len(flowerDF)):
        listDistances = [0, 0, 0] # Propagate list with Euclidean distances of entry to each centroid
        listDistances[0] = calcDistanceToCentroids(centroid1List, i)
        listDistances[1] = calcDistanceToCentroids(centroid2List, i)
        listDistances[2] = calcDistanceToCentroids(centroid3List, i)
        flowerDF.iat[i, 5] = listDistances.index(min(listDistances))+1 # Pick the shortest distance and assign cluster in flowerDF


    ### Calculate new centroids by averaging all points in each cluster

    centroid1List = findNewCentroids(1) 
    centroid2List = findNewCentroids(2)
    centroid3List = findNewCentroids(3)

    # Keep track of last iteration's centroids
    oldCentroid1List = []
    oldCentroid2List = []
    oldCentroid3List = []
    
    ### While last iteration's and current iteration's centroids are not the same, repeat the process above
    iterationNum = 1
    while(1==1):
        print("Iteration Number: ", iterationNum)

        # Calculate Euclidean distances to each centroid
        for i in range(len(flowerDF)):
            listDistances = [0, 0, 0]
            listDistances[0] = calcDistanceToCentroids(centroid1List, i)
            listDistances[1] = calcDistanceToCentroids(centroid2List, i)
            listDistances[2] = calcDistanceToCentroids(centroid3List, i)
            flowerDF.iat[i, 5] = listDistances.index(min(listDistances))+1

        # Keep track of last iteration's centroids
        oldCentroid1List = centroid1List
        oldCentroid2List = centroid2List
        oldCentroid3List = centroid3List

        # Calculate and update new centroids
        centroid1List = findNewCentroids(1)
        centroid2List = findNewCentroids(2)
        centroid3List = findNewCentroids(3)

        iterationNum = iterationNum+1

        # Break from loop if centroids are identical
        if(centroid1List == oldCentroid1List and centroid2List == oldCentroid2List and centroid3List == oldCentroid3List):
            break

    # Stop timing
    stopTime = timeit.default_timer()

    ### Print Results
    print("Final Clustering Results:")
    print(flowerDF.to_string())

    ### Print Validation Data
    print("Validation Data Clusters:")
    print("Cluster A")
    print(flowerDF.iloc[135, 5]) # Corresponding flowerDF index of 1st row in K_means_Valid.csv
    print(flowerDF.iloc[136, 5]) # Corresponding flowerDF index of 2nd row in K_means_valid.csv
    print("Cluster B")
    print(flowerDF.iloc[137, 5])
    print(flowerDF.iloc[138, 5])
    print("Cluster C")
    print(flowerDF.iloc[139, 5])
    print(flowerDF.iloc[134, 5])

    ### Print Test Data
    print("Test Data Clusters:")
    print(flowerDF.iloc[141, 5]) # Corresponding flowerDF index of 1st row in K_means_test.csv
    print(flowerDF.iloc[142, 5]) # Corresponding flowerDF index of 2nd row in K_means_test.csv
    print(flowerDF.iloc[143, 5])
    print(flowerDF.iloc[144, 5])
    print(flowerDF.iloc[145, 5])
    print(flowerDF.iloc[146, 5])
    print(flowerDF.iloc[147, 5])
    print(flowerDF.iloc[148, 5])
    print(flowerDF.iloc[149, 5])
    print(flowerDF.iloc[150, 5])

    ### Print Timer Data
    print("Time Elapsed:", stopTime-startTime)