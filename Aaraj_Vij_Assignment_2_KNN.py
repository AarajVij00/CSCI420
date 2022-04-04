import pandas as pd
import timeit

flowerDF = pd.read_csv(r"C:\Users\fllbm\Desktop\Data Mining Assignments\Assignment 2\KNN\KNN_train.csv") # Create dataframe from Iris data set
testDF = pd.read_csv(r"C:\Users\fllbm\Desktop\Data Mining Assignments\Assignment 2\KNN\KNN_test.csv") # Create dataframe from Iris data set

## This function calculates the Euclidean distance of the current point in the dataframe to the specific centroid
def calcDistanceToCentroids(centroidList, dfRow):
    centroidSLC = centroidList[1]
    centroidSWC = centroidList[2]
    centroidPLC = centroidList[3]
    centroidPWC = centroidList[4]
    
    # Calculate Euclidean distance
    insideEq = (centroidSLC - flowerDF.iat[dfRow, 1])**2 + (centroidSWC - flowerDF.iat[dfRow, 2])**2 + (centroidPLC - flowerDF.iat[dfRow, 3])**2 + (centroidPWC - flowerDF.iat[dfRow, 4])**2
    return insideEq**.5


if __name__ == "__main__":
    k = 70 # Set k here

    print("Starting program.")

    # Start timing; I used guidance from this link: https://stackoverflow.com/questions/5622976/how-do-you-calculate-program-run-time-in-python
    startTime = timeit.default_timer()
    
    print("Full training data set is:")
    print(flowerDF)

    print("Full set to be categorized is:")
    print(testDF)

    finalCategorizations = [] #Will be appended with each categorization

    for j in range(len(testDF)): # Iterate through each entry in testDF
        tempDF = flowerDF.copy() #Create a copy of flowerDF for manipulation
        tempDF["distToEntry"] = 0
        centroidList = testDF.iloc[j].to_list()[:-1] #Create a centroidList of current entry

        for i in range(len(flowerDF)): # Iterate through each row of flowerDF
            tempDF.iloc[i, 6] = calcDistanceToCentroids(centroidList, i) # Calculate distance to point

        tempDF = tempDF.sort_values(by=["distToEntry"]) #Sort tempDF by distance
        tempDF.reset_index(drop=True, inplace=True) #Reset indices

        tempDF = tempDF.truncate(after=k-1) #Truncate after k entries
        labelFreq = tempDF["Labels"].value_counts().index.to_list() #Frequency table of labels
        finalCategorizations.append(labelFreq[0])

    for result in finalCategorizations:
        print(result)

    stopTime = timeit.default_timer()
    print(stopTime-startTime) #Stop timing