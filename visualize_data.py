from lib.MyStats import *
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        path = "datasets/data.csv"
    else:
        path = sys.argv[1]
    describe(path, header=False)
    input("~~Enter To Continue~~")
    pair_plot(path, _class=1, header=False, column_range_features=list(range(2, 32)))


    #DESCRIBE
    #See if data is correct in terms of numbers
    #See if data needs to be normalized
    #Are there any missing datas?
    #Skewness result far away from 0 means alot of skewness, not good data feature for AI
    #Kurtosis result high number means the dataset has lots of outliers not good for AI (outliers can be removed)

    #PAIRPLOT
    #Pairplots compares two features over the different classes, in a line plot and scatterplot
    #Scatterplots are useful to find correlations and homogenousity between two features.
    #If one of two features are the same, one of them is not interesting for AI and can be eliminated.
    #Line plots are useful to find correlations between classes in one feature
    #Features that are homogenous or have low variation over the classes are not interesting for AI neither as they have low predictive power
    #CONCLUSIONS
