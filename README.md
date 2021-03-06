# Implementation-of-a-system-that-Describes-the-outcome-of-Linear-Regression
Write a python code that takes input from console in a modular* approach. The console input may contain a set of switches as parameter followed by the program name and their corresponding parameter values. Let's name the program "CT" that stands for "Classification Task".

1. -a algoName (algoName can be "LR". If no "-a" switch is found, default will be linear regression)
2. -f fileName (fileName is the name of the dataset. By default the dataset is "Irish" which is available in a certain package of python) 
3. -n foldNum (foldNum = the number of folds to be used. The default value is 1) 
4. -t testFileName (testFileName is the name of a file name that contains testing data, at least a single row should be there. If "-t" is specified then you need to use the tuples in the file and display the number of accurately classified tuples and wrongly classified tuples, and the assigned class labels performed for each tuples).
5. -r ratio (ratio is a floating point value where 0.7 means 70% of the data will be used for training and the remaining for testing. If this number is not specified then 100% data will be used for training. The testing data will be used exactly as "-t" switch is handled. Moreover, any command with -t and -r switches together should be warned and one of them has to be ignored)
6. -e det/brf (-e means explain and either "det" or "brf" will follow "-e". "det" means detail report and "brf" means brief explanation of the linear regression **.)
7. -c classLevel (classLevel represent the name of the attribute of the dataset that is going to be used as target attribute. In case of this parameter is missing, use the last attribute of the dataset as target class label).  

Sample inputs:  
1. py LR_System.py -a LR 
2. py LR_System.py -f strange.csv -n 3 -r 0.8 -e brf 

** -e switch will be valid only if LR is chosen for -a. Otherwise, warn the user for using invalid combination of switches. But simply ignore the request of explanation. A brief explanation will contain at least the following: Num of Observations, coef, std err, t, P>|t|, R-Squared, Adj. R-squared, F-statistic. And a detail report should contain the aforementioned values as well as their meaning.

Use any Consol to use this system.
