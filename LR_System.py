import sys
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
from sklearn.metrics import r2_score

algoName = "LR"    # -a
fileName = "iris"  # -f
foldNum = 1        # -n
testFileName = ""  # -t
ratio = 1          # -r
det_brf = ""       # -e
classLevel = ""    # -c

X = pd.DataFrame({})
Y = pd.DataFrame({})
X_t = pd.DataFrame({})
Y_t = pd.DataFrame({})
X_test = pd.DataFrame({})
Y_test = pd.DataFrame({})
X_train = pd.DataFrame({})
Y_train = pd.DataFrame({})

file = open("Report.txt", "w")


def read_switch():
    global algoName, fileName, foldNum, testFileName, ratio, det_brf, classLevel

    for i in range(1, NumOfParams):
        if sys.argv[i].replace(" ", "") == '-a':
            algoName = sys.argv[i+1]
        elif sys.argv[i].replace(" ", "") == '-f':
            fileName = sys.argv[i+1]
        elif sys.argv[i].replace(" ", "") == '-n':
            foldNum = sys.argv[i + 1]
        elif sys.argv[i].replace(" ", "") == '-t':
            testFileName = sys.argv[i + 1]
        elif sys.argv[i].replace(" ", "") == '-r':
            ratio = sys.argv[i + 1]
        elif sys.argv[i].replace(" ", "") == '-e':
            det_brf = sys.argv[i + 1]
        elif sys.argv[i].replace(" ", "") == '-c':
            classLevel = sys.argv[i + 1]

    print("Algorithm Name is (-a): ", algoName)
    print("File Name is      (-f): ", fileName)
    print("Fold Number is    (-n): ", foldNum)
    print("Test File Name is (-t): ", testFileName)
    print("Ratio is          (-r): ", ratio)
    print("Explanation Type  (-e): ", det_brf)
    print("Class Level is    (-c): ", classLevel)


def read_dataset():
    global fileName
    if fileName == "iris":
        data = datasets.load_iris()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    elif fileName == "boston":
        data = datasets.load_boston()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    elif fileName == "breast_cancer":
        data = datasets.load_breast_cancer()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    elif fileName == "diabetes":
        data = datasets.load_diabetes()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    elif fileName == "digits":
        data = datasets.load_digits()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    elif fileName == "files":
        data = datasets.load_files()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    elif fileName == "linnerud":
        data = datasets.load_linnerud()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    elif fileName == "wine":
        data = datasets.load_wine()
        dataset = pd.DataFrame(data.data, columns=data.feature_names)
    else:
        dataset = pd.read_csv(fileName)
    return dataset


def read_testset():
    global testFileName
    testset = pd.DataFrame({})
    if len(testFileName) != 0:
        testset = pd.read_csv(testFileName)
        return testset
    else:
        return testset


def encoding_dataset(dataset):
    global fileName
    if fileName != "iris" and dataset.shape[0] != 0:
        Label_Encoder = LabelEncoder()
        columns = [column for column in dataset.columns if dataset[column].dtype in ['O']]
        dataset[columns] = dataset[columns].apply(LabelEncoder().fit_transform)
        return dataset
    else:
        return dataset


def dataset_target_split(dataset):
    global X, Y, classLevel
    if len(classLevel) != 0:
        Y = dataset[classLevel]
        X = dataset.drop(classLevel, axis=1)
    else:
        Y = dataset.iloc[:, -1]
        X = dataset
        X  = X.iloc[: , :-1]


def testset_target_split(testset):
    global X_t, Y_t, classLevel

    if testset.shape[0] != 0:
        if len(classLevel) != 0:
            Y_t = testset[classLevel]
            X_t = testset.drop(classLevel, axis=1)
        else:
            Y_t = testset.iloc[:, -1]
            X_t = testset
            X_t = X_t.iloc[:, :-1]
    else:
        X_t = pd.DataFrame({})
        Y_t = pd.DataFrame({})


def dataset_split():
    global X, Y, X_t, Y_t, ratio, X_train, Y_train, X_test, Y_test
    r = float(ratio)
    if (X_t.shape[0] != 0 and Y_t.shape[0] != 0 and r != 1) or (X_t.shape[0] != 0 and Y_t.shape[0] != 0):
        X_train = X
        Y_train = Y
        X_test = X_t
        Y_test = Y_t
        if X_t.shape[0] != 0 and Y_t.shape[0] != 0 and r != 1:
            print("\n**************************** Warning ****************************\n")
            print("*** Both -t and -r switches is used. So, -r switche is ignored. ***")
            print("\n**************************** Warning ****************************\n")
    else:
        if(r == 1.0):
            r = int(r)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=r, random_state=0)
        elif r < 0 or r > 1:
            print("\n************************************ Warning ************************************\n")
            print("Invalid input of switche -r. Should be less than or equal to 1 and greater than 0.")
            print("\n************************************ Warning ************************************\n")
            exit()
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=r, random_state=0)


def nfold():
    global X_train, Y_train, X_test, Y_test, foldNum
    foldNum = int(foldNum)
    if foldNum >= 2:
        n_fold = KFold(n_splits=foldNum, random_state=1, shuffle=True)
        return n_fold
    else:
        return 0


def models():
    global X_train, Y_train, X_test, Y_test, algoName

    if algoName == "LR":
        LR_model = LinearRegression()
        LR_model.fit(X_train, Y_train)
        return LR_model
    else:
        print("\n******************************************** Warning ********************************************\n")
        print("                Invalid input of switche -a. Only Linear Regression ( LR ) available.  ")
        print("\n******************************************** Warning ********************************************\n")
        exit()


def num_of_Observations(X_train):
    global file
    num_of_Observations = X_train.shape[0]
    print("\n\nNum of Observations is : ", num_of_Observations+1)
    if det_brf == "brf":
        file.write("\n\nNum of Observations is : ")
        file.write(str(num_of_Observations + 1))
    elif det_brf == "det":
        file.write("\n\nNum of Observations is : ")
        file.write(str(num_of_Observations + 1))
        file.write("\nThis is the Num of Observations used to build the model.")


def r_2_Score(Y_test, Y_pred):
    global file
    r_2_Score = r2_score(Y_test, Y_pred)
    print('\nR2 score is                   : ', r_2_Score)
    if det_brf == "brf":
        file.write("\n\nR2 score is                  : ")
        file.write(str(r_2_Score))
    elif det_brf == "det":
        file.write("\n\n\nR2 score is                   : ")
        file.write(str(r_2_Score))
        file.write("\n\nR2 score and Accuracy Score are the same which is described below.")


def accuracy_det(Accuracy,accu_per):
    global file
    file.write("\n\n\nAccuracy Score is (R-Squared) : ")
    file.write(str(Accuracy))
    file.write(" ( ")
    file.write(str(accu_per))
    file.write(" % )\n")
    if Accuracy == 1:
        file.write("\nAs the accuracy is 1 this model perfectly fits the the observed data. "
                   "\nThe accuracy of 100% reveals that 100% of the data fit the regression"
                   "\nmodel. Generally, a higher accuracy indicates a better fit for the model.")
    elif Accuracy < 1 and Accuracy > 0.5:
        file.write("\nThe Accuracy value of ")
        file.write(str(np.round(Accuracy, 6)))
        file.write(" would indicate that ")
        file.write(str(accu_per))
        file.write("% of the variance of\nthe dependent variable being studied is explained by the variance of the"
                   "\nindependent variable. For instance, if a accuracy value of ")
        file.write(str(np.round(Accuracy, 6)))
        file.write(" relative to\nits benchmark, that would indicate that ")
        file.write(str(accu_per))
        file.write("% of the variance of the accuracy\nis explained by the variance of its benchmark index.")
        file.write("\nThe accuracy of ")
        file.write(str(accu_per))
        file.write("% reveals that ")
        file.write(str(accu_per))
        file.write("% of the data fit the regression"
                   "\nmodel. Generally, a higher accuracy indicates a better fit for the model.")
    elif Accuracy <= 0.5:
        file.write("\nThe Accuracy value of ")
        file.write(str(np.round(Accuracy, 6)))
        file.write(" would indicate that ")
        file.write(str(accu_per))
        file.write("% of the variance of\nthe dependent variable being studied is explained by the variance of the"
                   "\nindependent variable. For instance, if a accuracy value of ")
        file.write(str(np.round(Accuracy, 6)))
        file.write(" relative to\nits benchmark, that would indicate that ")
        file.write(str(accu_per))
        file.write("% of the variance of the accuracy\nis explained by the variance of its benchmark index.")
        file.write("\nThe accuracy of ")
        file.write(str(accu_per))
        file.write("% reveals that ")
        file.write(str(accu_per))
        file.write("% of the data fit the regression"
                   "\nmodel. Generally, a higher accuracy indicates a better fit for the model.")
        file.write("\nAs the value of accuracy is very low, I think that the model is not linear.\n"
                   "We should try another model.")
        if Accuracy < 0:
            file.write(" And as the Accuracy Score is Negative whic\nmeans that chosen model does not"
                       " follow the trend of the data,\nso fits worse than a horizontal line.")
    else:
        file.write("\nThe Accuracy value of ")
        file.write(str(np.round(Accuracy, 6)))
        file.write(" would indicate that ")
        file.write(str(accu_per))
        file.write("% of the variance of\nthe dependent variable being studied is explained by the variance of the"
                   "\nindependent variable. For instance, if a accuracy value of ")
        file.write(str(np.round(Accuracy, 6)))
        file.write(" relative to\nits benchmark, that would indicate that ")
        file.write(str(accu_per))
        file.write("% of the variance of the accuracy\nis explained by the variance of its benchmark index.")
        file.write("\nThe accuracy of ")
        file.write(str(accu_per))
        file.write("% reveals that ")
        file.write(str(accu_per))
        file.write("% of the data fit the regression"
                   "\nmodel. Generally, a higher accuracy indicates a better fit for the model.")
        file.write("\nBut As the value of accuracy is very high which is more the 100%,"
                   "\nI think that the model is not linear."
                   " We should try another model.")


def accuracy(model, X_test, Y_test):
    global file
    Accuracy = model.score(X_test, Y_test)
    accu_per = np.round(Accuracy * 100, 2)
    print("Accuracy Score is (R-Squared) : ", Accuracy," ( " ,accu_per," % )")
    if det_brf == "brf":
        file.write("\n\nAccuracy Score is(R-Squared) : ")
        file.write(str(Accuracy))
        file.write(" ( ")
        file.write(str(accu_per))
        file.write(" % )")
    elif det_brf == "det":
        accuracy_det(Accuracy, accu_per)


def adj_r_2_Score_det(adj_r_2_Score):
    global file
    file.write("\n\nThis is the modified version of R-squared which is adjusted for the number of variables\nin the "
               "regression. It increases only when an additional variable adds to the explanatory\npower to the regression."
               " It decreases when a predictor improves the model by less than\nexpected by chance. The adjusted R-squared "
               "can be negative, but it is usually not.\nIt is always lower than the R-squared.")


def adj_r_2_Score(Y_test, Y_pred, X_test):
    global file
    r_2_Score = metrics.r2_score(Y_test, Y_pred)
    adj_r_2_Score = 1 - (1 - r_2_Score) * (len(Y_test) - 1) / (len(Y_test) - X_test.shape[1] - 1)
    print("Adj. R-squared is             : ", adj_r_2_Score)
    if det_brf == "brf":
        file.write("\n\nAdj. R-squared is            : ")
        file.write(str(adj_r_2_Score))
    elif det_brf == "det":
        file.write("\n\n\nAdj. R-squared is            : ")
        file.write(str(adj_r_2_Score))
        adj_r_2_Score_det(adj_r_2_Score)


def mean_Squared_Erro_det(mean_Squared_Error):
    global file
    file.write('\n\n\nMean Squared Error (MSE)  is  :  ')
    file.write(str(mean_Squared_Error))
    file.write("\n\nThe Mean Squared Error (MSE) is the average of the square of the errors.\n"
               "The larger the number the larger the error. Here, error means the difference\n"
               "between the observed values and the predicted values. There is no correct value for MSE.\n"
               "Simply put, the lower the value the better and 0 means the model is perfect.")
    if mean_Squared_Error < 0.1:
        file.write("\nWe can a accept the value. As the value of ")
        file.write(str(np.round(mean_Squared_Error, 6)))
        file.write(" close to 0.")
    elif mean_Squared_Error >= 0.1 and mean_Squared_Error <=0.5:
        file.write("\nWe can a accept the value. However, the value of  ")
        file.write(str(np.round(mean_Squared_Error, 6)))
        file.write(" is not very close to 0. We should try to improve the MSE. ")
    else:
        file.write("\nAs the value of MSE is ")
        file.write(str(np.round(mean_Squared_Error, 6)))
        file.write(" which in not very close to 0. We have to improve the MSE. ")


def mean_Squared_Error(Y_test, Y_pred):
    global file
    mean_Squared_Error = metrics.mean_squared_error(Y_test, Y_pred)
    print('Mean Squared Error is         : ', mean_Squared_Error)
    if det_brf == "brf":
        file.write('\n\nMean Squared Error (MSE)  is :  ')
        file.write(str(mean_Squared_Error))
    elif det_brf == "det":
        mean_Squared_Erro_det(mean_Squared_Error)


def mean_Absolute_Erro_det(mean_absolute_error):
    global file
    file.write('\n\n\nMean Absolute Error (MAE) is  :  ')
    file.write(str(mean_absolute_error))
    file.write("\n\nThe Mean Absolute Error (MAE) loss is useful if the training data is corrupted with outliers."
               "\nWe erroneously receive unrealistically huge negative and positive values"
               " in our training environment\nbut not our testing environment."
               "If we only had to give one prediction for all the observations that\ntry to minimize MSE,"
               "then that prediction should be the mean of all target values. But if we try to\nminimize"
               " MAE, that prediction would be the median of all observations. We know that median is more\n"
               "robust to outliers than mean, which consequently makes MAE more robust to outliers than MSE."
               "\nThe Mean Absolute Error (MAE) will be the average vertical distance between each point."
               "\nIn this model the MAE value is ")
    file.write(str(np.round(mean_absolute_error, 6)))
    if mean_absolute_error <= 0.1:
        file.write(" . So the model performance should be excellent.")
    elif mean_absolute_error <= 0.2:
        file.write(" . So the model performance should be good.")
    elif mean_absolute_error <= 0.4:
        file.write(" . So the model performance should not be bad.")
    elif mean_absolute_error <= 0.6:
        file.write(" . So to get better performance from the model we should try to improve the MAE.")
    else:
        file.write(" . So to get better performance from the model we have to improve the MAE.")


def mean_absolute_error(Y_test, Y_pred):
    global file
    mean_absolute_error = metrics.mean_absolute_error(Y_test, Y_pred)
    print('Mean Absolute Error is        : ', mean_absolute_error)
    if det_brf == "brf":
        file.write('\n\nMean Absolute Error (MAE) is :  ')
        file.write(str(mean_absolute_error))
    elif det_brf == "det":
        mean_Absolute_Erro_det(mean_absolute_error)


def coefficient_det(col, coefficient):
    global file
    file.write('\nRegression coefficients represent the mean change in the response variable for one '
               'unit of change\nin the predictor variable while holding other predictors in the model constant. '
               'This statistical\ncontrol that regression provides is important because it isolates the role of'
               ' one variable from\nall of the others in the model. The key to understanding the coefficients is'
               ' to think of them as\nslopes, and they are often called slope coefficients. And the sign of each'
               ' coefficient indicates\nthe direction of the relationship between a predictor variable and'
               ' the response variable. \n\nHere,')

    for i in range(len(coefficient)):
        file.write("\n")
        file.write(col[i])
        file.write(" column coefficient is ")
        file.write(str(np.round(coefficient[i], 6)))
        if coefficient[i] == 0:
            file.write(" . As the coefficient is 0 it implies there is no linear correlation.")
        elif coefficient[i] == 1:
            file.write(" . As the coefficient is ( + 1 ) it implies there is a perfect positive correlation.")
        elif coefficient[i] == -1:
            file.write(" . As the coefficient is ( - 1 ) it implies there is a perfect negative correlation.")
        elif 0.1 >= coefficient[i] >= - 0.1:
            file.write(" . As the coefficient is ")
            file.write(str(np.round(coefficient[i], 6)))
            file.write(" it implies the linear\nrelationship between variables is very weak to nonexistent.")
        elif - 0.1 > coefficient[i]:
            file.write(" . As the coefficient is ")
            file.write(str(np.round(coefficient[i], 6)))
            file.write(" which is negative.\nIt indicates that as the predictor variable increases,"
                       " the response variable decreases.")
        elif 0.1 < coefficient[i]:
            file.write(" . As the coefficient is ")
            file.write(str(np.round(coefficient[i], 6)))
            file.write(" which is positive.\nIt indicates that as the predictor variable increases,"
                       " the response variable also increases.")
        file.write("\n")


def model_coefficient(model,X_train,col):
    global file
    coefficient = model.coef_
    if det_brf == "brf":
        file.write("\n\n\n         --->   Coefficients   <---         \n\n")
    elif det_brf == "det":
        file.write("\n\n\n         --->   Coefficients   <---         \n\n")

    print("\n\n         --->   Coefficients   <---         \n")
    for i in range(len(coefficient)):
        print(col[i], " column coefficient is : ", coefficient[i])
        if det_brf == "brf":
            file.write(col[i])
            file.write(" column coefficient is : ")
            file.write(str(coefficient[i]))
            file.write("\n")
        elif det_brf == "det":
            file.write(col[i])
            file.write(" column coefficient is : ")
            file.write(str(coefficient[i]))
            file.write("\n")
            if i == len(coefficient) - 1:
                coefficient_det(col, coefficient)


def standard_Errors_det(col_with_constant, standard_Errors):
    global file
    file.write('\nThe standard error of the regression, also known as the standard error of the estimate,'
               ' is a measure\nof how accurate a model is, it indicates the average distance between the '
               'observed and regression lines.\nConveniently, it tells how wrong the regression model is on'
               ' average using the units of the response variable.\nSmaller values are better because it '
               'indicates that the observations are closer to the fitted line.\nApproximately 95% of the '
               'observations should fall within plus/minus 2 * standard error of the regression\nfrom the'
               ' regression line, which is also a quick approximation of a 95% prediction interval. The '
               'standard\nerror of the regression might be more important than assessing R-squared. \n\nHere,')

    for i in range(1,len(col_with_constant)):
        file.write("\n")
        file.write(col_with_constant[i])
        file.write(" column Standard Errors is ")
        file.write(str(np.round(standard_Errors[i], 6)))
        file.write(" . Which means  Approximately 95% of the observations\nshould fall within plus/minus ")
        file.write(str(np.round(2 * standard_Errors[i], 6)))
        file.write(" of the regression from the regression line, which is also a quick\napproximation "
                   "of a 95% prediction interval.")
        file.write("\n")


def standard_Errors(v,col_with_constant):
    global file
    standard_Errors = np.sqrt(v)
    if det_brf == "brf":
        file.write("\n\n         --->   Standard Errors   <---         \n\n")
    elif det_brf == "det":
        file.write("\n\n         --->   Standard Errors   <---         \n\n")
    print("\n\n         --->   Standard Errors   <---         \n")
    for i in range(1,len(col_with_constant)):
        print(col_with_constant[i], " Standard Errors is : ", standard_Errors[i])
        if det_brf == "brf":
            file.write(col_with_constant[i])
            file.write(" column Standard Errors is : ")
            file.write(str(standard_Errors[i]))
            file.write("\n")
        elif det_brf == "det":
            file.write(col_with_constant[i])
            file.write(" column Standard Errors is : ")
            file.write(str(standard_Errors[i]))
            file.write("\n")
            if i == len(col_with_constant) - 1:
                standard_Errors_det(col_with_constant, standard_Errors)
    return standard_Errors


def t_value_det(col_with_constant, t_value):
    global file
    file.write("\nWhen we perform a t-test, we are usually trying to find evidence of a significant difference"
               " between population\nmeans (2-sample t) or between the population mean and a hypothesized value"
               " (1-sample t). The t-value measures\nthe size of the difference relative to the variation in our"
               " sample data. Put another way, T is simply the calculated\ndifference represented in units of"
               " standard error. The greater the magnitude of T, the greater the evidence against\nthe null"
               " hypothesis. This means there is greater evidence that there is a significant difference."
               " The closer T is to\n0, the more likely there isn't a significant difference. The larger the"
               " absolute value of the T value, the smaller the\nP value, and the greater the evidence against"
               " the null hypothesis. If we use the t-stat, we can reject the null\nhypothesis if the value of "
               "the t-stat is greater than the value, which corresponds to the level of significance alpha\non "
               "the Normal distribution table. As our level of significance is 0.05 the correspondent t-stat "
               "value is 1.96, thus\nwhen the t-stat reported in the output is higher than 1.96 we can reject the"
               " null hypothesis and our coefficient is\nsignificant at 5 % significance level.\n\nHere,")
    for i in range(1,len(col_with_constant)):
        file.write("\n")
        file.write(col_with_constant[i])
        file.write(" column T value is ")
        file.write(str(np.round(t_value[i], 4)))
        if t_value[i] == 0:
            file.write(" . We can sey that there is not a significant difference.")
        elif 1.96 >= t_value[i] >= - 1.96:
            file.write(" . As the absolute T value is less than the t-stat value we\ncan not reject the null hypothesis.")
        elif 1.96 < t_value[i] or t_value[i] < - 1.96:
            file.write(" . As the absolute T value is greater than the t-stat value\nwe can reject the null hypothesis.")
        file.write("\n")


def t_Value(int_coef, standard_Error, col_with_constant):
    global file
    t_value = int_coef / standard_Error
    if det_brf == "brf":
        file.write("\n\n         --->        T Value        <---         \n\n")
    elif det_brf == "det":
        file.write("\n\n         --->        T Value         <---         \n\n")
    print("\n\n         --->        T Value        <---         \n")
    for i in range(1,len(col_with_constant)):
        print(col_with_constant[i], " T Value is : ", t_value[i])
        if det_brf == "brf":
            file.write(col_with_constant[i])
            file.write(" column T Value is : ")
            file.write(str(t_value[i]))
            file.write("\n")
        elif det_brf == "det":
            file.write(col_with_constant[i])
            file.write(" column T Value is : ")
            file.write(str(t_value[i]))
            file.write("\n")
            if i == len(col_with_constant) - 1:
                t_value_det(col_with_constant, t_value)

    return t_value


def p_value_det(col_with_constant, p_value):
    global file
    file.write("\nThe P value ( Probability Value ) is a number, calculated from a statistical test, that describes how"
               " likely\nwe are to have found a particular set of observations if the null hypothesis were true. P Values"
               " are used in\nhypothesis testing to help decide whether to reject the null hypothesis. The smaller the P"
               " value, the more\nlikely we are to reject the null hypothesis. \n\nHere,")
    for i in range(1,len(col_with_constant)):
        file.write("\n")
        file.write(col_with_constant[i])
        file.write(" column P value is ")
        file.write(str(np.round(p_value[i], 4)))
        if p_value[i] <= 0.05:
            file.write(" . As the P value is less than the significance level which indicates\nthat we can reject the "
                       "null hypothesis")
        elif p_value[i] > 0.05:
            file.write(" . As the P value is greater than the significance level which indicates\nthat we cannot reject"
                       " the null hypothesis")

        file.write("\n")


def p_Value(X_train_with_one,t_value,col_with_constant):
    global file
    p_value = [(1 - stats.t.cdf(np.abs(i), (len(X_train_with_one) - len(X_train_with_one[0])))) * 2 for i in t_value]
    if det_brf == "brf":
        file.write("\n\n         --->   P Value ( P > |t| ) <---         \n\n")
    elif det_brf == "det":
        file.write("\n\n         --->   P Value ( P > |t| ) <---         \n\n")
    print("\n\n         --->   P Value ( P > |t| ) <---         \n")
    for i in range(1,len(col_with_constant)):
        print(col_with_constant[i], " P Value is : ", p_value[i])
        if det_brf == "brf":
            file.write(col_with_constant[i])
            file.write(" column P Value is : ")
            file.write(str(p_value[i]))
            file.write("\n")
        elif det_brf == "det":
            file.write(col_with_constant[i])
            file.write(" column P Value is : ")
            file.write(str(p_value[i]))
            file.write("\n")
            if i == len(col_with_constant) - 1:
                p_value_det(col_with_constant, p_value)
    return p_value


def f_statistic_det(f_statistic):
    global file
    file.write("\n\nF statistic also known as F value is used in regression analysis to identify the means"
               " between two populations\nare significantly different or not. In other words F statistic is ratio of "
               "two variances. F statistic accounts\ncorresponding degrees of freedom to estimate the population "
               "variance. F statistic is almost similar to t statistic.\nT-test states a single variable is"
               " statistically significant or not whereas F test states a group of variables are\nstatistically "
               "significant or not.F statistics are based on the ratio of mean squares. F statistic is the ratio of\nthe"
               " mean square for treatment or between groups with the Mean Square for error or within groups. If "
               "calculated\nF value is greater than the appropriate value of the F critical value, then the null"
               " hypothesis can be rejected.\nIf the p-value associated with the F-statistic is greater than equal 0.05. Then there is "
               "no relationship between\nany of the independent variables and Y. Also if the p-value associated with "
               "the F-statistic less than 0.05.\nThen, at least one independent variable is related to Y.")


def f_statistic(X_train,Y_train):
    global file
    import statsmodels.api as sm
    X_train_with_con = sm.add_constant(X_train)
    lin_reg = sm.OLS(Y_train, X_train_with_con).fit()
    print("\n\nF-statistic is       :", lin_reg.fvalue)
    if det_brf == "brf":
        file.write("\n\nF-statistic is       : ")
        file.write(str(lin_reg.fvalue))
    elif det_brf == "det":
        file.write("\n\nF-statistic is       : ")
        file.write(str(lin_reg.fvalue))
        f_statistic_det(lin_reg.fvalue)
    return lin_reg


def prob_f_statistic_det(prob_f_value):
    global file
    file.write("\n\nThe Probability F Statistic tells us the overall significance of the regression. This is to assess"
               " the significance\nlevel of all the variables together unlike the t statistic that measures it for"
               " individual variables. The null\nhypothesis under this is all the regression coefficients are equal to"
               " zero. Probability F Statistic depicts the\nprobability of null hypothesis being true. As per the above"
               " results, probability is close to zero. This implies\nthat overall the regressions is meaningful.")


def prob_F_statistic(lr):
    global file
    print("Prob (F-statistic)   :", lr.f_pvalue)
    if det_brf == "brf":
        file.write("\n\nProb (F-statistic) is: ")
        file.write(str(lr.f_pvalue))
    elif det_brf == "det":
        file.write("\n\n\nProb (F-statistic) is: ")
        file.write(str(lr.f_pvalue))
        prob_f_statistic_det(lr.f_pvalue)


def det_drf_write_report(model, n_fold):
    global X_train, Y_train, X_test, Y_test, foldNum, det_brf, file

    if foldNum <= 1 and X_test.shape[0] >1:
        Y_pred = model.predict(X_test)
    elif foldNum > X_test.shape[0]:
        print("\n******************************************** Warning ********************************************\n")
        print("Invalid input of switche -n. The number of fold must be less than or equal to the number of samples.")
        print("               With these inputs maximum number of fold can be ",X_test.shape[0] )
        print("\n******************************************** Warning ********************************************\n")
        exit()
    elif foldNum >= 2 and X_test.shape[0]>1:
        Y_pred = cross_val_predict(model, X_test, Y_test, cv=n_fold)
    elif foldNum <= 1 and X_test.shape[0]<= 1:
        X_test = X_train
        Y_test = Y_train
        Y_pred = model.predict(X_train)
    else:
        X_test = X_train
        Y_test = Y_train
        Y_pred = cross_val_predict(model, X_test, Y_test, cv=n_fold)

    if det_brf == "brf":
        file.write("************ Brief explanation of the Linear Regression ************")
    elif det_brf == "det":
        file.write("************ Detail Report of the Linear Regression ************")


    ## Num of Observations

    num_of_Observations(X_train)


    ## R2 score

    r_2_Score(Y_test, Y_pred)


    ## Accuracy Score

    accuracy(model,X_test, Y_test)


    ## Adj. R-squared

    adj_r_2_Score(Y_test, Y_pred, X_test)


    ## Mean Squared Error

    mean_Squared_Error(Y_test, Y_pred)


    ## Mean Absolute Error

    mean_absolute_error(Y_test, Y_pred)


    ##  Column Coefficients

    col = list(X_train.columns)
    model_coefficient(model,X_train,col)


    X_train_with_one = np.append(np.ones((len(X_train), 1)), X_train, axis=1)
    mean_Squared_Error1 = metrics.mean_squared_error(Y_test, Y_pred)
    v = mean_Squared_Error1 * (np.linalg.inv(np.dot(X_train_with_one.T, X_train_with_one)).diagonal())

    col_with_constant = []
    for i in range(len(col)+1):
        if i == 0:
            col_with_constant.append("Constant")
        else:
            col_with_constant.append(col[i-1])
    int_coef = np.append(model.intercept_, model.coef_)


    ## Standard Errors

    standard_Error = standard_Errors(v,col_with_constant)


    ##  T Value

    t_value = t_Value(int_coef, standard_Error, col_with_constant)


    ## P Value ( P > |t| )

    p_value = p_Value(X_train_with_one,t_value,col_with_constant)


    ## F-statistic

    lr = f_statistic(X_train,Y_train)


    ## Prob (F-statistic)

    prob_F_statistic(lr)


    if det_brf == "brf":
        file.write("\n\n\n************ End of the Brief Explanation ************")
        file.write("\n\n************          Thank You.          ************")
    elif det_brf == "det":
        file.write("\n\n\n************   End of the Detail Report   ************")
        file.write("\n\n************          Thank You           ************")

    file.close()


def det_drf_report(model, n_fold):
    global det_brf, algoName

    if algoName != "LR" and (det_brf == "det" or det_brf == "drf"):
        print("\n*************************************** Warning ***************************************\n")
        print("***                           Invalid combination of switches                           ***")
        print("*** To get the detail report or brief explanation algoName have to be linear regression ***")
        print("\n*************************************** Warning ***************************************\n")
    else:
        det_drf_write_report(model, n_fold)


if __name__ == "__main__":

    NumOfParams = len(sys.argv)

    print("Number of Parameter is: ", NumOfParams)

    read_switch()

    dataset = read_dataset()

    testset = read_testset()

    dataset_with_encoding = encoding_dataset(dataset)

    testset_with_encoding = encoding_dataset(testset)

    dataset_target_split(dataset_with_encoding)

    testset_target_split(testset_with_encoding)

    dataset_split()

    model = models()

    n_fold= nfold()

    det_drf_report(model, n_fold)