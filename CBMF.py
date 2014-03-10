# coding:UTF-8
import numpy as np
import time
import math

import xlwt
from pylab import *
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import random


# Function:Creatting Test data
# Input 
# R : Rating Matrix
# Output
# R_Test : Test Rating Matrix
def Creating_Testdata(R):
    R_test = R.copy()
    Rating_List = []  # Rating List
    for user in xrange(R.shape[0]):
        for item in xrange(R.shape[1]):
            if R[user, item] > 0:
                ID_Pair = (user, item)
                Rating_List.append(ID_Pair)
    
    Rating_List_copy = list(Rating_List)
    TestID_List = []
    TestID_Matrix = np.zeros([len(Rating_List) / 2, 2])
    
    # Create Test Data
    for i in xrange(len(Rating_List) / 2):
        ID = random.randint(0, (len(Rating_List_copy) - 1))
        R_test[ Rating_List_copy[ID][0], Rating_List_copy[ID][1] ] = 0.0
        TestID_List.append(Rating_List_copy[ID])
        TestID_Matrix[i, 0] = Rating_List_copy[ID][0]
        TestID_Matrix[i, 1] = Rating_List_copy[ID][1]
        Rating_List_copy.remove(Rating_List_copy[ID])
        
    return R_test, TestID_List, TestID_Matrix 


# Content-boosted Matrix Factorization Updating Function
# R : Rating Matrix (943 * 1687)
# P : User Matrix (943 * K)
# X : Item * Item's Attribute Matrix(1687 * 19)
# Phi : Item's Attribute Matrix (19 * K)
# K : Number of Latent Feature

def cbmf_agd(R, P, X, Phi, K, steps, alpha=0.0002, beta=0.02):
    X = X.T  # Transpose X
    Phi = Phi.T  # K*I

    N = R.shape[0]  # Number of Users
    M = R.shape[1]  # Number of Items
        
    Error_List = []
    Step_List = [] 
    for step in xrange(steps):
        for i in xrange( N ):
            for j in xrange( M ):
                if R[ i, j ] > 0:
                    Q = np.dot( Phi, X[:, j] )
                    eij = R[ i, j ] - np.dot( P[i, :], Q )
                    Phi_Temp = Phi        
                    P_Temp = P[i, :] 
                    P[i,:] = P_Temp + alpha * ( eij[0,0] * (np.dot(Phi_Temp,X[:,j])).T - beta * P_Temp)
                    # Phi = Phi_Temp +  alpha * ( eij [0,0] * np.dot(X[:,j], P_Temp).T - beta * Phi_Temp)
                
                    """
                    for k in xrange(K):
                        P_Temp = P[i, k]                                
                        # P[i,k] = P[i,k] + alpha * ( eij[0,0] * np.dot(Phi[k,:],X[:,j]) - beta * P[i,k])
                        P[i, k] = P_Temp + alpha * (eij[0, 0] * np.dot(Phi[k, :], X[:, j]) - beta * P_Temp)                        
                        # Phi[k,:] = Phi[k,:] +  alpha * ( eij [0,0] * np.dot(X[:,j], P[i,k]).T - beta * Phi[k,:])
                    """    
                    Phi = Phi_Temp + alpha * (eij [0, 0] * np.dot(X[:, j], P_Temp).T - beta * Phi_Temp)
                    
        Error = 0
        e = 0
        t = 0.0
        for i in xrange( N ):
            for j in xrange( M ):
                if R[i, j] > 0:
                    t += 1
                    Q = np.dot(Phi,X)
                    Error = Error + abs(R[i, j] - np.dot(np.dot(P[i, :], Phi), X[:, j]))
                    e = e + pow(R[i,j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in xrange(K):
                        e = e + (beta / 2) * (pow(P[i,k], 2) + pow(Q[k,j], 2))
                    
        Error = (1 / t) * Error
        Error_List.append(Error[0, 0])
        Step_List.append(step)
        if step % 10 == 0 :  
            print "%4d Steps - MAE : %f, e : %f" % (step, Error, e)
            
        if Error < 0.005:
            break
    print "e:",e
    Q = np.dot(Phi, X)
    Error_Array = np.array(Error_List)
    Step_Array = np.array(Step_List)
    
    return P, Q.T, Error, Error_Array, Step_Array

# Load Training Data Function
def loadMovieLens(path='F:/ICE document/cbmf/movielens'):
    movies = {}
    Item_List = []
    Attribute_List = []
    for line in open(path + '/u.item'):
        (id, title, r_date, vr_date, URL, unknown, action, adventure, anime
            , child, comedy, crime, document, drama, fantasy, noir, horror, music
             , mystery, romance, sci_fi, thriller, war, western) = line.split('|')[0:24]
             
        # Item Ingredient(genre)     
        genre = [int(unknown), int(action), int(adventure), int(anime)
                    , int(child), int(comedy), int(crime), int(document), int(drama), int(fantasy), int(noir), int(horror), int(music)
                    , int(mystery), int(romance), int(sci_fi), int(thriller), int(war), int(western)]
        movies[id] = title
        if id not in Item_List : 
            Item_List.append(id)
            Attribute_List.append(genre)
        
    # Load data
    prefs = {}
    for line in open(path + '/u1.base'):  # u.data
        # (User ID,Movie ID,Evaluation,Time Stamp)
        (user, movieid, rating, ts) = line.split('\t')
        prefs.setdefault(user, {})
        prefs[user][movieid] = float(rating)
    return Item_List, Attribute_List, prefs


def convert_matrix(Item_List, prefs):    
    # prefs : Dictionary type, Key:user, Item:rating list  
    User_List = []
    RatingIndex_List_alluser = []
    item_biases = []
    user_biases = []

    # Creating User List 
    for user in prefs:
        if user not in User_List : User_List.append(user)
    
    # Creating Rating List
    for u in User_List:
        RatingIndex_List_eachuser = []        
        for t in Item_List:
            # user evaluated item 
            if t in prefs[u]:RatingIndex_List_eachuser.append(prefs[u][t])
            # No rating
            else:RatingIndex_List_eachuser.append(0.0)         
        RatingIndex_List_alluser.append(RatingIndex_List_eachuser)
    
    Ratings_array = np.array(RatingIndex_List_alluser)
    # R contains ratings in u1.base data set
    R = np.matrix(Ratings_array)

    # get user biases
    User_Biases = {}
    for user in xrange(len(User_List)):
        observed_count = 0.0
        for item in xrange(len(Item_List)):
            if R[user, item] > 0:
                observed_count += 1.0
        User_Biases.setdefault(user, 0.0)
        User_Biases[user] += np.sum(R[user, :]) / observed_count

    # get item biases
    Item_Biases = {}
    for item in xrange(len(Item_List)):
        observed_count = 0.0
        temp_sum = 0.0
        for user in xrange(len(User_List)):
            if R[user, item] > 0:
                observed_count += 1.0
                temp_sum += R[user, item]
        Item_Biases.setdefault(item, 0.0)
        if observed_count != 0:
            Item_Biases[item] += np.sum(R[:, item]) / observed_count
    
    RB = np.zeros([len(User_List), len(Item_List)])
    RB = np.matrix(RB)    
    for u in xrange(len(User_List)):
        for i in xrange(len(Item_List)):
            if R[u, i] > 0:
                RB[u, i] = R[u, i] - User_Biases[u] - Item_Biases[i]

    return User_List, R, RB 


# RMSE, MAE : Matrix ver
def RMSE(result, answer, TestIndex):    
    RMSE = 0
    MAE = 0
    T = len(TestIndex) 
    print "T:", T
    for id in xrange(T):
        RMSE = RMSE + pow((result[TestIndex[id, 0], TestIndex[id, 1]] - answer[TestIndex[id, 0], TestIndex[id, 1]]), 2)
        MAE = MAE + abs(result[TestIndex[id, 0], TestIndex[id, 1]] - answer[TestIndex[id, 0], TestIndex[id, 1]])
             
    RMSE = math.sqrt(RMSE / T)
    MAE = MAE / T
    return RMSE, MAE
    
###############################################################################

if __name__ == "__main__":

    time1 = time.clock()
    
    # Parameters
    method = 0
    step_num = 5000  # 5000
    K = 5  # n_f
    alpha_temp = 0.0002  # alpha    0.0002
    beta_temp = 0.02  # beta
    Test_Num = 1

    if method == 0:
        
        print " Content-Boosted Matrix Factorization "
        print " Execution Start Time: ", time.ctime()
                
        Item_List, Attribute_List, prefs = loadMovieLens()

        # User_List,Attribute_List,R,rm = make_matrix( Item_List,prefs )
        User_List, R, R_Biases = convert_matrix(Item_List, prefs)

        # X : Item * Ingredient Matrix
        X = np.array(Attribute_List)  # Ingredient Matrix
        X = np.matrix(X)  # Ingredient Matrix
        
        # Number of Users
        N = R.shape[0]      
        print "Number of Users : ", N
        
        # Number of Items
        M = R.shape[1]  # n_r
        print "Number of Items : ", M
        
        # Number of Ingredients
        I = X.shape[1]  # n_i
        print "Number of Attribute  : ", I
        
        # P : Initial User Matrix
        P = np.random.rand(N, K)
        P = np.matrix(P)
        
        # Phi : Initial Ingredient Matrix
        Phi = np.random.rand(I, K)
        Phi = np.matrix(Phi)
        
        # R_Test : Test Rating Matrix
        # remove list : Missing ratings List
        R_Test, TestIndex, TestIndex_Matrix = Creating_Testdata(R)
        # R_Test, TestIndex, TestIndex_Matrix = Creating_Testdata(R_Biases)
        
        # R_Test = Creating_Testdata_addbiases( R_Biases, TestIndex)
        
        print "Number of Test Data : ", len(TestIndex)
                    
        # nP, nQ , mae = content_boosted_mf(R_Test, P, X, Phi, K, steps=step_num, alpha=alpha_temp, beta=beta_temp)
        nP, nQ , mae, e_array, s_array = cbmf_agd(R_Test, P, X, Phi, K, steps=step_num, alpha=alpha_temp, beta=beta_temp)

        Result = np.dot(nP, np.transpose(nQ))
        
        # Result Data Truncation
        for ti in TestIndex:
                if Result[ti[0], ti[1]] > 5:   Result[ti[0], ti[1]] = 5
                if Result[ti[0], ti[1]] < 1:   Result[ti[0], ti[1]] = 0
        
        print "Result"
        print Result
        print "Result shape : ", Result.shape
        print "Phi shape : ", Phi.shape
        print "X shape : ", X.shape
        print "nP shape : ", nP.shape
        print "nQ shape : ", nQ.shape
        print "R_Test shape : ", R_Test.shape
        print "K:", K
        print "Steps : ", step_num
        print "Final MAE : ", mae[0, 0]
        
        # rmse, mae = RMSE(Result, R_Biases, TestIndex_Matrix)
        rmse, mae = RMSE(Result, R, TestIndex_Matrix)
        
        
        print "RMSE:", rmse
        print "MAE:", mae
                
        time2 = time.clock()
        print "Execution End Time: ", time.ctime()
        ptime = time2 - time1
        ptime = float(ptime)
        print "Processing Time : %f seconds(%f minutes)" % (ptime, ptime / 60.0)
    
    
    
# Content-Boosted Matrix Factorization(Matrix Version)
    if method == 3:
        
        print "Content-Boosted Matrix Factorization 3(Matrix Version)"
        print "Execution Start Time: ", time.ctime()
        R = np.genfromtxt('R2.csv', delimiter=',')
        R = np.matrix(R)
        RB = np.genfromtxt('RB.csv', delimiter=',')
        RB = np.matrix(RB)
        X = np.genfromtxt('X2.csv', delimiter=',')
        X = np.matrix(X)
        print "Read Preference Data"
        
        # Number of Users
        N = R.shape[0]      
        
        # Number of Items
        M = R.shape[1]  # n_r
        
        # Number of Ingredients
        I = X.shape[1]  # n_i
        
        # P : Initial User Matrix
        P = np.random.rand(N, K)
        P = np.matrix(P)
        
        # Phi : Initial Ingredient Matrix
        Phi = np.random.rand(I, K)
        Phi = np.matrix(Phi)
        
        # R_Test : Test Rating Matrix
        # remove list : Missing ratings List
        # R_Test, TestIndex,TestIndex_Matrix = Creating_Testdata( R )
        # np.savetxt("R_Test.csv",R_Test, fmt="%.0f",delimiter=",")
        # np.savetxt("TestIndex.csv",TestIndex_Matrix, fmt="%.0f",delimiter=",")
        
        TestIndex = np.genfromtxt('Test_Id.csv', delimiter=',')
        R_Test = np.genfromtxt('R_test.csv', delimiter=',')
        
        # R_Biases, TestIndex = Creating_Testdata( R )
        # R_Test = Creating_Testdata_addbiases( RB, TestIndex)
        
        print "Number of Test Data : ", TestIndex.shape[0]
        
        
        MAE_List = []
        for i  in xrange(Test_Num):
            print  "Test Number : ", i     
            # P : Initial User Matrix
            P = np.random.rand(N, K)
            P = np.matrix(P)
        
            # Phi : Initial Ingredient Matrix
            Phi = np.random.rand(I, K)
            Phi = np.matrix(Phi)       
            # nP, nQ , mae = content_boosted_mf(R_Test, P, X, Phi, K, steps=step_num, alpha=alpha_temp, beta=beta_temp)
            nP, nQ , mae , e_array, s_array = cbmf_agd(R_Test, P, X, Phi, K, steps=step_num, alpha=alpha_temp, beta=beta_temp)
            plt.plot(s_array, e_array)
            plt.title('Content-Boosted Matrix Factorization')
            plt.grid()
            plt.xlabel('Step')
            plt.ylabel('Error')
            savefig('CBMF_Result.png')
            show()
            Result = np.dot(nP, np.transpose(nQ))
        
            # Result Data Truncation
            for ti in xrange(TestIndex.shape[0]):
                if Result[TestIndex[ti, 0], TestIndex[ti, 1]] > 5:   Result[TestIndex[ti, 0], TestIndex[ti, 1]] = 5
                if Result[TestIndex[ti, 0], TestIndex[ti, 1]] < 1:   Result[TestIndex[ti, 0], TestIndex[ti, 1]] = 1
        
            print "Result shape : ", Result.shape
            print "nP shape : ", nP.shape
            print "nQ shape : ", nQ.shape
            print "K:", K
            print "Steps : ", step_num
            print "Final MAE : ", mae[0, 0]
        
            rmse, mae = RMSE(Result, R, TestIndex)
            print "RMSE:%f,\t MAE:%f" % (rmse, mae)
            MAE_List.append(mae)     
            
        print " ----- MAE Average -----"
        print sum(MAE_List) / len(MAE_List)
        
        time2 = time.clock()
        print "Execution End Time: ", time.ctime()
        ptime = time2 - time1
        ptime = float(ptime)
        print "Processing Time : %f seconds(%f minutes)" % (ptime, ptime / 60.0)

        
