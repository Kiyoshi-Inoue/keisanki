import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math

def gene_data(p_1,p_2,n,r_J,r_1,r_2,r_prop,w_J,w_1,w_2,X1_erro,X2_erro,y_erro):
    #U_iを生成
    U_1=np.random.uniform(low=0.5, high=1, size=(p_1,r_J))
    U_2=np.random.uniform(low=0.5, high=1, size=(p_2,r_J))
    #theta_1を生成
    vec_theta_1_a=np.random.uniform(low=0.5, high=1, size=int(r_J*r_prop))
    vec_theta_1_b=np.zeros(int(r_J-r_J*r_prop))
    vec_theta_1=np.concatenate((vec_theta_1_a,vec_theta_1_b))
    vec_theta_1=vec_theta_1.reshape(1,r_J)
    #U_iとtheta_1を更新
    U_theta_1=np.row_stack((U_1,U_2,vec_theta_1))
    ##QR分解
    U_theta_1_Q,R=np.linalg.qr(U_theta_1)
    ##正規化
    U_theta_1_new=U_theta_1_Q/np.linalg.norm(U_theta_1_Q)
    U_1_new=U_theta_1_new[:p_1,:]
    U_2_new=U_theta_1_new[p_1:p_1+p_2,:]
    theta_1_new=U_theta_1_new[p_1+p_2:,:]

    #S_Jの作成
    diag_joint_matrix = np.diag([w_J] * r_J)
    S_J_old=np.random.uniform(low=0, high=1, size=(r_J,n))
    S_J=np.dot(diag_joint_matrix,S_J_old)
    
    #W_iの生成
    W_1=np.random.uniform(low=0.5, high=1, size=(p_1,r_1))
    W_2=np.random.uniform(low=0.5, high=1, size=(p_2,r_2))
    #theta_2iの生成
    vec_theta_21_a=np.random.uniform(low=0.5, high=1, size=int(r_1*r_prop))
    vec_theta_21_b=np.zeros(int(r_1-r_1*r_prop))
    vec_theta_21=np.concatenate((vec_theta_21_a,vec_theta_21_b))
    vec_theta_21=vec_theta_21.reshape(1,r_1)
    vec_theta_22_a=np.random.uniform(low=0.5, high=1, size=int(r_2*r_prop))
    vec_theta_22_b=np.zeros(int(r_2-r_2*r_prop))
    vec_theta_22=np.concatenate((vec_theta_22_a,vec_theta_22_b))
    vec_theta_22=vec_theta_22.reshape(1,r_2)
    #W_iとtheta_2iの更新
    W_theta_21=np.row_stack((W_1,vec_theta_21))
    ##QR分解
    W_theta_21_Q,R=np.linalg.qr(W_theta_21)
    W_theta_22=np.row_stack((W_2,vec_theta_22))
    ##QR分解
    W_theta_22_Q,R=np.linalg.qr(W_theta_22)
    ##正規化
    W_theta_21_new=W_theta_21_Q/np.linalg.norm(W_theta_21_Q)
    W_theta_22_new=W_theta_22_Q/np.linalg.norm(W_theta_22_Q)
    W_1_new=W_theta_21_new[:p_1,:]
    W_2_new=W_theta_22_new[:p_2,:]
    theta_21_new=W_theta_21_new[p_1:,:]
    theta_22_new=W_theta_22_new[p_2:,:]
    #S_iの生成
    diag_x1_matrix = np.diag([w_1] * r_1)
    S_1_old=np.random.uniform(low=0, high=1, size=(r_1,n))
    P_SJ=S_J.T.dot(np.linalg.inv(S_J.dot(S_J.T))).dot(S_J)
    diag_matrix = np.diag([1] * n)
    P_SJ_C=diag_matrix-P_SJ
    S_1=diag_x1_matrix.dot(S_1_old).dot(P_SJ_C)
    diag_x2_matrix = np.diag([w_2] * r_2)
    S_2_old=np.random.uniform(low=0, high=1, size=(r_2,n))
    S_2=diag_x2_matrix.dot(S_2_old).dot(P_SJ_C)
    #X_iの生成
    X_1=U_1_new.dot(S_J)+W_1_new.dot(S_1)
    var_X1=np.var(X_1)
    var_E1=X1_erro*var_X1
    E_1=np.random.normal(loc=0, scale=np.sqrt(var_E1), size=(p_1,n))
    X_1=X_1+E_1
    X_2=U_2_new.dot(S_J)+W_2_new.dot(S_2)
    var_X2=np.var(X_2)
    var_E2=X2_erro*var_X2
    E_2=np.random.normal(loc=0, scale=np.sqrt(var_E2), size=(p_2,n))
    X_2=X_2+E_2
    #yの生成
    y=theta_1_new.dot(S_J)+theta_21_new.dot(S_1)+theta_22_new.dot(S_2)
    var_y=np.var(y)
    var_e_y=y_erro*var_y
    e_y=np.random.normal(loc=0, scale=np.sqrt(var_e_y), size=(1,n))
    y=y+e_y
    #標準化
    ms = StandardScaler()
    X_1_final = ms.fit_transform(X_1)
    X_2_final = ms.fit_transform(X_2)
    y_final=(y-np.mean(y))/np.std(y)

    return X_1_final,X_2_final,y_final

def sJIVE(eta,times,r_J,r_1,r_2,X_1_or,X_2_or,y_or):
    number_best=0
    threshold=10.0
    times=times
    erro_lis=[]
    erro_best=1000.0
    p_1=X_1_or.shape[0]
    p_2=X_2_or.shape[0]
    n=X_1_or.shape[1]
    X_1=math.sqrt(eta)*X_1_or
    X_2=math.sqrt(eta)*X_2_or
    y=math.sqrt(1-eta)*y_or
    #初期値を入れる
    U_1=U_1_best=math.sqrt(eta)*np.random.uniform(low=-0.1, high=0.1, size=(p_1,r_J))
    U_2=U_2_best=math.sqrt(eta)*np.random.uniform(low=-0.1, high=0.1, size=(p_2,r_J))
    W_1=W_1_best=math.sqrt(eta)*np.random.uniform(low=-0.1, high=0.1, size=(p_1,r_1))
    W_2=W_2_best=math.sqrt(eta)*np.random.uniform(low=-0.1, high=0.1, size=(p_2,r_2))
    theta_1=theta_1_best=math.sqrt(1-eta)*np.random.uniform(low=-0.1, high=0.1, size=int(r_J))
    theta_21=theta_21_best=math.sqrt(1-eta)*np.random.uniform(low=-0.1, high=0.1, size=int(r_1))
    theta_22=theta_22_best=math.sqrt(1-eta)*np.random.uniform(low=-0.1, high=0.1, size=int(r_2))
    S_J=S_J_best=np.random.uniform(low=-0.1, high=0.1, size=(r_J,n))
    S_1=S_1_best=np.random.uniform(low=-0.1, high=0.1, size=(r_1,n))
    S_2=S_2_best=np.random.uniform(low=-0.1, high=0.1, size=(r_2,n))
    hat_X_y_best=np.random.uniform(low=-0.1, high=0.1, size=(p_1+p_2+1,n))

    zeros_1=np.zeros((p_2, r_1))
    zeros_2=np.zeros((p_1, r_2))

    for i in range(times):
        #以下を誤差が収束するまで繰り返し
        #S_Jを更新
        U_theta_1=np.row_stack((U_1,U_2,theta_1)) #1
        X_y=np.row_stack((X_1,X_2,y)) #2
        W_1_S_1=W_1.dot(S_1)
        W_2_S_2=W_2.dot(S_2)
        W_S=np.row_stack((W_1_S_1,W_2_S_2))
        theta_2i_S_i=theta_21.dot(S_1)+theta_22.dot(S_2)
        W_theta_2i_S_i=np.row_stack((W_S,theta_2i_S_i)) #3
        S_J=U_theta_1.T.dot(X_y-W_theta_2i_S_i) #1,2,3からS_Jを更新

        #推定値を計算
        U_theta_1=np.row_stack((U_1,U_2,theta_1))
        W_1_theta_21=np.row_stack((W_1,zeros_1,theta_21))
        W_2_theta_22=np.row_stack((zeros_2,W_2,theta_22))
        hat_X_y=U_theta_1.dot(S_J)+W_1_theta_21.dot(S_1)+W_2_theta_22.dot(S_2) #1,2,3から推定値を算出
        #誤差を計算
        X_y=np.row_stack((X_1,X_2,y))
        erro=np.linalg.norm(X_y-hat_X_y,ord=2)**2
        erro_lis.append(erro)

        if erro<erro_best:
            number_best=i
            erro_best=erro
            S_J_best=S_J
            U_1_best=U_1
            U_2_best=U_2
            theta_1_best=theta_1
            S_1_best=S_1
            W_1_best=W_1
            theta_21_best=theta_21
            S_2_best=S_1
            W_2_best=W_2
            theta_22_best=theta_22
            hat_X_y_best=hat_X_y

        if erro<threshold:
            break

        #U_1,U_2,theta_1を更新
        X_y_joint=X_y-W_theta_2i_S_i
        U_J,sigma_J,VT_J=np.linalg.svd(X_y_joint,full_matrices=False) #特異値分解を行う
        U_theta_1=U_J[:,:r_J]
        U_1=U_theta_1[:p_1,:]
        U_2=U_theta_1[p_1:p_1+p_2,:]
        theta_1=U_theta_1[p_1+p_2:,:]

        #推定値を計算
        U_theta_1=np.row_stack((U_1,U_2,theta_1))
        W_1_theta_21=np.row_stack((W_1,zeros_1,theta_21))
        W_2_theta_22=np.row_stack((zeros_2,W_2,theta_22))
        hat_X_y=U_theta_1.dot(S_J)+W_1_theta_21.dot(S_1)+W_2_theta_22.dot(S_2) #1,2,3から推定値を算出
        #誤差を計算
        X_y=np.row_stack((X_1,X_2,y))
        erro=np.linalg.norm(X_y-hat_X_y,ord=2)**2
        erro_lis.append(erro)

        if erro<erro_best:
            number_best=i
            erro_best=erro
            S_J_best=S_J
            U_1_best=U_1
            U_2_best=U_2
            theta_1_best=theta_1
            S_1_best=S_1
            W_1_best=W_1
            theta_21_best=theta_21
            S_2_best=S_1
            W_2_best=W_2
            theta_22_best=theta_22
            hat_X_y_best=hat_X_y

        if erro<threshold:
            break
        
        #S_1を更新
        W_1_theta_21=np.row_stack((W_1,theta_21)) #1
        y_theta_22S_2=y-theta_22.dot(S_2)
        X_1_y_theta_22S_2=np.row_stack((X_1,y_theta_22S_2)) #2
        U_1_S_J=U_1.dot(S_J)
        theta_1_S_J=theta_1.dot(S_J)
        U_1_S_J_theta_1_S_J=np.row_stack((U_1_S_J,theta_1_S_J)) #3
        P_SJ=S_J.T.dot(np.linalg.inv(S_J.dot(S_J.T))).dot(S_J)
        diag_matrix = np.diag([1] * n)
        P_SJ_C=diag_matrix-P_SJ #Jointの直交補空間を作成 #4
        S_1=W_1_theta_21.T.dot((X_1_y_theta_22S_2-U_1_S_J_theta_1_S_J)).dot(P_SJ_C) #1,2,3,4からS_1を更新

        #推定値を計算
        U_theta_1=np.row_stack((U_1,U_2,theta_1))
        W_1_theta_21=np.row_stack((W_1,zeros_1,theta_21))
        W_2_theta_22=np.row_stack((zeros_2,W_2,theta_22))
        hat_X_y=U_theta_1.dot(S_J)+W_1_theta_21.dot(S_1)+W_2_theta_22.dot(S_2) #1,2,3から推定値を算出
        #誤差を計算
        X_y=np.row_stack((X_1,X_2,y))
        erro=np.linalg.norm(X_y-hat_X_y,ord=2)**2
        erro_lis.append(erro)

        if erro<erro_best:
            number_best=i
            erro_best=erro
            S_J_best=S_J
            U_1_best=U_1
            U_2_best=U_2
            theta_1_best=theta_1
            S_1_best=S_1
            W_1_best=W_1
            theta_21_best=theta_21
            S_2_best=S_1
            W_2_best=W_2
            theta_22_best=theta_22
            hat_X_y_best=hat_X_y


        if erro<threshold:
            break

        #W_1,theta_21を更新
        U_I_1,sigma_1,VT_1=np.linalg.svd((X_1_y_theta_22S_2-U_1_S_J_theta_1_S_J).dot(P_SJ_C),full_matrices=False) #特異値分解を行う
        W_1=U_I_1[:p_1,:r_1]
        zeros_1=np.zeros((p_2, r_1))
        theta_21=U_I_1[p_1:,:r_1]
        W_1_theta_21_re=np.row_stack((W_1,zeros_1,theta_21))

        #推定値を計算
        U_theta_1=np.row_stack((U_1,U_2,theta_1))
        W_1_theta_21=np.row_stack((W_1,zeros_1,theta_21))
        W_2_theta_22=np.row_stack((zeros_2,W_2,theta_22))
        hat_X_y=U_theta_1.dot(S_J)+W_1_theta_21.dot(S_1)+W_2_theta_22.dot(S_2) #1,2,3から推定値を算出
        #誤差を計算
        X_y=np.row_stack((X_1,X_2,y))
        erro=np.linalg.norm(X_y-hat_X_y,ord=2)**2
        erro_lis.append(erro)

        if erro<erro_best:
            number_best=i
            erro_best=erro
            S_J_best=S_J
            U_1_best=U_1
            U_2_best=U_2
            theta_1_best=theta_1
            S_1_best=S_1
            W_1_best=W_1
            theta_21_best=theta_21
            S_2_best=S_1
            W_2_best=W_2
            theta_22_best=theta_22
            hat_X_y_best=hat_X_y

        if erro<threshold:
            break


        #S_2を更新
        W_2_theta_22=np.row_stack((W_2,theta_22)) #1
        y_theta_21S_1=y-theta_21.dot(S_1)
        X_2_y_theta_21S_1=np.row_stack((X_2,y_theta_21S_1)) #2
        U_2_S_J=U_2.dot(S_J)
        theta_1_S_J=theta_1.dot(S_J)
        U_2_S_J_theta_1_S_J=np.row_stack((U_2_S_J,theta_1_S_J)) #3
        S_2=W_2_theta_22.T.dot(X_2_y_theta_21S_1-U_2_S_J_theta_1_S_J).dot(P_SJ_C) #1,2,3からS_2を更新

        #推定値を計算
        U_theta_1=np.row_stack((U_1,U_2,theta_1))
        W_1_theta_21=np.row_stack((W_1,zeros_1,theta_21))
        W_2_theta_22=np.row_stack((zeros_2,W_2,theta_22))
        hat_X_y=U_theta_1.dot(S_J)+W_1_theta_21.dot(S_1)+W_2_theta_22.dot(S_2) #1,2,3から推定値を算出
        #誤差を計算
        X_y=np.row_stack((X_1,X_2,y))
        erro=np.linalg.norm(X_y-hat_X_y,ord=2)**2
        erro_lis.append(erro)

        if erro<erro_best:
            number_best=i
            erro_best=erro
            S_J_best=S_J
            U_1_best=U_1
            U_2_best=U_2
            theta_1_best=theta_1
            S_1_best=S_1
            W_1_best=W_1
            theta_21_best=theta_21
            S_2_best=S_1
            W_2_best=W_2
            theta_22_best=theta_22
            hat_X_y_best=hat_X_y

        if erro<threshold:
            break


        #W_2,theta_22を更新
        U_I_2,sigma_2,VT_2=np.linalg.svd((X_2_y_theta_21S_1-U_2_S_J_theta_1_S_J).dot(P_SJ_C),full_matrices=False) #特異値分解を行う
        zeros_2=np.zeros((p_1, r_2))
        W_2=U_I_2[:p_2,:r_2]
        theta_22=U_I_2[p_2:,:r_2]
        W_2_theta_22_re=np.row_stack((zeros_2,W_2,theta_22))

        #推定値を計算
        hat_X_y=U_theta_1.dot(S_J)+W_1_theta_21_re.dot(S_1)+W_2_theta_22_re.dot(S_2) #1,2,3から推定値を算出
        #誤差を計算
        X_y=np.row_stack((X_1,X_2,y))
        erro=np.linalg.norm(X_y-hat_X_y,ord=2)**2
        erro_lis.append(erro)

        if erro<erro_best:
            number_best=i
            erro_best=erro
            S_J_best=S_J
            U_1_best=U_1
            U_2_best=U_2
            theta_1_best=theta_1
            S_1_best=S_1
            W_1_best=W_1
            theta_21_best=theta_21
            S_2_best=S_1
            W_2_best=W_2
            theta_22_best=theta_22
            hat_X_y_best=hat_X_y

        if erro<threshold:
            break


    return erro_lis,number_best,erro_best,S_J_best,U_1_best,U_2_best,theta_1_best,S_1_best,W_1_best,theta_21_best,S_2_best,W_2_best,theta_22_best,hat_X_y_best

def sJIVE_prediction(X_1_tes,X_2_tes,y_tes,U_1_best,U_2_best,W_1_best,W_2_best,theta_1_best,theta_21_best,theta_22_best,times_tes,threshold_tes):
    n_tes=X_1_tes.shape[1]
    r_J=U_1_best.shape[1]
    r_1=W_1_best.shape[1]
    r_2=W_2_best.shape[1]

    times_tes=times_tes
    erro_tes_best=1000
    erro_tes_lis=[]

    U_1_hat=U_1_best
    U_2_hat=U_2_best
    U_hat=np.row_stack((U_1_hat,U_2_hat))

    W_1_hat=W_1_best
    W_2_hat=W_2_best
    W_hat=np.row_stack((W_1_hat,W_2_hat))

    theta_1_hat=theta_1_best
    theta_21_hat=theta_21_best
    theta_22_hat=theta_22_best

    X_new=np.row_stack((X_1_tes,X_2_tes))

    S_J_new=np.random.uniform(low=-0.1, high=0.1, size=(r_J,n_tes))
    S_1_new=np.random.uniform(low=-0.1, high=0.1, size=(r_1,n_tes))
    S_2_new=np.random.uniform(low=-0.1, high=0.1, size=(r_2,n_tes))

    for j in range(times_tes):
        #推定
        W_1_S_1_new=W_1_hat.dot(S_1_new)
        W_2_S_2_new=W_2_hat.dot(S_2_new)
        W_S_new=np.row_stack((W_1_S_1_new,W_2_S_2_new))

        S_J_new=U_hat.T.dot(X_new-W_S_new)

        #推定値を計算
        hat_X_1_tes=U_1_hat.dot(S_J_new)+W_1_hat.dot(S_1_new)
        hat_X_2_tes=U_2_hat.dot(S_J_new)+W_2_hat.dot(S_2_new)
        hat_X_tes=np.row_stack((hat_X_1_tes,hat_X_2_tes))

        #誤差を計算
        erro_tes=np.linalg.norm(X_1_tes-hat_X_1_tes)**2+np.linalg.norm(X_2_tes-hat_X_2_tes)**2
        erro_tes_lis.append(erro_tes)

        if erro_tes<erro_tes_best:
            number_tes_best=j
            erro_tes_best=erro_tes
            S_J_new_best=S_J_new
            S_1_new_best=S_1_new
            S_2_new_best=S_2_new

        if erro_tes<threshold_tes:
            break

        S_1_new=W_1_hat.T.dot(X_1_tes-U_1_hat.dot(S_J_new))

        #推定値を計算
        hat_X_1_tes=U_1_hat.dot(S_J_new)+W_1_hat.dot(S_1_new)
        hat_X_2_tes=U_2_hat.dot(S_J_new)+W_2_hat.dot(S_2_new)
        hat_X_tes=np.row_stack((hat_X_1_tes,hat_X_2_tes))
        
        #誤差を計算
        erro_tes=np.linalg.norm(X_1_tes-hat_X_1_tes)**2+np.linalg.norm(X_2_tes-hat_X_2_tes)**2
        erro_tes_lis.append(erro_tes)

        if erro_tes<erro_tes_best:
            number_tes_best=j
            erro_tes_best=erro_tes
            S_J_new_best=S_J_new
            S_1_new_best=S_1_new
            S_2_new_best=S_2_new

        if erro_tes<threshold_tes:
            break


        S_2_new=W_2_hat.T.dot(X_2_tes-U_2_hat.dot(S_J_new))
        #推定値を計算
        hat_X_1_tes=U_1_hat.dot(S_J_new)+W_1_hat.dot(S_1_new)
        hat_X_2_tes=U_2_hat.dot(S_J_new)+W_2_hat.dot(S_2_new)
        hat_X_tes=np.row_stack((hat_X_1_tes,hat_X_2_tes))

        #誤差を計算
        erro_tes=np.linalg.norm(X_1_tes-hat_X_1_tes)**2+np.linalg.norm(X_2_tes-hat_X_2_tes)**2
        erro_tes_lis.append(erro_tes)

        if erro_tes<erro_tes_best:
            number_tes_best=j
            erro_tes_best=erro_tes
            S_J_new_best=S_J_new
            S_1_new_best=S_1_new
            S_2_new_best=S_2_new

        if erro_tes<threshold_tes:
            break
    
    y_new=theta_1_hat.dot(S_J_new_best)+theta_21_hat.dot(S_1_new_best)+theta_22_hat.dot(S_2_new_best)
    erro_result=np.linalg.norm(y_tes-y_new)**2

    return erro_tes_lis,number_tes_best,erro_tes_best,S_J_new_best,S_1_new_best,S_2_new_best,y_new,erro_result



