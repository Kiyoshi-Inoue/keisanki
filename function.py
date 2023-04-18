import random
import numpy as np
from sklearn.preprocessing import StandardScaler

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
    U_theta_1_Q,R=np.linalg.qr(U_theta_1)
    U_1_new=U_theta_1_Q[:p_1,:]
    U_2_new=U_theta_1_Q[p_1:p_1+p_2,:]
    theta_1_new=U_theta_1_Q[p_1+p_2:,:]
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
    W_theta_21_Q,R=np.linalg.qr(W_theta_21)
    W_theta_22=np.row_stack((W_2,vec_theta_22))
    W_theta_22_Q,R=np.linalg.qr(W_theta_22)
    W_1_new=W_theta_21_Q[:p_1,:]
    W_2_new=W_theta_22_Q[:p_2,:]
    theta_21_new=W_theta_21_Q[p_1:,:]
    theta_22_new=W_theta_22_Q[p_2:,:]
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
    y_final=ms.fit_transform(y)

    return X_1_final,X_2_final,y_final