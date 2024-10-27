import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spi
import numba
import time 

def function(X, image, lambd) :
    return np.sum((X - image)**2)/2 + R(X)*lambd

def R(X) :
    to_return = 0
    for i in range(1, len(X)-1) :
        for j in range(1, np.shape(X)[1]-1) : 
            to_return += (X[i+1][j] - X[i][j])**2 + (X[i][j+1] - X[i][j])**2

    return to_return

@numba.jit(nopython=True, parallel=True)
def gradient(X, image, lambd) :
    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    for i in numba.prange(0, len(X)) :
        for j in numba.prange(0, np.shape(X)[1]) :
            to_return[i][j] = X[i][j] - image[i][j]
    to_return = np.reshape(to_return, (np.shape(X)[0], np.shape(X)[1]))
    return to_return + R_prime(X)*lambd

@numba.jit(nopython=True, parallel=True)
def R_prime(X) :
    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    m,n = np.shape(X)[0], np.shape(X)[1]
    
    to_return[1][1] = 4*X[1][1] - 2*X[2][1] - 2*X[1][2]
    for i in numba.prange(2,m-1):
        to_return[i][1] = 6*X[i][1] - 2*X[i+1][1] - 2*X[i-1][1] - 2*X[i][2]
    for i in numba.prange(1,m-1):
        to_return[i,n-1] = 2*X[i,n-1] - 2*X[i,n-2]
    for j in numba.prange(2,n-1):
        to_return[1][j] = 6*X[1][j] - 2*X[2][j] - 2*X[1][j+1] - 2*X[1][j-1]
    for j in numba.prange(1,n-1):
        to_return[m-1,j] = 2*X[m-1,j] - 2*X[m-2,j]
    for i in numba.prange(2, m-1) :
        for j in numba.prange(2, n-1) :
            to_return[i][j] = 8*X[i][j] - 2*X[i+1][j] - 2*X[i-1][j] - 2*X[i][j+1] - 2*X[i][j-1]
    
    return to_return

def projected_gradient_method(X, image, epsilon, lambd, L) :
    k = 0
    while True :
        grad = gradient(X, image, lambd)
        #frobenius_norm_grad = np.linalg.norm(grad)
        k += 1
        old = X
        X = X - grad/L
        X = np.clip(X, 0, 255)
        G = L*(old-X)
        norm_G = np.linalg.norm(G, 'fro')
        if k%10 == 0 :
            print('iteration :', k)
            #print ('X:', X)
            #print ('gradient:', grad)
            print('G_L norm: ', norm_G)
            print('biggest grad componant:', np.max(grad))
        if norm_G < epsilon :
            break
    
    return X

def read_image(image_path):
    image = plt.imread(image_path)
    if image.max() <= 1:  # Normalize only if needed
        image *= 255
        image = image.astype(np.uint8)
    return image


def question1() :
    image = read_image("son_goku.png")
    epsilon = 1e-5
    lambd = 0.1
    L = np.sqrt(5)*(1+16*lambd)
    X_red = image[:,:,0]
    X_green = image[:,:,1]
    X_blue = image[:,:,2]
    image_red = image[:,:,0]
    image_green = image[:,:,1]
    image_blue = image[:,:,2]
    #grad_X = gradient(X, image_red, lambd)
    #print(grad_X)
    X_new_red = projected_gradient_method(X_red, image_red, epsilon, lambd, L)
    X_new_green = projected_gradient_method(X_green, image_green, epsilon, lambd, L)
    X_new_blue = projected_gradient_method(X_blue, image_blue, epsilon, lambd, L)
    X_new = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    X_new[:,:,0] = X_new_red
    X_new[:,:,1] = X_new_green
    X_new[:,:,2] = X_new_blue

    plt.imshow(X_new)
    plt.savefig("Goku_lambda_0_1.png")
    plt.show()

####QUESTION II #####

def pandemic_model(t, x):
    beta = 0.2
    gamma = 0.05
    S, I, R = x
    dSdt = -beta*S*I
    dIdt = beta*S*I - gamma*I
    dRdt = gamma*I
    return [dSdt, dIdt, dRdt]

def SYNDy(t,x):
    dS = np.dot(all_functions(x),globall[0])
    dI = np.dot(all_functions(x),globall[1])
    dR = np.dot(all_functions(x),globall[2])
    return np.array([dS, dI, dR])

def find_b(X,m) :
    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    for l in range (len(X)) :
        for i in range(m-1) :
            to_return[l][i] = X[l][i+1] - X[l][i]
        to_return[l][m-1] = X[l][m-1] - X[l][m-2]

    return to_return

def all_functions(x) :
    return np.array([x[0], x[1], x[2], x[0]*x[1], x[1]*x[2]])

@numba.jit(nopython=True, parallel=True)
def gradient_q2 (alpha, b, thetas) :
    m = np.shape(thetas)[0]
    p = len(alpha)
    to_return = np.zeros(p)
    temp = 0
    for k in numba.prange(p):
        for i in numba.prange(m):
            temp = -b[i]
            for j in numba.prange(p):
                temp += alpha[j]*thetas[i][j]
            to_return[k] += thetas[i][k]*temp
    return to_return

def prox(alpha , L, lambd) :
    p =len(alpha)
    y = np.zeros(p)
    for i in range(p) :
        y[i] = np.maximum(0, np.abs(alpha[i]) - lambd/L)*np.sign(alpha[i])
    return y

def proximal_gradient_method(alpha, b, theta, L, lambd) :

    k = 0
    while True :
        print("iteration :", k)
        old = alpha
        grad1 = gradient_q2(alpha, b, theta)
        alpha = alpha - (1/L)*grad1
        alpha = prox(alpha,L,lambd)
        k += 1
        G_L = L*(old - alpha)
        norm_G_L = np.linalg.norm(G_L)
        #print("regularizer: ", lambd*np.linalg.norm(alpha, ord=1))
        #print("grad :", grad1)
        #print("gradient norm :", np.linalg.norm(grad1, ord=2))
        #print("my next alpha :", alpha)
        #fonction_value = (1/2)*np.linalg.norm(b-(theta@alpha),ord=2)**2 + lambd*np.linalg.norm(alpha,ord=1)
        #print("valeur de ma fonction:", fonction_value)
        print ('G_L :', norm_G_L)
        if norm_G_L < 1e-5 :
            return alpha
        #time.sleep(3)

globall = 0.0

def question2() :
    #2.1
    m = 201
    p = 5
    n = 3

    fun = pandemic_model
    sol = spi.solve_ivp(fun, [0, m-1], [0.995, 0.005, 0], t_eval=np.linspace(0, m-1, m))
    b = find_b(sol.y, m).T 
    X = sol.y.T   

    #with open("data.txt", "x") as file:
    #    file.write("X matrix \n" + str(X) + "\n")
    #    file.write("b vectors \n" + str(b) + "\n")

    alpha = np.ones((n,p))*0.5
    thetas = np.zeros((m,5))
    for i in range(m):
        thetas[i] = all_functions(X[i])
    L = 1000
    lambd = 1e-1

    for i in range(0,n):
        print("\n Resolution de la " + str(i+1) + "eme edo:")
        alpha[i] = proximal_gradient_method(alpha[i], b[:,i], thetas, L, lambd)
        time.sleep(3)
        print(alpha)

    global globall 
    globall = alpha
    sol2 = spi.solve_ivp(SYNDy, [0, m-1], [0.995, 0.005, 0], t_eval=np.linspace(0, m-1, m))
    X2 = sol2.y.T

    plt.figure()
    plt.title("SIR model")
    plt.xlabel("Temps")
    plt.ylabel("Population")
    plt.plot(np.linspace(0,m-1,m),X2[:,0], color='cyan', linestyle='--', label="SINDy S")
    plt.plot(np.linspace(0,m-1,m),X2[:,1], color='magenta', linestyle='--', label="SINDy I")
    plt.plot(np.linspace(0,m-1,m),X2[:,2], color='yellow', linestyle='--', label="SINDy R")
    plt.plot(np.linspace(0,m-1,m),X[:,0], color='blue', label = "S")
    plt.plot(np.linspace(0,m-1,m),X[:,1], color='green', label ="I")
    plt.plot(np.linspace(0,m-1,m),X[:,2], color='red', label = "R")
    plt.legend(loc = 'upper right')
    plt.show()

    return alpha



if __name__ == "__main__" :
    question2()




"""def minimizer(x, L, lambd) :

    to_return = np.zeros(5)
    y_negs = np.zeros(5)
    y_pos = np.zeros(5)
    
    y_pos[:] = x + lambd/L
    y_negs[:] = x - lambd/L
    for i in range(5) :
        if y_pos[i] > 0 :
            if y_negs[i] < 0 :
                if (y_pos[i]-x[i])**2 + np.abs(y_pos[i])*lambd/L < (x[i]-y_negs[i])**2 + np.abs(y_negs[i])*lambd/L :
                    to_return[i] = y_pos[i]
                else :
                    to_return[i] = y_negs[i]
            else : to_return[i] = y_pos[i]
        else :
            to_return[i] = y_negs[i]

    return to_return"""