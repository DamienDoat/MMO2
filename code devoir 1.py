import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spi
import numba

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
    to_return[1][n-1] = 2*X[1][n-1] - 2*X[2][n-1]
    to_return[m-1][1] = 2*X[m-1][1] - 2*X[m-2][1]
    for i in numba.prange(2, m-1) :
        to_return[i][1] = 6*X[i][1] - 2*X[i+1][1] - 2*X[i-1][1] - 2*X[i][2]
        to_return[1][i] = 6*X[1][i] - 2*X[2][i] - 2*X[1][i+1] - 2*X[1][i-1]
        to_return[i,n-1] = 2*X[i,n-1] - 2*X[i,n-2]
        to_return[m-1,i] = 2*X[m-1,i] - 2*X[m-2,i]
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
            print ('X:', X)
            print ('gradient:', grad)
            print ('gradient norm:', norm_G)
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
    lambd = 5
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
    plt.show()

def gradient_q2 (alpha, m, b, thetas) :
    ###A FAIRE ENCORE###
    to_return = np.zeros((3,5))
    print ('b :', b)
    print ('alpha :', alpha)
    print ('thetas :', thetas)
    for l in range(3) :
        for i in range(m) :
            for k in range(5) :
                to_return[l][k] -= b[l][i]
                for j in range(5) :
                    to_return[l][k] += thetas[i][j]*alpha[l][j]
                to_return[l][k] *= thetas[i][k]
    print ('gradient :', to_return)
    return to_return
            
            
def pandemic_model(t, x):
        # Example model, replace with actual equations
        beta = 0.3
        gamma = 0.1
        S, I, R = x
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return [dSdt, dIdt, dRdt]

def prox(alpha, L, lambd) :
    y = np.zeros((3,5))
    for i in range(3) :
        y[i] = np.maximum(0, np.abs(alpha[i]) - lambd/L)*np.sign(alpha[i])
    return y

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

def proximal_gradient_method(m, X, b, functions, L, lambd) :

    alpha = np.zeros((3,5))
    thetas = np.zeros((m,5))
    for i in range(m) :
        thetas[i,:] = functions(X[:,i])

    i = 0
    while True :
        grad = gradient_q2(alpha, m, b, thetas)
        alpha = alpha - 1/L*grad
        alpha = prox(alpha,L,lambd)
        i+=1
        print ('norm grad :', np.linalg.norm(grad))
        if np.linalg.norm(grad) < 1e-5 :
            break
        if i > 3 :
            break

    return alpha

def find_b(X,m) :

    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))

    for l in range (len(X)) :
        for i in range(m-1) :
            to_return[l][i] = X[l][i+1] - X[l][i]
        to_return[l][m-1] = X[l][m-1] - X[l][m-2]

    return to_return

def all_functions(x) :
    return [x[0], x[1], x[2], x[0]*x[1], x[1]*x[2]]

def question2() :

    m = 200

    fun = pandemic_model

    X = spi.solve_ivp(fun, [0, m], [0.995, 0.005, 0], t_eval=np.linspace(0, m, m))

    b = find_b(X.y, m) 

    L = 3

    lambd = 1e-3

    functions = all_functions

    alphas = proximal_gradient_method(m, X.y, b, functions, L, lambd)

    return
if __name__ == "__main__" :
    question1()
