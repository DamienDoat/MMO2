import numpy as np
import matplotlib.pyplot as plt

def function(X, image, lambd) :
    return np.sum((X - image)**2)/2 + R(X)*lambd

def R(X) :
    to_return = 0
    for i in range(1, len(X)-1) :
        for j in range(1, np.shape(X)[1]-1) : 
            to_return += (X[i+1][j] - X[i][j])**2 + (X[i][j+1] - X[i][j])**2

    return to_return

def gradient(X, image, lambd) :
    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    for i in range(0, len(X)) :
        for j in range(0, np.shape(X)[1]) :
            to_return[i][j] = X[i][j] - image[i][j]
    to_return = np.reshape(to_return, (np.shape(X)[0], np.shape(X)[1]))
    return to_return + R_prime(X)*lambd

def R_prime(X) :
    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    for i in range(1, len(X)-1) :
        for j in range(1, np.shape(X)[1]-1) :
            to_return[i][j] = 2*X[i][j]*4 - 2*X[i+1][j] - 2*X[i-1][j] - 2*X[i][j+1] - 2*X[i][j-1]
    return to_return

def projected_gradient_method(X, image, epsilon, lambd, L) :
    k = 0
    while True :
        grad = gradient(X, image, lambd)
        frobenius_norm_grad = np.linalg.norm(grad)
        k += 1
        print ('X:', X)
        print ('gradient:', grad)
        print ('gradient norm:', frobenius_norm_grad)
        print('biggest grad componant:', np.max(grad))
        X = X - grad / L
        X = np.clip(X, 0, 255)
        if np.linalg.norm(frobenius_norm_grad) < epsilon :
            break
    
    return X

def read_image(image_path):
    image = plt.imread(image_path)
    if image.max() <= 1:  # Normalize only if needed
        image *= 255
    return image


def question1() :
    image = read_image("son_goku.png")
    epsilon = 1e-5
    lambd = .1
    L = (1+4*lambd)*4
    X = np.random.rand(image.shape[0], image.shape[1]) * 255  # Initialize X in [0, 255]
    print (X.shape)
    image_red = image[:,:,0]
    image_green = image[:,:,1]
    image_blue = image[:,:,2]
    grad_X = gradient(X, image_red, lambd)
    print(grad_X)
    X_new_red = projected_gradient_method(X.copy(), image_red, epsilon, lambd, L)
    X_new_green = projected_gradient_method(X.copy(), image_green, epsilon, lambd, L)
    X_new_blue = projected_gradient_method(X.copy(), image_blue, epsilon, lambd, L)
    X_new = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    X_new[:,:,0] = X_new_red
    X_new[:,:,1] = X_new_green
    X_new[:,:,2] = X_new_blue
    plt.imshow(X_new)

def question2() :
    

    return
if __name__ == "__main__" :
    question1()
