import numpy as np
import matplotlib.pyplot as plt

def function(X, image, lambd) :
    return np.sum((X - image)**2)/2 + R(X)*lambd

def R(X) :
    to_return = 0
    for i in range(2, len(X)) :
        for j in range(2, np.shape(X)[1]-1) : 
            to_return += (X[i+1][j] - X[i][j])**2 + (X[i][j+1] - X[i][j])**2

    return to_return

def gradient(X, image, lambd) :
    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    for i in range(1, len(X)) :
        for j in range(2, np.shape(X)[1]-1) :
            to_return[i][j] = X[i][j] - image[i][j]
    to_return = np.reshape(to_return, (np.shape(X)[0], np.shape(X)[1]))
    return to_return + R_prime(X)*lambd

def R_prime(X) :
    to_return = np.zeros((np.shape(X)[0], np.shape(X)[1]))
    for i in range(2, len(X)-1) :
        for j in range(2, np.shape(X)[1]-1) :
            to_return[i][j] = 2*X[i][j]*4 - 2*X[i+1][j] - 2*X[i-1][j] - 2*X[i][j+1] - 2*X[i][j-1]
    return to_return

def projected_gradient_method(X, image, epsilon, lambd, L) :
    while True :
        plt.figure()
        plt.imshow(X)
        plt.show()
        print ('X:', X)
        print ('gradient:', gradient(X, image, lambd))
        X = X - gradient(X, image, lambd)/L
        for i in range(1, len(X)) :
            for j in range(1, np.shape(X)[1]) :
                if X[i][j] < 0 :
                    X[i][j] = 0
                if X[i][j] > 255 :
                    X[i][j] = 255
        if np.linalg.norm(gradient(X, image, lambd)) < epsilon :
            break
    
    return X

def read_image(image_path) :
    image = plt.imread(image_path)
    return image

def question1() :
    image = read_image("son_goku.png")
    epsilon = 1e-5
    lambd = .5
    L = 1
    X = np.random.rand(image.shape[0], image.shape[1])
    print (X.shape)
    image_red = image[:,:,0]
    image_green = image[:,:,1]
    image_blue = image[:,:,2]
    grad_X = gradient(X, image_red, lambd)
    print(grad_X)
    X_new_red = projected_gradient_method(X, image_red, epsilon, lambd, L)
    X_new_green = projected_gradient_method(X, image_green, epsilon, lambd, L)
    X_new_blue = projected_gradient_method(X, image_blue, epsilon, lambd, L)
    X_new = np.zeros((image.shape[0], image.shape[1], 3))
    X_new[:,:,0] = X_new_red
    X_new[:,:,1] = X_new_green
    X_new[:,:,2] = X_new_blue
    plt.imshow(X_new)

def question2() :
    

    return
if __name__ == "__main__" :
    question2()
