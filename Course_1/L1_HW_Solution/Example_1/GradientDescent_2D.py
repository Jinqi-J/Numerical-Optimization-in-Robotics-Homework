import numpy as np
import matplotlib.pyplot as plt

def Rosenbrock(x):
    return 100*(x[0]**2.0 - x[1])**2.0 + (x[0] - 1)**2

def RosenbrockGradient(x):
    gradX1 = 400 * x[0] * (x[0]**2 - x[1]) + 2*(x[0] - 1)
    gradX2 = -200 * (x[0]**2 - x[1])  
    grad = np.array([gradX1, gradX2])
    return grad

def Armijo(x, grad):
    c = 0.1
    tau = 1
    x1 = x[0] - tau * grad[0]
    x2 = x[1] - tau * grad[1]
    nextX = np.array([x1, x2])

    while Rosenbrock(nextX) > Rosenbrock(x) + (c * tau) * np.dot(grad, grad):
        tau *= 0.5
        x1 = x[0] - tau * grad[0]
        x2 = x[1] - tau * grad[1]
        nextX = np.array([x1, x2])

    alpha = tau
    return alpha

def LineSearch(x0):
    pointList = x0

    iter = 1
    maxIter = 5000
    
    x = x0
    error = 10
    tolerance = 0.01

    while (iter < maxIter) and (error > tolerance):
        grad = RosenbrockGradient(x)
        error = np.linalg.norm(grad)

        alpha = Armijo(x, grad)

        X0 = x[0] - alpha * grad[0]
        X1 = x[1] - alpha * grad[1]
        x = np.array([X0, X1])

        iter += 1
        pointList = np.row_stack((pointList, x))

        print("Iteration: ", iter, ", Error", error, ", Local Minimum: ", x)

    return x, pointList

if __name__ == '__main__':
    x0 = np.array([0, 0])
    globalMinimum, pointList = LineSearch(x0)

    x = np.arange(-0.5, 1.5, 0.1)
    y = np.arange(-0.5, 1.5, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = 100*(X**2.0 - Y)**2.0 + (X - 1)**2
    
    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, Z)
    plt.contour(X, Y, Z)

    lastI = 0
    for i in range(pointList.shape[0] - 1):
        
        if i % 300 == 0:
            plt.scatter(pointList[i, 0], pointList[i, 1], s=10)
            
            xAxis = np.array([pointList[lastI, 0], pointList[i, 0]])
            yAxis = np.array([pointList[lastI, 1], pointList[i, 1]])
            lastI = i
            
            plt.plot(xAxis, yAxis)
            plt.pause(0.1)
            
    plt.show()