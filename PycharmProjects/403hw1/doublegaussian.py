import scipy.special
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt



def p(x,y):
    A = 3.93134
    B = 3.61447
    x1 = .5
    y1 = .5
    x2 = .65
    y2 = .75
    s1 = .2
    s2 = .04
    p= A*np.exp(-((((x-x1)**2)+((y-y1)**2))/(2*(s1**2)))) + B*np.exp(-((((x-x2)**2)+((y-y2)**2))/(2*(s2**2))))
    return p

def px(x):
    # type: (object) -> object
    A = 3.93134
    B = 3.61447
    x1 = .5
    y1 = .5
    x2 = .65
    y2 = .75
    s1 = .2
    s2 = .04
    p = np.sqrt((np.pi) / 2) * s1 * (A * np.exp(-(((x - x1) ** 2) / (2 * (s1 ** 2)))) * (
    scipy.special.erf((1 - y1) / (np.sqrt(2) * s1)) + scipy.special.erf((y1) / (np.sqrt(2) * s1)))) + np.sqrt(
        (np.pi) / 2) * s2 * (B * np.exp(-(((x - x2) ** 2) / (2 * (s2 ** 2)))) * (
    scipy.special.erf((1 - y2) / (np.sqrt(2) * s2)) + scipy.special.erf((y2) / (np.sqrt(2) * s2))))
    return p

def py(y):
    A = 3.93134
    B = 3.61447
    x1 = .5
    y1 = .5
    x2 = .65
    y2 = .75
    s1 = .2
    s2 = .04
    p = np.sqrt((np.pi) / 2) * s1 * (A * np.exp(-(((y - y1) ** 2) / (2 * (s1 ** 2)))) * (
    scipy.special.erf((1 - x1) / (np.sqrt(2) * s1)) + scipy.special.erf((x1) / (np.sqrt(2) * s1)))) + np.sqrt(
        (np.pi) / 2) * s2 * (B * np.exp(-(((y - y2) ** 2) / (2 * (s2 ** 2)))) * (
    scipy.special.erf((1 - x2) / (np.sqrt(2) * s2)) + scipy.special.erf((x2) / (np.sqrt(2) * s2))))
    return p



def totaldisplot():
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)

    X, Y = np.meshgrid(x, y)
    z = p(X, Y)
    plt.figure()
    cp = plt.contourf(X, Y, z)
    plt.colorbar(cp)
    plt.title('Double Gaussian')
    plt.xlabel('x ')
    plt.ylabel('y ')
    plt.savefig('totaldist.png')
    plt.show()

def pxa():
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    mypx = px(x)
    plt.plot(x, mypx)
    plt.title('Marginal X Distribution')
    plt.xlabel('x ')
    plt.ylabel('Px')
    plt.savefig('marginalx.png')
    plt.show()

def pya():
    N = 100
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    mypy = py(y)
    plt.plot(y, mypy)
    plt.title('Marginal Y Distribution')
    plt.xlabel('y ')
    plt.ylabel('Py')
    plt.savefig('marginaly.png')
    plt.show()

ansx, errx=scipy.integrate.quad(px,0,1)
#print ansx

ansy, erry=scipy.integrate.quad(py,0,1)
#print ansy
totaldisplot()
#pxa()
#pya()
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

X, Y = np.meshgrid(x, y)
z = p(X, Y)