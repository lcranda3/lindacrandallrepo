#!/usr/bin/python
# Aran Garcia-Bellido (Feb 2017)
'''
  Example of a simple max likelihood estimation
  inspired in the examples from Cowan, chap.6 and in ROOT
  tutorial Ifit.C
   - generate fake data according to a given function
   - get the parameters with the max likelihood method
   - plot histogram of generated data
   - plot 2D contours
   - plot scan of FCN vs each the parameters (DeltaChi2 plot)
   - do MC study: scatter plot, marginalized parameters, and -lnL distribution
'''

# EMCEE:
# http://dan.iel.fm/emcee/current/user/line/

# Simple fitting of data produced following a known pdf:
# http://glowingpython.blogspot.com/2012/07/distribution-fitting-with-scipy.html

# https://github.com/adrienrenaud/Profile_likelihood_ratio

# http://www.scipy-lectures.org/advanced/mathematical_optimization/

# http://www-ekp.physik.uni-karlsruhe.de/~quast/kafe/htmldoc/examples.html

from scipy import stats, optimize
import numpy as np
from matplotlib import pyplot as plt

xmin = -0.95
xmax = +0.95
alpha = 0.5
beta = 0.5
Npoints = 1000


##______________________________________________________________________________
def my_function(x, par):
    ''' This is the function we want to minimimze f(x;alpha,beta).
        It can have as many parameters as you want
        Note the normalization'''
    value = (1.0 + par[0] * x + par[1] * x * x) / (
    (xmax - xmin) + par[0] * (xmax ** 2 - xmin ** 2) / 2.0 + par[1] * (xmax ** 3 - xmin ** 3) / 3.0)
    return value


##______________________________________________________________________________
def generate_data(Npoints):
    ''' Acc. Rejection method to generate random data following our function f'''
    xdata = np.zeros(Npoints)
    naccept = 0
    while (naccept < Npoints):
        x = 2.0 * (np.random.uniform(0., 1., size=1)) - 1.0  # x goes from -1 to +1
        y = my_function(1.00, [alpha, beta]) * np.random.uniform(0., 1.,
                                                                 size=1)  # y goes from 0 to the value of f(1) (for alpha,beta >0)
        if y <= my_function(x, [alpha, beta]):
            if ((x < xmax) and (x > xmin)):  # accept the point if between xmax and xmin
                xdata[naccept] = x
                naccept = naccept + 1

    return xdata


##______________________________________________________________________________
def NegLogLhood(parameters, xdata):
    ''' LogLhood function. This is the function that will be called by minimize.
        It is simply the negative log of my_function(), which should be defined beforehand.
        my_function(x,par) returns the value of the function for a point x.
        The first input here should be the parameters that will be varied inside minimize.
        The second argument xdata is the data points.
        In minimize, the call to NegLogLhood is done without ANY argument. It is assummed
        that the first argument are the parameters, and the other arguments are passed in args=()
        inside minimize. In our case, since we only have on extra argument,
        the xdata has to be passed as args=(xdata,)
    '''
    logl = 0.
    # for x in xdata:
    #    logl = logl + np.log(my_function(x,parameters))
    logl = np.sum(np.log(my_function(xdata, parameters)))  # This is the same as the for loop above.

    return -logl


##______________________________________________________________________________
def scan_Lhood_parameters_1D(xdata, minLhood, param_values, param_ranges):
    ''' Inputs:
        xdata: is an array that contains the data points
        minLhood: the minimum value of the likelihood, as returned by the fit
        param_values: the best estimates of each parameter
        param_ranges: for each parameter, the minimum, maximum and step size to scan over

        my_function(x,params): is the function we are trying to minimize, x is the ran.var.
        Will take each parameter and vary it, keeping the other fixed at their best estimate.
        Will return two arrays and two nunbers for each parameter: the scanned parameter values, and the negative-log-likehood values for each value of the parameter. And the upper and lower errors, based on -lnLmin+0.5.
    '''

    ploerr, phierr, pscan, pnll = [], [], [], []  # define outputs as normal python lists
    # pscan and pnll will contain numpy.arrays as elements
    # First check that param_values are contained within the given ranges:
    for i in range(len(param_values)):
        if param_values[i] < param_ranges[i][0] or param_values[i] > param_ranges[i][1]:
            print(
            'Error!!! The central value: %6.3f is outside of the passed range: ' % param_values[i], param_ranges[i])
            return

    height = minLhood + 0.50
    for pindex in range(len(param_values)):
        pmin = param_ranges[pindex][0];
        pmax = param_ranges[pindex][1];
        pstep = param_ranges[pindex][2]
        scan = np.arange(pmin, pmax, pstep)
        nll = np.zeros(len(scan))
        i = 0
        for s in scan:
            pv = np.copy(param_values)
            np.put(pv, pindex, s)  # put the scanned value s in pindex, keep the rest fixed.
            nll[i] = -np.sum(np.log(my_function(xdata, pv)))
            i = i + 1
        # add scan and nll to pscan and pnll output (once for each parameter):
        pscan.append(scan)
        pnll.append(nll)
        # Now need to find what values of alpha correspond to height
        nll_left = nll[scan < param_values[pindex]]  # break nll for decreasing values down to the minimum (left)
        nll_right = nll[scan > param_values[pindex]]  # break nll for increasing values from the minimum (right)
        idxl = (np.abs(nll_left - height)).argmin()  # find closest value to height, return index
        plo = scan[idxl]  # this is the left tip of the errorbar
        idxr = len(nll_left) + (np.abs(nll_right - height)).argmin()  # find closest value to height, return index
        phi = scan[idxr]  # this is the right tip of the errorbar
        # add to ploerr and phierr
        ploerr.append(param_values[pindex] - plo)  # Negative Std. Err. on the best esimate of the parameter
        phierr.append(phi - param_values[pindex])  # Positive ...

    return pscan, pnll, ploerr, phierr


##______________________________________________________________________________
def plot_contours(xdata, minLhood, param_values, index_a=0, index_b=1):
    '''
    Plot the 1 and 2 sigma contours around the minimum neg log lhood in the param1 vs param2 plane
    The function may have more than 2 parameters, but only index_a and index_b will be changed simultaneously here.
    All the others will be kept at their estimated value.
    Some help for contour plotting:
    http://eli.thegreenplace.net/2014/meshgrids-and-disambiguating-rows-and-columns-from-cartesian-coordinates/
    http://matplotlib.org/examples/pylab_examples/contour_demo.html

    Will plot contour lines at the 68.3% and 95.5% CL (1 and 2 sigma). In the case of two free parameters, that means:
    -lnLmin+0.5*2.30 and -lnLmin+0.5*6.18. (The obvious -lnLmin+0.5, and -lnLmin+2 only apply for one free parameter [dof=1])
    '''
    import scipy
    amin = 0.0;
    amax = 1.0;
    bmin = 0.0;
    bmax = 1.2
    alphas = np.arange(amin, amax, 0.01)  #
    betas = np.arange(bmin, bmax, 0.01)

    OneSigma = scipy.stats.chi2.cdf(1 ** 2, 1)  # 0.6827
    TwoSigma = scipy.stats.chi2.cdf(2 ** 2, 1)  # 0.9545
    dchi2_1s_2par = scipy.stats.chi2.ppf(OneSigma, 2)  # 2.30
    dchi2_2s_2par = scipy.stats.chi2.ppf(TwoSigma, 2)  # 6.18

    # A, B = np.meshgrid(alphas, betas)
    NLL = np.zeros((len(alphas), len(betas)))
    # NLL = -np.sum(np.log(my_function(xdata,[A,B])))

    for i in range(len(alphas)):
        for j in range(len(betas)):
            pv = np.copy(param_values)
            np.put(pv, index_a, alphas[i])  # put this value of alpha in the list of parameters
            np.put(pv, index_b, betas[j])  # put this value of beta in the list of parameters
            NLL[i, j] = -np.sum(np.log(my_function(xdata, pv)))

    plt.figure()
    # CS=plt.contour(alphas, betas, NLL.T, levels=[minLhood+0.5,minLhood+2.],lwidths=2)
    mylevels = [minLhood, minLhood + 0.5 * dchi2_1s_2par, minLhood + 0.5 * dchi2_2s_2par]
    CS = plt.contourf(alphas, betas, NLL.T, levels=mylevels, colors=('g', 'y', 'b'), alpha=0.4)
    # For some reason the filled areas start after the second level, so I need to specify minLhood as the first (!?)
    e = plt.errorbar(param_values[0], param_values[1], fmt='o', color='k', capthick=3, lw=2)
    plt.ylim(bmin, bmax)
    plt.xlim(amin, amax)
    plt.title('ML 1 and 2 sigma contours')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    plt.legend([e], ['ML fit result'], loc='upper left', frameon=False, numpoints=1)
    plt.savefig('ML_contours.png')


##______________________________________________________________________________
def do_MC_study(Nexp=10, NLLdata=-999):
    ''' We repeat our experiment (generating the data, minimizing the Lhood, obtaining the parameters)
        a large number of times. This allows to see what would happen if we were to repeat our measurement.
        How likely is it that we would get values as similar as what we got.
    '''
    import time
    start = time.time()

    pnames = [r'$\alpha$', r'$\beta$']
    alphas = []
    betas = []
    nll = []
    print 'Generating MC experiments...'
    for exp in range(Nexp):
        xdata = generate_data(Npoints)
        res = optimize.minimize(NegLogLhood, np.array([1, 1]), args=(xdata,), method='Nelder-Mead')
        res = optimize.minimize(NegLogLhood, res.x, args=(xdata,), method='BFGS')
        p = res.x  # The best estimates of the parameters
        nllmin = res.fun  # The minimum ofthe NegLogLhood
        alphas.append(p[0])
        betas.append(p[1])
        nll.append(nllmin)

    # Now draw the alpha vs beta plane, and the marginal alpha and beta
    plt.figure()
    fig, ax = plt.subplots(nrows=2, ncols=2)  # split canvas into two columns and two rows
    ax[0, 0].plot(alphas, betas, linestyle='None', color='black', marker='o', markersize=1.2)
    ax[0, 0].set_title('MC study')
    ax[0, 0].set_xlabel(r'$\hat{\alpha}$')
    ax[0, 0].set_ylabel(r'$\hat{\beta}$')
    ax[0, 0].set_xlim(0., 1.);
    ax[0, 0].set_ylim(0., 1.)
    ax[0, 1].hist(betas, bins=20)
    ax[0, 1].set_xlabel(r'$\hat{\beta}$')
    ax[0, 1].set_xlim(0., 1.)
    ax[1, 0].hist(alphas, bins=20)
    ax[1, 0].set_xlabel(r'$\hat{\alpha}$')
    ax[1, 0].set_xlim(0., 1.)
    ax[1, 1].axis('off')
    fig.tight_layout()
    plt.savefig('ML_MCstudy.png')

    print('Variance V = ', np.cov(alphas, betas))
    print('Correlation r = ', np.corrcoef(alphas, betas))
    print('sample mean alpha = %.4f' % np.mean(alphas))
    print('sample mean beta = %.4f' % np.mean(betas))
    print('sample std. dev. alpha = %.4f' % np.std(alphas, ddof=1))
    print('sample std. dev. beta = %.4f' % np.std(betas, ddof=1))
    print('Time for MC study %s sec' % format(time.time() - start))

    # Draw negloglhood distribution
    plt.figure()
    plt.hist(nll, bins=20, facecolor='b', histtype='step')
    plt.title('-lnLmin distribution from %i MC toy experiments' % Nexp)
    plt.xlabel('-lnL')
    plt.ylabel('MC toys')
    plt.arrow(NLLdata, 10., 0., -10., color='r', width=0.1, length_includes_head=True)  # x,y,dx,dy
    plt.text(NLLdata, 13, 'Observed', fontsize=12, color='r', ha='center')
    plt.savefig('ML_MCstudy_NLL_histogram.png')


def main():
    np.random.seed(1234)
    xdata = generate_data(Npoints)
    # Plot distribution
    plt.hist(xdata, bins=40, facecolor='g', histtype='stepfilled', normed=True, label='MC data')
    plt.ylim(0.)
    plt.title(r'f(x)=(1+$\alpha$x+$\beta$x$^2$)/(2+2$\beta$/3)')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # Perform Maximum Likelihood fit:
    # lhood_model = optimize.minimize(NegLogLhood, np.array([1,1]), method='L-BFGS-B') #L-BFGS-B
    lhood_model = optimize.minimize(NegLogLhood, np.array([1, 1]), args=(xdata,), method='Nelder-Mead')
    lhood_model = optimize.minimize(NegLogLhood, lhood_model.x, args=(xdata,), method='BFGS')
    print lhood_model
    pnames = [r'$\alpha$', r'$\beta$']
    p = lhood_model.x
    perr = np.sqrt(np.diag(lhood_model.hess_inv))  # the standard error is the sqrt of the inverse of the hessian
    for n, a, e in zip(pnames, p, perr):
        print('%s = %.4f +- %.4f' % (n, a, e))

    rho = lhood_model.hess_inv[0][1] / (perr[0] * perr[1])
    print('Corr. coeff rho = %4.3f \n' % rho)

    # Plot distribution with result of fit
    xfine = np.arange(xmin, xmax, 1.e-3)
    plt.plot(xfine, my_function(xfine, p), 'r-',
             label=r'ML fit result %s=%.3f %s=%.3f' % (pnames[0], p[0], pnames[0], p[1]), linewidth=2)
    # I need to reverse the order of the legend:
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[::-1], labels[::-1], loc='upper left', frameon=False)
    plt.savefig('ML_histo.png')

    plt.figure()
    # Now scan the two parameters
    # Fix one and plot the -LogLhood as a function of the other.
    p = lhood_model['x']  # p1 and p2
    minLhood = lhood_model['fun']  # the minimum value
    param_ranges = [[0.40, 0.65, 0.005], [0.55, 0.90, 0.005]]  #
    vscan, vnll, vloerr, vhierr = scan_Lhood_parameters_1D(xdata, minLhood, p, param_ranges)
    for n, a, eh, el in zip(pnames, p, vhierr, vloerr):
        print('Graphical method: %s = %.4f +%.4f -%.4f' % (n, a, eh, el))
    print('Be careful! These strongly depends on the param_ranges, specially on the step')

    height = minLhood + 0.50
    fig, ax = plt.subplots(nrows=np.remainder(len(p), 2) + 1,
                           ncols=2)  # split canvas into two columns (and as many rows as needed)
    for i in range(len(p)):  # Loop over parameters
        scan = vscan[i]
        nll = vnll[i]
        # Draw NegLogLhood as a function of parameter:
        ax[i].plot(scan, nll, 'k-', linewidth=2)
        # Draw dashed line for value of -lnLmin+0.5:
        ax[i].plot([scan[0], scan[-1]], [height, height], 'b--', lw=2)
        ax[i].text(p[i], height + 0.01, r'-lnL$_{\rm min}+0.5$', fontsize=12, color='b', ha='center')
        # Draw error bar around best value:
        ax[i].errorbar(p[i], minLhood - 0.1, xerr=[[vloerr[i]], [vhierr[i]]], fmt='o', color='g', capthick=2, lw=2)
        ax[i].set_title(r'ML profile of %s' % pnames[i])
        ax[i].set_xlabel(r'%s' % pnames[i])
        ax[i].set_ylabel(r'-$\ln$L')
    fig.tight_layout()
    plt.savefig('ML_vs_params.png')

    plot_contours(xdata, minLhood, p, 0, 1)


main()
# do_MC_study(500,606.682)
