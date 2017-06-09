import scipy.stats as stats
import astropy.stats
import statsmodels.stats.proportion

print stats.binom.interval(.3415,2,.2,loc=.2)

print astropy.stats.binom_conf_interval(2,10,conf=.683,interval='wilson')

print astropy.stats.binom_conf_interval(2,10,conf=.683,interval='flat')

print astropy.stats.binom_conf_interval(2,10,conf=.683,interval='jeffreys')

print astropy.stats.binom_conf_interval(2,10,conf=.683,interval='wald')

print statsmodels.stats.proportion.proportion_confint(2,10,.683)

print statsmodels.stats.proportion.proportion_confint(2,10,.683,method='wilson')