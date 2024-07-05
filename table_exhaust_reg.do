clear all
capture log close 
set more off

capture cd ~/decum_py/
capture cd ~/ire/Projets/decum/decum_py/
import delimited  using output/data_with_exhaust_sim.csv, clear


xtile qtotinc = totinc, nq(4)
xtile qretinc = retinc, nq(4)
xtile qwealth_total = wealth_total, nq(4)
gen home_equity = home_value - mort_balance
xtile qhome_equity = home_equity, nq(4)

sum pexhaust pexhaust85_sim, d

global demo "female married" 
global eco "ib1.qwealth_total  ib1.qretinc"

qui: reg pexhaust  $demo $eco 
estimates store data
qui: reg pexhaust85_sim $demo $eco
estimates store sim

estimates table data sim, star












