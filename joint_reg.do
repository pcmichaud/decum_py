clear all
capture log close 
set more off

cd ~/decum_py/

import delimited  using output/joint_opt.csv, clear

rename price_ann_fair risk_ann 
rename price_ltci_fair risk_ltci
rename price_rmr_fair risk_rmr

xtile qtotinc = totinc, nq(4)
xtile qretinc = retinc, nq(4)
xtile qwealth_total = wealth_total, nq(4)
gen home_equity = home_value - mort_balance
xtile qhome_equity = home_equity, nq(4)

replace risk_rmr = risk_rmr*100

global demo "age female college university  married" 
global eco "ib1.qwealth_total ib1.qhome_equity ib1.qtotinc ib1.qretinc"
global health "xi mu zeta risk_*"
global pref "pref_beq_money pref_home pref_live_fast pref_risk_averse"

qui: reg buy_ann_indp $demo $eco $health $pref
estimates store indp_ann
qui: reg buy_ltci_indp $demo $eco $health $pref
estimates store indp_ltci
qui: reg buy_rmr_indp $demo $eco $health $pref
estimates store indp_rmr

estimates table indp_*, star


qui: reg buy_ann_joint $demo $eco $health $pref
estimates store joint_ann
qui: reg buy_ltci_joint $demo $eco $health $pref
estimates store joint_ltci
qui: reg buy_rmr_joint $demo $eco $health $pref
estimates store joint_rmr

estimates table joint_*, star

gen  joint = 0 if buy_ann_joint==0 & buy_ltci_joint==0 & buy_rmr_joint==0
replace joint = 1 if buy_ann_joint>0 & buy_ltci_joint==0 & buy_rmr_joint==0
replace joint = 2 if buy_ann_joint==0 & buy_ltci_joint>0 & buy_rmr_joint==0
replace joint = 3 if buy_ann_joint==0 & buy_ltci_joint==0 & buy_rmr_joint>0
replace joint = 4 if buy_ann_joint>0 & buy_ltci_joint>0 & buy_rmr_joint==0
replace joint = 5 if buy_ann_joint>0 & buy_ltci_joint==0 & buy_rmr_joint>0
replace joint = 6 if buy_ann_joint==0 & buy_ltci_joint>0 & buy_rmr_joint>0
replace joint = 7 if buy_ann_joint>0 & buy_ltci_joint>0 & buy_rmr_joint>0

label def joint 0 "none" 1 "ann" 2 "ltci" 3 "rmr" 4 "ann-ltci" 5 "ann-rmr" 6 "ltci-rmr" 7 "all"
label values joint joint 

mlogit joint $demo $eco $health $pref, base(0)
est store m
forval i = 0/7 {
est res m
margins, dydx(*) predict(outcome(`i')) post
est store m`i'
}
estimates table  m0 m1 m2 m3 m4 m5 m6 m7, star

esttab m0 m1 m2 m3 m4 m5 m6 m7 using "output/joint_reg.tex", se replace tex













