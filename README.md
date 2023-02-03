# Asset Decumulation and Risk Management in Retirement

Pierre-Carl Michaud and Pascal St-Amour


This repository contains the following directories: 

- backup: for internal use
- inputs: data and parameter inputs for the model to run
- logs: log files of model run 
- output: stores output files


The run sequence to reproduce results in the paper is the following: 

- compute_estimate.py: will estimate structural parameters by NLS and store them in output. 
- compute_gradients.py: will compute the gradients of NLS for standard error computations. 
- compute_reference.py: computes the expected value for each respondent and scenario at value of parameters computed above. 
- compute_deltas.py: computes the probability shifters (status quo) parameters.
- compute_exhaust.py: computes the simulated probability of spending all financial wealth by age 85
- compute_fair_prices.py: computes the fair price for all three products. 

The tables are produced with the following programs:

- table_takeup.py: produces table of take-up with data, simulated and model. Also table for deltas
- table_data_elasticity.py: produces table with price and quantity elasticity from data (experiment)
- table_estimate.py: produces table of estimates. 
- table_sim_elasticity.py: produces table which compares elasticities (sim, data and model)
- merge_exhaust.py: program to merge simulated data with main data
- table_exhaust_reg.do: (stata file) computes stats for probability of exhausting resources at 85 in data and simulation. 
- table_decompose.py: produces table which does counterfactuals on motives to purchase risk management. 
- table_bundling.py: produces table for statistics on demand for bundling. 
-

Finally, the following programs are internal to the model and used to compute: 

- actors.py: defines households, etc. 
- budget.py: defines budget constraint
- frame.py: program with functions that interface data with model
- joint_*.py: programs to run counterfactuals on demand for products, they can be run at once with batch_joint.sh
- optim.py: optimization programs for estimation 
- prefs.py: specification of preferences 
- solve.py: model solution functions 
- space.py: functions to create state-space
- survival.py: functions to manipulate health probabilities
- tools.py: other functions used in solving the model.

    Questions or comments can be sent to [Pierre-Carl Michaud](mailto:pierre-carl.michaud@hec.ca).




