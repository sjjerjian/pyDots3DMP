


Data Structures


Raw data is constructed into a list/series of "Population" objects
Each populations contains the actual spike times for 1 or more "Unit" objects

From a list/series of Population objects, I apply the get_firing_rates method on each one to extract a 
unit x trials x timestamps array of firing rates. 

These can be stacked across sessions, or kept as individual. They can also be averaged across trials to become 
units x conditions x timestamps, or stay as single trials.
Each firing-rates array is part of a "RatePopulation" object, which also stores the associated stimulus/behavior information for each trial/condition.

To consider how best to structure scripts and modules, consider what analyses we want to do, and specifically what they require:
1. single trial activity or condition averages
2. within-session vs pseudo-population
3. require tuning paradigm, or just task?
4. normalized to a baseline period, or just standardized across time?

Analyses
1. Number of trials per condition for each unit (for exclusion criteria, task and/or tuning)
    - keep individual trials
2. Average firing rates across conditions for each unit (for exclusion criteria, task and/or tuning)
3. PSTH (+ raster?) plots for each unit for different condition groupings
4. Choice/Wager Probabilities
5. Tuning to heading in tuning task
6. Tuning to heading
7. Modulation across time (timing of peak modulation?)
8. Logistic Regression/Support Vector Machine to predict choice/wager
9. TDR (Regression) analysis to evaluate encoding of different task variables
10. PCA with 'RT' axis (see Chand, Remington papers?)
11. (all of the above, split by area?)



1. 