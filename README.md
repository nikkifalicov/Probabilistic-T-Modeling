# Probabilistic Modeling of Green Line Delays
## Final Project for Statistical Pattern Recognition, Fall 2025

### About the project:

Travel times on the Green Line are notoriously inconsistent, to the extent that there is an unofficial
six mile course known as the Charlie Card Challenge, in which runners seek to out-race the T as it
travels from Boston College to Park Street. However, the speed at which runners must run can vary
from a 6:40 to 10:00 minute per mile pace. Inspired by this race, we wanted to see if probabilistic
modeling could accurately predict the travel time between the two stations.

We used univariate regression to train a model that given the date and time of departure, will
produce a single estimate for the travel time from Boston College to Park Street via the Green Line
during peak use hours (7AM - 7PM). We will be training on data recorded from January through
May and evaluating the quality of predictions on trips during June.

Probabilistic modeling is suitable for this task because the Green Line is subject to stochastic vari-
ables such as car traffic, weather conditions, differing user demand, and skipping low demand stations when delays are especially high. Using a probabilistic approach will better account for this
uncertainty and ideally produce more reliable predictions.

### Running the code:
1. Clone the repository
2. Create a virtual environment and install requirements from requirements.txt
3. To run the baseline model, run run_baseline.py
4. To run the upgrade model, run run_gmm_upgrade.ipynb

### Our full report is available at final_report.pdf