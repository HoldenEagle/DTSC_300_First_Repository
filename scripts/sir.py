#lets generate the simulation of SIR model (susceptible, infected , and recovered)
#parameters: beta (contagious rate) , gamma (recovery time)


#lets initialize gamma and beta
gamma = 0.05
beta = 0.3

#implement these functions St+1−St=−βItStN,It+1−It=βItStN−γIt,Rt+1−Rt=γIt
import numpy as np
import matplotlib.pyplot as plt

def sir_model(S0, I0, R0, beta, gamma, days):
    S = [S0]
    I = [I0]
    R = [R0]

    for day in range(days):
        new_infections = beta * I[-1] * S[-1] / (S[-1] + I[-1] + R[-1])
        new_recoveries = gamma * I[-1]

        S.append(S[-1] - new_infections)
        I.append(I[-1] + new_infections - new_recoveries)
        R.append(R[-1] + new_recoveries)

    return S, I, R

#lets run this model with gamma, beta, and initial conditions
days = 1000
S0 = 9990
I0 = 10
R0 = 0

S, I, R = sir_model(S0, I0, R0, beta, gamma, days)
#lets plot the results
plt.figure(figsize=(10,6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()


#adding a vaccination term to the model
#lets say 40% of the population gets vaccinated each year, which is 0.4/365 per day
vaccination_rate = 0.4 / 365

#we can modify the SIR model to include vaccination. The new equations will be:
#St+1−St=−βItStN−vaccination_rate*St

#now lets making the function for the SIR model with vaccination
def sir_model_vaccination(S0, I0, R0, beta, gamma, vaccination_rate, days):
    S = [S0]
    I = [I0]
    R = [R0]

    for day in range(days):
        new_infections = beta * I[-1] * S[-1] / (S[-1] + I[-1] + R[-1])
        new_recoveries = gamma * I[-1]
        new_vaccinations = vaccination_rate * S[-1]

        S.append(S[-1] - new_infections - new_vaccinations)
        I.append(I[-1] + new_infections - new_recoveries)
        R.append(R[-1] + new_recoveries + new_vaccinations)

    return S, I, R

#lets run this model with vaccination
S_vacc, I_vacc, R_vacc = sir_model_vaccination(S0, I0, R0, beta, gamma, vaccination_rate, days)
#lets plot the results  
plt.figure(figsize=(10,6))
plt.plot(S_vacc, label='Susceptible with Vaccination')
plt.plot(I_vacc, label='Infected with Vaccination')
plt.plot(R_vacc, label='Recovered with Vaccination')
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.title('SIR Model Simulation with Vaccination')
plt.legend()
plt.show()

#lets compare the two models by plotting them on the same graph
plt.figure(figsize=(10,6))
plt.plot(S, label='Susceptible without Vaccination')
plt.plot(I, label='Infected without Vaccination')
plt.plot(R, label='Recovered without Vaccination')
plt.plot(S_vacc, label='Susceptible with Vaccination')
plt.plot(I_vacc, label='Infected with Vaccination')
plt.plot(R_vacc, label='Recovered with Vaccination')
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.title('Comparison of SIR Model with and without Vaccination')
plt.legend()
plt.show()


#now lets simulate a new strand coming in after 100 days with a higher beta (more contagious) and see how it affects the model.
# We will assume that the new strand has a beta of 0.35 and starts with 10 infected individuals.
#we also will assume the strand is immune to the vaccine, so the vaccination rate will not affect
# the new strand. Lets simulate this by running initial SIR model for 100 days, then at day 100 we will change the beta and I0 and run the model for another 900 days.

new_beta = 0.35
new_I0 = 10
#run the initial model for 100 days
S_initial, I_initial, R_initial = sir_model_vaccination(S0, I0, R0, beta, gamma, vaccination_rate, 100)
#at day 100, we will change the beta and I0 and run the model for
#another 900 days. We will also need to update the S0 and R0 to reflect the new initial conditions at day 100.
S_new = 9990
R_new = 0
vaccination_rate_new = 0 #since the new strand is immune to the vaccine, we will set the vaccination rate to 0 for the new model
S_new, I_new, R_new = sir_model_vaccination(S_new, new_I0, R_new, new_beta, gamma, vaccination_rate, 300)
#lets plot the results
plt.figure(figsize=(10,6))
plt.plot(S_initial + S_new, label='Susceptible')
plt.plot(I_initial + I_new, label='Infected')
plt.plot(R_initial + R_new, label='Recovered')
plt.xlabel('Days')
plt.ylabel('Number of People')
plt.title('SIR Model Simulation with New Strand')
plt.legend()
plt.show()




