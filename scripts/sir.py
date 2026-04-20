#lets generate the simulation of SIR model (susceptible, infected , and recovered)
#parameters: beta (contagious rate) , gamma (recovery time)


#lets initialize gamma and beta
gamma = 0.1
beta = 0.25

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
days = 200
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
#lets say 20% of the population gets vaccinated each year, which is 0.2/365 per day
vaccination_rate = 0.2 / 365

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