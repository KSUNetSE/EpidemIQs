
P10=0.008056640625
z=3
second=15
z_prime=(z - 10*P10)/(1-P10)
second_prime=(second - 100*P10)/(1-P10)
q_prime=(second_prime - z_prime)/z_prime
print(z_prime, second_prime, q_prime)
