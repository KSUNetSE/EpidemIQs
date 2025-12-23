
import matplotlib.pyplot as plt
import os

# Plot S, I, R over time
def plot_sir_curves(time, S, I, R):
    plt.figure(figsize=(10,6))
    plt.plot(time, S, label='Susceptible')
    plt.plot(time, I, label='Infected')
    plt.plot(time, R, label='Recovered')
    plt.xlabel('Time (unit)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Epidemic Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_dir = os.path.join(os.getcwd(), 'output')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, 'sir_epidemic_curves.png')
    plt.savefig(path)
    plt.close()
    return path

plot_path = plot_sir_curves(time, susceptible, infected, recovered)
plot_path