
import matplotlib.pyplot as plt
import pandas as pd
import os

def quick_plot(csv_path, label, color):
    df = pd.read_csv(csv_path)
    plt.plot(df['time'], df['I'], label=label, color=color)

plt.figure(figsize=(8,5))
quick_plot(os.path.join(os.getcwd(),'output','results-1-1.csv'), 'Homogeneous (ER)', 'b')
quick_plot(os.path.join(os.getcwd(),'output','results-1-2.csv'), 'Heterogeneous (BA)', 'r')
# overlay deterministic
ref_df = pd.read_csv(os.path.join(os.getcwd(),'output','seir_ode_reference.png').replace('.png','.csv'), error_bad_lines=False) if os.path.exists(os.path.join(os.getcwd(),'output','seir_ode_reference.csv')) else None
if ref_df is not None:
    plt.plot(ref_df['time'], ref_df['I'], label='Deterministic ODE', color='k', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Number Infectious (I)')
plt.title('SEIR I(t): Homogeneous vs Heterogeneous Network')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.getcwd(), 'output', 'I_curve_comparison.png'))
plt.close()