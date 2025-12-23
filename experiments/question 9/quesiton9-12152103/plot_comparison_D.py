
import matplotlib.pyplot as plt
import os

# Plot D(t) curves for both scenarios
plt.figure(figsize=(10, 6))
plt.plot(df_core['time'], df_core['D'], label='Core-seeded crisis', color='red')
plt.fill_between(df_core['time'], df_core['D_90ci_lower'], df_core['D_90ci_upper'], color='red', alpha=0.2)
plt.plot(df_periphery['time'], df_periphery['D'], label='Periphery-seeded crisis', color='blue')
plt.fill_between(df_periphery['time'], df_periphery['D_90ci_lower'], df_periphery['D_90ci_upper'], color='blue', alpha=0.2)

plt.xlabel('Time (arbitrary units)')
plt.ylabel('Number of Distressed Banks (D)')
plt.title('Distressed Banks Over Time for Core vs Periphery Seeding')
plt.legend()
plt.grid(True)

# Save the plot
output_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plot_path = os.path.join(output_dir, 'D_comparison_core_periphery.png')
plt.savefig(plot_path)
plt.close()

plot_path