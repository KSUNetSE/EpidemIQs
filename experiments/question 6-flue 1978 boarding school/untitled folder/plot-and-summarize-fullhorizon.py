
# Full horizon days covered by empirical data
full_days = empirical_days

# Interpolate simulation B and C at full empirical days (already done in previous step as sim_B_interp and sim_C_interp)
# Calculate mean empirical and simulated counts to summarize
mean_emp_B = np.mean(empirical_B)
mean_emp_C = np.mean(empirical_C)
mean_sim_B = np.mean(sim_B_interp)
mean_sim_C = np.mean(sim_C_interp)

# Calculate RMSE, MAE, R2 again on full empirical days (redundant but ensured)
rmse_B_full = np.sqrt(mean_squared_error(empirical_B, sim_B_interp))
mae_B_full = mean_absolute_error(empirical_B, sim_B_interp)
r2_B_full = r2_score(empirical_B, sim_B_interp)

rmse_C_full = np.sqrt(mean_squared_error(empirical_C, sim_C_interp))
mae_C_full = mean_absolute_error(empirical_C, sim_C_interp)
r2_C_full = r2_score(empirical_C, sim_C_interp)

# Plot overlay for full horizon
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# B plot
axs[0].plot(empirical_days, empirical_B, label='Empirical B', marker='o', linestyle='-', color='blue')
axs[0].plot(empirical_days, sim_B_interp, label='Simulated B', linestyle='--', color='cyan')
axs[0].set_ylabel('Number of Bed-confined Individuals')
axs[0].set_title('Full Horizon Bed-confined (B) : Simulated vs Empirical')

# Highlight divergence or secondary peaks
axs[0].annotate('Peak Empirical', xy=(emp_peak_B_time, emp_peak_B_value), xytext=(emp_peak_B_time+1, emp_peak_B_value+20),
                arrowprops=dict(facecolor='blue', shrink=0.05), color='blue')
axs[0].annotate('Peak Simulated', xy=(peak_B_time, peak_B_value), xytext=(peak_B_time+3, peak_B_value-50),
                arrowprops=dict(facecolor='cyan', shrink=0.05), color='cyan')

# C plot
axs[1].plot(empirical_days, empirical_C, label='Empirical C', marker='o', linestyle='-', color='red')
axs[1].plot(empirical_days, sim_C_interp, label='Simulated C', linestyle='--', color='orange')
axs[1].set_ylabel('Number of Convalescent Individuals')
axs[1].set_title('Full Horizon Convalescent (C) : Simulated vs Empirical')
axs[1].set_xlabel('Days from Outbreak Start')

# Additional annotations for divergence
axs[1].annotate('Model underfits convalescent peak', xy=(7, 163), xytext=(9, 200), arrowprops=dict(facecolor='orange', shrink=0.05), color='orange')

for ax in axs:
    ax.legend()
    ax.grid(True)

# Add summary stats text
summary_text = (
    f'Mean Empirical B: {mean_emp_B:.1f}, Simulated B: {mean_sim_B:.1f}\n'
    f'RMSE B: {rmse_B_full:.1f}, MAE B: {mae_B_full:.1f}, R2 B: {r2_B_full:.2f}\n'
    f'Mean Empirical C: {mean_emp_C:.1f}, Simulated C: {mean_sim_C:.1f}\n'
    f'RMSE C: {rmse_C_full:.1f}, MAE C: {mae_C_full:.1f}, R2 C: {r2_C_full:.2f}'
)
fig.text(0.5, 0.01, summary_text, ha='center', fontsize=12, color='black')

# Save figure
plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig('output/sim_vs_empirical_overlay_fullhorizon.png')
plt.close()

rmse_B_full, mae_B_full, r2_B_full, rmse_C_full, mae_C_full, r2_C_full, mean_emp_B, mean_sim_B, mean_emp_C, mean_sim_C