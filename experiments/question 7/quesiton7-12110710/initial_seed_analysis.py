
# Extract initial seed nodes
seed_nodes = df['seed_nodes'].apply(eval)  # convert string lists to actual lists

# Count how many initial seeds in core and periphery
core_count = []
periphery_count = []

for seeds in seed_nodes:
    core_seeds = sum(1 for s in seeds if s < 20)
    periphery_seeds = sum(1 for s in seeds if s >= 20)
    core_count.append(core_seeds)
    periphery_count.append(periphery_seeds)

mean_core_init_fail = np.mean(core_count)
median_core_init_fail = np.median(core_count)
mean_periphery_init_fail = np.mean(periphery_count)
median_periphery_init_fail = np.median(periphery_count)

mean_core_init_fail, median_core_init_fail, mean_periphery_init_fail, median_periphery_init_fail