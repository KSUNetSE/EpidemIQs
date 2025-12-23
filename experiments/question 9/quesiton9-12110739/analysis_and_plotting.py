
import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = ['/Users/hosseinsamaei/phd/epidemiqs/output/results-110.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-120.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-310.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-320.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-210.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-220.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-410.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-420.csv']

# Fix phi extraction: proper float mapping for last digit (0-9) to phi
# 0 means no threshold set or 0.0, assume allowed

metadata = {}
for path in file_paths:
    file = os.path.basename(path)
    base, _ = os.path.splitext(file)
    _, code = base.split('-')
    topology = 'ER' if code[0] in ['1', '2'] else 'BA'
    init_cond = 'random' if code[1] == '1' else 'targeted'
    phi = 0.1 * int(code[2])
    metadata[path] = {'topology': topology, 'initial_condition': init_cond, 'phi': phi}

results = []

# Directory for saving plots
output_dir = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for path in file_paths:
    df = pd.read_csv(path)
    total_banks = df['S'][0] + df['I'][0]  # Should always be 500
    I = df['I'].values
    fraction_defaulted = I / total_banks

    final_cascade_size = fraction_defaulted[-1]  # Fraction defaulted at last step

    # Cascade duration: last timestep where fraction defaulted changes
    diffs = abs(fraction_defaulted[1:] - fraction_defaulted[:-1])
    stable_steps = (diffs < 1e-5).nonzero()[0]  # indices where change negligible
    if len(stable_steps) > 0:
        cascade_duration = stable_steps[0] + 1  # +1 since diffs index shifted by 1
    else:
        cascade_duration = len(fraction_defaulted) - 1

    # Peak default rate: max increase in fraction defaulted in one step
    peak_default_rate = max(diffs)

    meta = metadata[path]

    # Save plot
    plt.figure()
    plt.plot(df['step'], fraction_defaulted, marker='o', linestyle='-')
    plt.xlabel('Time step')
    plt.ylabel('Fraction Defaulted')
    plt.title(f"Default Fraction over Time\nTopology: {meta['topology']} - Init: {meta['initial_condition']} - Phi: {meta['phi']}")
    plot_filename = os.path.join(output_dir, f"default_fraction_{os.path.basename(path).replace('.csv', '')}.png")
    plt.savefig(plot_filename)
    plt.close()

    results.append({
        'file': path,
        'topology': meta['topology'],
        'initial_condition': meta['initial_condition'],
        'phi': meta['phi'],
        'final_cascade_size': final_cascade_size,
        'cascade_duration': cascade_duration,
        'peak_default_rate': peak_default_rate,
        'plot_path': plot_filename
    })

