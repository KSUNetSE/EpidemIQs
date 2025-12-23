
import os

# Parsing context from filename
# Filename example: 'results-110.csv' -> interpreting
# First digit = topology, 1 or 2 or 3 or 4 (ER or BA, to be inferred)
# Second digit = initial condition type
# Third digit = phi

file_paths = ['/Users/hosseinsamaei/phd/epidemiqs/output/results-110.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-120.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-310.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-320.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-210.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-220.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-410.csv',
              '/Users/hosseinsamaei/phd/epidemiqs/output/results-420.csv']

metadata = {}

for path in file_paths:
    file = os.path.basename(path)
    base, _ = os.path.splitext(file)
    _, code = base.split('-')
    # topology mapping: assuming 1,2 are ER; 3,4 are BA as per instruction
    topology = 'ER' if code[0] in ['1', '2'] else 'BA'
    # initial condition type mapping: 1 = random, 2 = targeted
    init_cond = 'random' if code[1] == '1' else 'targeted'
    # phi mapping (from third digit): we consider it as 0.1*digit
    phi = 0.1 * int(code[2])
    metadata[path] = {'topology': topology, 'initial_condition': init_cond, 'phi': phi}

