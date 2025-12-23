
# First, let's inspect the nodegroups file to figure out its format,
# since the error suggests there might be header or commas, not just plain labels.
import os
with open(os.path.join(os.getcwd(), 'output', 'nodegroups-coreperiphery.txt'), 'r') as fp:
    lines = fp.readlines()
# Return first 10 lines (for inspection), and the total number of lines.
lines[:10], len(lines)
