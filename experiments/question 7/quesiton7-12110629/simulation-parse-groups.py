
# Now that we've inspected the format: '<index>,<group>\n'.
# Parse it to a numpy array: core=0, periphery=1.
import numpy as np
import os
def parse_labels(nodegroup_path):
    labels = []
    with open(nodegroup_path, 'r') as fp:
        for line in fp:
            idx, grp = line.strip().split(',')
            if grp.lower().startswith('core'):
                labels.append(0)
            else:
                labels.append(1)
    return np.array(labels)
# Let's quickly test and return group counts.
group_path = os.path.join(os.getcwd(), 'output', 'nodegroups-coreperiphery.txt')
node_labels = parse_labels(group_path)
core_count = (node_labels == 0).sum()
peri_count = (node_labels == 1).sum()
core_count, peri_count, node_labels[:10], node_labels[-10:]
