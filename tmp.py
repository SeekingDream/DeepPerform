import os
import torch

overehads = []
for file_name in os.listdir('AbuseGanPerturbation'):
    file_path = os.path.join('AbuseGanPerturbation', file_name + '/overhead.tar')
    overehad = torch.load(file_path)
    overehads.extend(overehad)

print('-----------------------------')

