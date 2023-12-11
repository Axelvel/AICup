import joblib
import torch

device = "cpu"
output = joblib.load('output.plk')
output_tensor = output[0].to('cpu')
print(output)