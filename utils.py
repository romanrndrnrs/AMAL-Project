import pylatexenc
import matplotlib.pyplot as plt
import numpy as np
import time
import torch

# SPSA recommended when dealing with noise
from qiskit.algorithms.optimizers import COBYLA, SPSA
import qiskit.algorithms.optimizers as optimq
from qiskit.algorithms.minimum_eigensolvers import VQE, NumPyMinimumEigensolver, MinimumEigensolverResult
from qiskit.primitives import Sampler
from qiskit import QuantumRegister,ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import EfficientSU2
from qiskit.circuit import Parameter
# settings.tensor_unwrapping = False
# settings.use_pauli_sum_op = False
from qiskit_machine_learning.kernels import FidelityQuantumKernel
import sklearn
from sklearn.svm import SVC
from math import sqrt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from math import ceil

#%%
def compute_kta_tensor(kernel,data_x,data_y,n,R,device):
  """
  Parameters
  ---------
  kernel: the fidelity quantum kernel from ansatz feature space used as a kernel
  data_x/y: the preprocessed data to compute the KTA on
  ansatz: the ansatz used as a kernel
  n: the subset of examples taken from data to compute the gram matrix
  R: the number of classes, default=2
  device: the device used for computation
  """
  data_x = torch.tensor(data_x.clone().detach(),device = device)
  data_y = torch.tensor(data_y.clone().detach(), device = device)
  idx_set = torch.linspace(start=0,end=n-1,steps=n,dtype=int)
  idx = np.random.choice(idx_set, n,replace=False)
  # print("sidx",idx)
  # gram, idx = compute_gram(ansatz,data_x,n)
  x = data_x[idx]
  # print(data_y)
  y = data_y[idx]
  gram = torch.tensor(kernel.evaluate(x.cpu()),device=device,dtype=torch.float64)
  if R < 2:
    print("Please provide a correct number of classes")
    return 0
  if R == 2:
    yy = torch.outer(y,y,device=device)
  else:
    yy = torch.zeros((n,n),dtype=torch.float64,device=device)
    for i in range(n):
      for j in range(n):
        if y[i] == y[j]:
          yy[i][j] = 1
        else:
          yy[i][j] = -1 / (R-1)
  kta = torch.trace(torch.matmul(gram.t(),yy)) / (sqrt( torch.trace(torch.matmul(gram.t(),gram))* torch.trace(torch.matmul(yy.t(),yy))))
  return kta, idx
#%%
def sample_ansatz(p:int, l0:int,N:int):
  """
  Parameters
  ---------
  p: reduced number of dimensions
  l0: number of layers
  N: number of qubits

  Returns
  ---------
  ansatz: the qiskit ansatz
  ansatz_attr: dict containing the different gates in img form
  ansatz_tensor: tensor containing the different gates, shape: rx_mat x rz_mat x ry_mat x cx_ctl_mat x cx_tgt_mat
  """
  # B: number of blocks consisting of single qubit rotation gate and 2 qubit gate
  B = l0*ceil(p/N)
  rx_mat = torch.zeros((N,B*2))
  rz_mat = torch.zeros((N,B*2))
  ry_mat = torch.zeros((N,B*2))
  cx_ctl_mat = torch.zeros((N,B*2))
  cx_tgt_mat = torch.zeros((N,B*2))

  paulis = np.array(["X","Y","Z"])
  two_qb_g = np.array(["I","CNOT"])
  nb_params = B*N
  params = [Parameter(f"θ[{i}]") for i in range(nb_params)]
  ansatz = QuantumCircuit(N)
  for i in range(B):
    pauli_choice = np.random.choice(paulis)
    idx_pauli = np.where(paulis == pauli_choice)[0][0]
    reduced_paulis = np.delete(paulis,idx_pauli)
    alt_pauli_choice = np.random.choice(reduced_paulis)

    for j in range(N):
      if ((j % 2) == 0):
        if pauli_choice == "X": 
          rx_mat[j][2*i] = 1
          ansatz.rx(params[(j + i*N)],j)
        elif pauli_choice == "Y": 
          ry_mat[j][2*i] = 1
          ansatz.ry(params[j + i*N],j)
        else: 
          rz_mat[j][2*i] = 1
          ansatz.rz(params[j + i*N],j)
      else:
        if alt_pauli_choice == "X": 
          rx_mat[j][2*i] = 1
          ansatz.rx(params[j + i*N],j)
        elif alt_pauli_choice == "Y": 
          ry_mat[j][2*i] = 1
          ansatz.ry(params[j + i*N],j)
        else: 
          rz_mat[j][2*i] = 1
          ansatz.rz(params[j + i*N],j)

    for j in range(N-1):
      two_qb_g_choice = np.random.choice(two_qb_g)
      if two_qb_g_choice == "CNOT":
        cx_ctl_mat[j][2*i+1] = 1
        cx_tgt_mat[j+1][2*i+1] = 1
        ansatz.cx(j,j+1)
  ansatz_tensor = torch.cat((rx_mat.unsqueeze(0),ry_mat.unsqueeze(0),rz_mat.unsqueeze(0),cx_ctl_mat.unsqueeze(0),cx_tgt_mat.unsqueeze(0)))
  return ansatz, {"rx":rx_mat,"ry":ry_mat,"rz":rz_mat,"cx_ctl":cx_ctl_mat,"cx_tgt":cx_tgt_mat}, ansatz_tensor
#%%
def format_data(nb, datax):
  """
  Parameters
  ---------
  nb: number of parameters in the ansatz
  datax: data to process

  Returns
  ---------
  datax_res: the processed data
  """
  if nb == datax.size(1):
    return datax
  elif nb < datax.size(1):
    return datax[:nb]
  else:
    for i in range(1,nb // datax.size(1)):
      if i ==1:
        datax_res = torch.cat((datax,datax),dim = 1)
      else:
        datax_res = torch.cat((datax_res,datax),dim = 1)
    for i in range(nb % datax.size(1)):
      val = torch.zeros((10,1))
      datax_res = torch.cat((datax_res,val),dim=1)
    return datax_res
#%%
def compute_example(inputs,labels,layers,R=5):
    """
  Parameters
  ---------
  inputs: the input examples of size (BATCH,DIM) = (300,40)
  labels: labels corresponding to the input for the classification problem
  layers: number of layers of the ansatz = depth

  Returns
  ---------
  ansatz_tensor: the (CHANNELS,QUBITS,CIRCUIT_DEPTH) = (5,8,10*layers) image for p = 50, QUBITS = 8
  kta: the kta of the random ansatz generated from a sample space
  """
  assert inputs.shape == (300,40)
  N = 8
  l0 = layers
  p = 40
  
  ansatz, ansatz_dict, ansatz_tensor = sample_ansatz(p,l0,N)
  nb_params = ansatz.num_parameters
  print(nb_params)
  datax = format_data(nb_params,inputs)
  print(datax.shape)
  datay = labels
  qker = FidelityQuantumKernel(feature_map=ansatz)
  kta, _ = compute_kta_tensor(qker,datax,datay,100,R,device)
  return ansatz_tensor, kta
#%%