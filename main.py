import numpy as np
import torch 

data = np.array([[1, 2], [3, 4]])
tensor_data = torch.from_numpy(data)
#or
data2 = [[5, 6], [7, 8]]
tensor_data2 = torch.tensor(data2)   
#make tensor from another tensor

x_ones = torch.ones_like(tensor_data)   # retains the properties of tensor_data
x_rand = torch.rand_like(tensor_data2, dtype=torch.float32)  # retains the properties of tensor_data2


print(f'Data 1:\n{tensor_data}\n')
print(f'Data 2:\n{tensor_data2}\n')
print(f'Ones Tensor:\n{x_ones}\n')
print(f'Random Tensor:\n{x_rand}\n')

shape=(3,4)
x_rand2 = torch.rand(shape)
x_ones2 = torch.ones(shape)

print(f'random Tensor with shape {shape}:\n{x_rand2}\n')
print(f'ones Tensor with shape {shape}:\n{x_ones2}\n')
print(f'shape of our tensor is: {x_rand2.shape}\n, and the datatype is: {x_rand2.dtype}\n')

#indexing and slicing
print(f'First row; \n{tensor_data[0]}\n')
print(f'First column; \n{tensor_data[:,0]}\n')
print(f'Last column; \n{tensor_data[..., -1]}\n')  

#joining tensor 
joined_tensor = torch.cat([tensor_data, tensor_data2], dim=0)  # concatenate along rows
print(f'Joined Tensor along rows:\n{joined_tensor}\n')

#matrix multiplication
mul1 = tensor_data @ tensor_data2.T  # matrix multiplication
mul2 = torch.matmul(tensor_data, tensor_data2.T)  # another way 

#element-wise multiplication
elem_mul = tensor_data * tensor_data2  # element-wise multiplication
elem_mul2 = torch.mul(tensor_data, tensor_data2)  # another way

#almost all operations in torch are same as numpy
tensor = torch.ones(5)
numpy_tensor = tensor.numpy()  # convert to numpy array
print(f'Converted tensor to numpy array:\n{numpy_tensor}\n')
print(f'tensor itself is:\n{tensor}\n')


