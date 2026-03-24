from nets.backend import set_device, get_backend


# CPU
set_device("cpu")
backend = get_backend()

a = backend.array([[1,2],[3,4]])
b = backend.array([[5,6],[7,8]])

print("CPU result:")
print(backend.matmul(a,b))


# GPU
set_device("cuda")
backend = get_backend()

a = backend.array([[1,2],[3,4]])
b = backend.array([[5,6],[7,8]])

print("GPU result:")
print(backend.matmul(a,b))