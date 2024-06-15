# 像教女朋友一样教你写cuda算子

## Code structure
```shell
├── include
│   └── add2.h # header file of add2 cuda kernel
├── kernel
│   └── add2_kernel.cu # add2 cuda kernel
├── pytorch
│   ├── add2_ops.cpp # torch wrapper of add2 cuda kernel
│   ├── time.py # time comparison of cuda kernel and torch
│   ├── train.py # training using custom cuda kernel
│   ├── setup.py
│   └── CMakeLists.txt
└── README.md
```

## PyTorch
### Compile cpp and cuda
**JIT**  
Directly run the python code.

**Setuptools**  
```shell
python3 pytorch/setup.py install
```

**CMake**  
```shell
mkdir build
cd build
cmake ../pytorch
make
```

### Run python
**Compare kernel running time**  
```shell
python3 pytorch/time.py --compiler jit
python3 pytorch/time.py --compiler setup
python3 pytorch/time.py --compiler cmake
```

**Train model**  
```shell
python3 pytorch/train.py --compiler jit
python3 pytorch/train.py --compiler setup
python3 pytorch/train.py --compiler cmake
```

## F.A.Q
> **Q.** ImportError: libc10.so: cannot open shared object file: No such file or directory  
**A.** You must do `import torch` before `import add2`.

> **Q.** tensorflow.python.framework.errors_impl.NotFoundError: build/libadd2.so: undefined symbol: _ZTIN10tensorflow8OpKernelE  
**A.** Check if `${TF_LFLAGS}` in `CmakeLists.txt` is correct.
