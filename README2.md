# A *Faster* Pytorch Implementation of Faster R-CNN

## Write at the beginning
```shell script
git clone 
cd faster-r-cnn
mkdir data
conda create -n faster-r-cnn python=3.6
conda activate faster-r-cnn
conda search pytorch
conda install pytorch
#下载requirements.txt中的文件
python -m pip install opencv-python
pip install msgpack

conda install cython cffi scipy easydict matplotlib pyyaml tensorboardX

```

报错1
```shell script
(faster-r-cnn) ➜  lib git:(master) ✗ sh make.sh
running build_ext
skipping 'model/utils/bbox.c' Cython extension (up-to-date)
skipping 'pycocotools/_mask.c' Cython extension (up-to-date)
Compiling nms kernels by nvcc...
make.sh: line 25: nvcc: command not found
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
Compiling roi pooling kernels by nvcc...
make.sh: line 35: nvcc: command not found
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
Compiling roi align kernels by nvcc...
make.sh: line 44: nvcc: command not found
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
Compiling roi crop kernels by nvcc...
make.sh: line 53: nvcc: command not found
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
```
解决，删除make.sh中cuda相关的编译脚本

报错2
```shell script
(faster-r-cnn) ➜  lib git:(master) ✗ sh make.sh
running build_ext
skipping 'model/utils/bbox.c' Cython extension (up-to-date)
skipping 'pycocotools/_mask.c' Cython extension (up-to-date)
Compiling nms kernels by nvcc...
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
Compiling roi pooling kernels by nvcc...
make.sh: line 35: nvcc: command not found
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
Compiling roi align kernels by nvcc...
make.sh: line 44: nvcc: command not found
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
Compiling roi crop kernels by nvcc...
make.sh: line 53: nvcc: command not found
Traceback (most recent call last):
  File "build.py", line 4, in <module>
    from torch.utils.ffi import create_extension
  File "/Users/louis/anaconda3/envs/faster-r-cnn/lib/python3.6/site-packages/torch/utils/ffi/__init__.py", line 1, in <module>
    raise ImportError("torch.utils.ffi is deprecated. Please use cpp extensions instead.")
ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.
```

解决办法，uninstall 调conda中的pytorch
`https://github.com/jwyang/faster-rcnn.pytorch/issues/706`
`https://github.com/jwyang/faster-rcnn.pytorch/issues/772`
```shell script
conda uninstall pytorch
pip install torch==0.4.0
pip install torchvision==0.2.0

conda search python
conda install python=2.7.18
conda deactivate
conda activate faster-r-cnn
python -V

pip install -r requirements.txt

conda remove -n faster-r-cnn
conda create -n faster-r-cnn python=2.7
```
`  Getting requirements to build wheel ... error`

` pip install opencv-python==4.2.0.32`

' error: no such file or directory: '/Users/louis/Documents/git/faster-rcnn.pytorch/lib/model/nms/src/nms_cuda_kernel.cu.o''

 安装不了cuda，就没有nvcc，所以换到有nvidia的显卡的机器上再弄
 
 先安装cuda-tool-kit
 安装完成之后需要写入环境变量
 执行nvcc -V 查看是否已经成功
 如果执行sh make.sh的时候显示不支持compute_30，按如下操作
 cd到/usr/local/cuda-11.1/samples/1_Utilities/deviceQuery
 执行那个Makefile文件
 成功后再执行 ./deviceQuery , 就可以查看nvcc可执行的计算数
 
 启动步骤
 https://blog.csdn.net/jgj123321/article/details/104942717
 
 ``error: invalid command 'develop'``的解决方案
 https://github.com/django-extensions/django-extensions/issues/92#issuecomment-946641

ImportError: /content/faster-rcnn.pytorch/lib/model/roi_crop/_ext/roi_crop/_roi_crop.so: undefined symbol: _cudaRegisterFatBinaryEnd
https://github.com/jwyang/faster-rcnn.pytorch/issues/465


使用apt-get install nvcc后重启了系统，可能会导致系统启动失败
这个时候进入ubuntu的recovery模式，连上网，使用root进入系统
```shell script
apt --fix-bromen remove
apt-get remove nvidia*
apt-get update; apt-get upgrade -f
```
看代码的疑问：
1. roibatchLoader中的bt_boxes
2. restnet101是如何被使用的
3. 是如何调用C++代码的