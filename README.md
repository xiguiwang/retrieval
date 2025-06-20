# Retrieval Image augument with LVM

This repo retrieve Image by Text or Image.
Images are embedded by CLIP model into vectors. The vecotrs are stored into Redis database.
You can find the images through text or an input image.

Retrieval Image by Text or Image with LVM improve the accuray

![Retrieval Image by Text or Image](./demo_result/output1_480.gif)
Search Image by Text

![Retrieval Image by Text or Image](./demo_result/output2_480.gif)
Search Image by Image

## Set up Environment

```
conda create -n image-retrieval python=3.11
conda activate image-retrieval
git clone https://github.com/xiguiwang/retrieval.git
cd retrieval
pip install -r requirements.txt
```

### Install Pytoch and intel-extension-for-pytorch for XPU
**Note** intel-extension-for-pytorch is optional

For more infomation please refer to https://pytorch.org/docs/stable/notes/get_start_xpu.html

Linux
```
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.7.10+xpu oneccl_bind_pt==2.7.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```
Windows
```
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/
```

## Start Up
Start redis vector database
`docker compose -f docker-compose-redis.yml up -d`

Linux version
```
cd retrieval
python image_retrieval.py
```

Windows:

```
git checkout windows
python search_guit.py
```
