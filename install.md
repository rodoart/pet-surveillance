# Installation guide

## Prerequisites

- [Python 3.6 or higher](https://www.python.org/downloads/release/python-368/)
- [Conda](https://docs.conda.io/en/latest/)  (Optional)
- Protocol Buffer [Linux, Mac](https://grpc.io/docs/protoc-installation/#install-using-a-package-manager) or [Windows](https://medium.com/@dev.ashurai/protoc-protobuf-installation-on-windows-linux-mac-d70d5380489d). 
- [OpenCV2](https://pypi.org/project/opencv-python/)


## Clone the repository

```bash
git clone https://github.com/rodoart/pet-surveillance
cd pet-surveillance
```

## Create and activate environment


## Pip

```bash
python -m venv .env
```

Activate the environment:

Linux:

```bash
source .env/bin/activate
```

Windows:

```bash
.\.venv\Scripts\activate
```

Download the libraries:

```
pip install -r requirements.txt
```


## or Conda

```bash
conda create -n pet_surveillance --file requirements.txt
conda activate pet_surveillance
```

# Set up

Compile `protoc`

```bash
protoc tensorflow/models/research/object_detection/protos/*.proto --python_out=.
```

# Start

Run:

```
python main.py
``` 

The program automatically runs the example of the video downloaded before, if you want to capture usb with a camera in real time, you must modify the options in main.py.