# REID - AOD - MGC

## Project Structure

```
ReID-AOD-MGC/
├── dataset/                  # Data processing and loading
│   └── readdataset.py         # Dataset implementation      
├── model/                # Model architecture
│   ├── Anchor.py    # AOD blocks
│   └── orientation.py   # orientation blocks
|   └── encoder.py   # MGC blocks
|   └── aod_mgc.py   # AOD_MGC model
├── reid.py                # Test
└── readme.md              # This file
```



## Requirements

Please see `requirements.txt` for a complete list of dependencies.



## Installation

1. Clone this repository:

   ```
   git clone https://github.com/dpjqw195/ReID-AOD-MGC.git
   cd ReID-AOD-MGC
   ```

   

2. Create a virtual environment (recommended):

   ```
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows
   ```



3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```



## Usage

#### Data Preparation
The file shared via Baidu Netdisk is: data.rar
Link: https://pan.baidu.com/s/12RCz2Af2M2-9UsstXvQHTg
Extraction code: 1234

`Unzip the compressed file named 'data.rar'.`

#### Data Structure

```
data/
├── train/
│   ├── class1/
│   │   ├── 1.npy
│   │   └── 2.npy
│   └── class2/
│       ├── 1.npy
│       └── 2.npy
├── test/
│   ├── class31/
│   │   ├── 1.npy
│   │   └── 2.npy
│   └── class32/
│       ├── 1.npy
│       └── 2.npy
```



#### Evaluation
Model parameters: model_34.pth  
Link: [https://pan.baidu.com/s/1PsadSU6h5dWS8s_rhJ_JKA?pwd=1234](https://pan.baidu.com/s/1PsadSU6h5dWS8s_rhJ_JKA)
Extraction code: 1234
<br>
To test model:

```
python reid.py
```


