# ElasticTraining
ElasticTraining
## Setup
- Create and get into the conda env
```bash
conda create -n ehgq python=3.12 -y
conda activate ehgq
```
- install the HGQ2
```bash
pip install HGQ2==0.1.6
python -m pip show HGQ2
```
- install the HGQ2
da4ml 0.5.1 has bug with model test in its layer part, they are currently rafactoring it.
```bash
pip install da4ml==0.5.0
```
- install the hls4ml
```bash
pip install hls4ml>=1.2.0
```
- install the torch cpu version
```bash
pip install "tensorflow-cpu==2.16.1" 
# pip install tensorflow
pip install pandas
pip install scikit-learn
pip install matplotlib
```