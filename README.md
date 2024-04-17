## Setup
(On GPU) Downgrade CUDA to 11.7 to be compatible with fast projection.
```sh
conda install -y -n [env] python=3.10
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install trak[fast]
```

Install dependencies
```sh
pip install -r requirements.txt
```

Download OLMo and Dolma
```sh
cd data
chmod +x get_dolma.sh
# - Get an HF access token: https://huggingface.co/settings/tokens
# - Confirm Dolma dataset access: https://huggingface.co/datasets/allenai/dolma
./get_dolma.sh
```
