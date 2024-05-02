Code for data quality estimation paper for CS 4644.

## (Optional) OLMo Fine-Tuning Setup
For many experiments we fine-tune the OLMo 1B model on instruction datasets. To replicate our setup, please use the Tulu library which provides training scripts. The model used in our work is available at [davidheineman/OLMo-1B-Instruct](https://huggingface.co/davidheineman/OLMo-1B-Instruct).

```sh
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
pip install -r requirements.txt
./scripts/prepare_train_data.sh
./scripts/finetine_with_accelerate.sh
```

## Attribution Setup
We performed our experiments on 1 NVIDIA A40 GPU. Below are instructions to setup data attribution on a GPU configuration:
```sh
conda install -y -n [env] python=3.10
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia # Downgrade CUDA to 11.7 to be compatible with fast projection
pip install trak[fast]
```

Install dependencies
```sh
pip install -r requirements.txt
```

To run attribution on a test dataset (e.g., LIMA+MMLU):
```sh
cd src
python trak_olmo.py
```

To train a BERT prediction model on attribution scores:
```sh
python attribution_bert.py
```

To run attribution prediction using a trained BERT model:
```sh
python run_attribution_bert.py
```

## Prediction & Analysis
Sections 4-5 were performed in Google Collab. Our project drive is available at [drive.google.com/drive/folders/174WnPDXGnrzSYUB6bzVNlOnz0NnMZGIH](https://drive.google.com/drive/folders/174WnPDXGnrzSYUB6bzVNlOnz0NnMZGIH).
