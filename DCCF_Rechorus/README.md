# Disentangled Contrastive Collaborative Filtering

ReChorus version

## Environment

The codes are written in Python 3.10.12 with the following dependencies.

- numpy == 1.26.4
- pytorch == 2.4.1 (GPU version)
- torch-scatter == 2.1.2
- torch-sparse == 0.6.18
- scipy == 1.13.1

##  Dataset

MovieLens_1M，Grocery_and_Gourmet_Food

## Examples to run the codes

The command to train DCCF on the MovieLens_1M / Grocery_and_Gourmet_Food dataset is as follows.

  - MovieLens_1M

    ```python main.py --model_name DCCF --dataset 'MovieLens_1M/ML_1MTOPK' --batch_size 10240```   

  - Grocery_and_Gourmet_Food:

    ```python main.py --model_name DCCF --dataset 'Grocery_and_Gourmet_Food' --batch_size 10240```
