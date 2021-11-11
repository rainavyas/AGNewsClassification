# Objective

Train a transformer-based architecture to classify the news items in the [AG News](https://huggingface.co/datasets/ag_news) dataset into one of four categories: _World, Sports, Business, Science_. 

# Dependencies

## Install Using PyPI

`pip install datasets`
`pip install torch`
`pip install transformers`

# Train

An example command to train is given below

`python train.py my_model.th electra --B=8 --lr=0.00001 --epochs=2 --seed=1`
