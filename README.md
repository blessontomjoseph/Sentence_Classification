# Contradictory sentence classification

### Data
This project classifies pair of textual data into entailment, contradiction, or neutral. data are in 15 different languages.
[~more deatails on the data](https://www.kaggle.com/competitions/contradictory-my-dear-watson)


### Architecture
- This is neural network based implementation on pytorch, where multiple pretrained hugging face models are tuned and optimized on a variety of hyperparameters.

- #### Augmentation   
    - Further, data augmentation is performed using the [M2M100](https://huggingface.co/facebook/m2m100_418M) a multilingual encoder-decoder (seq-to-seq) model trained for many-to-many multilingual translation. from hugging face. [~more on augmentation](https://www.kaggle.com/datasets/blessontomjoseph/contradictory-my-dear-watsonmore-data?select=en_to_bg.cs)

- #### Models used for classification
    - [bert base cased]() 
    - [roberta base]()

- #### Training details:
    - trained on gpu
    - 5 fold cv

- #### Evaluation
    - f1 score
