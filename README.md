# Sentence Classification

### Data
This project classifies pair of textual data into entailment, contradiction, or neutral. data are in 15 different languages.
[more deatails on the data](https://www.kaggle.com/competitions/contradictory-my-dear-watson)


### Architecture
This is neural network based implementation on pytorch, where multiple pretrained hugging face models are tuned and optimized on a variety of hyperparameters.

- #### Augmentation   
    - Further, data augmentation is performed using the [M2M100](https://huggingface.co/facebook/m2m100_418M) a multilingual encoder-decoder (seq-to-seq) model trained for many-to-many multilingual translation. from hugging face. [~more on augmentation](https://www.kaggle.com/datasets/blessontomjoseph/contradictory-my-dear-watsonmore-data?select=en_to_bg.cs)

- #### Models used for classification
    - [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) 
    - [bert-base-multilingual-cased](https://huggingface.co/bert-base-multilingual-cased)
    - [sentence-transformers/paraphrase-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2)


- #### Optimization: ![stuff](rd_files/frame1.jpg)
    - the scoring used is f1 score
    - the hyperparams are optimized on optuna with the objective of maximizing the mean of best f1_score over all of the folds.

- suff optimized and selected:
    - model architecture
    - learning rate
    - traing batch size
    - validation batch size
    - dropout probability
  
- #### Training
    - the entire data is trained on the best found parametersand the best model state is saved on the model directory.

### Test the model:
To build the Docker image for the project, run the following command:

```http
docker build -t classification-model .

```
This command will create a Docker image with the name classification-model using the Dockerfile in the root directory of the project.



To run the Docker image, use the following command:
```http
docker run classification-model

```
This will start the classification model using the parameters specified in the Dockerfile and entrypoint.

1. You can then access the API using a client tool such as Postman or by making a request from your web browser.

2. To make a request to the API, you need to provide a query parameter key in the URL. For example, to get the prediction for the key "abc", you can send a GET request to http://0.0.0.0:5000/predicts?key=abc.

3. The API will return a JSON response with the following format:

```http
{
  "response": {
    "key": "abc",
    "time": 0.123456,
    "ans": "prediction for key abc"
  }
}
```
The time field shows the time taken to compute the prediction, and the ans field contains the prediction itself.

