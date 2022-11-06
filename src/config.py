from transformers import AutoModel,AutoTokenizer


class config:
    device = 'cuda'
    epochs = 5
    checkpoint = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    base_model = AutoModel.from_pretrained(checkpoint)
    train_batch_size = 8
    val_batch_size = 8
