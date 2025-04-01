import logging

import pandas as pd
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading dataset")
    dataset = load_dataset("agemagician/uniref50")
    logger.info("Dataset loaded")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    valid_dataset = dataset["validation"]

    print("features of train_dataset", train_dataset.features)
    print("number of rows in train_dataset", len(train_dataset))

    # we get rid of sequence longer than 1024
    train_dataset = train_dataset.filter(lambda x: len(x["text"]) <= 1024)
    test_dataset = test_dataset.filter(lambda x: len(x["text"]) <= 1024)
    valid_dataset = valid_dataset.filter(lambda x: len(x["text"]) <= 1024)

    print("number of rows in train_dataset", len(train_dataset))

    train_dataset = train_dataset.shuffle(seed=42).select(range(500000))

    # CSV with id name text
    df = pd.DataFrame(train_dataset)
    df.to_csv("train_data.csv", index=False)

    # CSV with id name text
    df = pd.DataFrame(test_dataset)
    df.to_csv("test_data.csv", index=False)

    # CSV with id name text
    df = pd.DataFrame(valid_dataset)
    df.to_csv("valid_data.csv", index=False)


if __name__ == "__main__":
    main()
