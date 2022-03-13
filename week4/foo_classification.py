import xml.etree.ElementTree as ET
from pathlib import Path
from pprint import pprint
from typing import Optional, Dict, Any

import fasttext
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from week3.utilities.foo_classification import write_dataframe_as_text, load_fasttext_as_dataframe, stratify_dataframe, \
    assess_random_baseline


def read_query_file(file: Path, category_column: str = "category", query_column: str = "query",
                    nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_csv(file, nrows=nrows)[[category_column, query_column]]


def normalize_text(text: str, stem: bool = True, remove_stop_words: bool = True, lower_case: bool = True,
                   remove_digits: bool = False, remove_punctuation: bool = True) -> str:
    stemmer = SnowballStemmer("english")
    stop_words = set(stopwords.words('english'))
    text = text.replace("®", "").replace("™", "")
    if remove_digits:
        chars = [c for c in text if not c.isdigit()]
        text = "".join(chars)
    if lower_case:
        results = word_tokenize(text.lower())
    else:
        results = word_tokenize(text)
    if remove_stop_words:
        results = [w for w in results if w not in stop_words]
    if remove_punctuation:
        punctuation = ".!?()[]{}\\`/-.':`,<>.".split()
        punctuation.append("''")
        punctuation.append("-")
        punctuation = set(punctuation)
        results = [w for w in results if w not in punctuation]
    if stem:
        results = [stemmer.stem(t) for t in results]
    return " ".join(results)


def read_category_file(file: Path) -> pd.DataFrame:
    # The root category, named Best Buy with id cat00000, doesn't have a parent.
    root_category_id = 'cat00000'
    tree = ET.parse(str(file))
    root = tree.getroot()
    categories = []
    parents = []
    for child in root:
        cat_path = child.find('path')
        cat_path_ids = [cat.find('id').text for cat in cat_path]
        leaf_id = cat_path_ids[-1]
        if leaf_id != root_category_id:
            categories.append(leaf_id)
            parents.append(cat_path_ids[-2])
    return pd.DataFrame(list(zip(categories, parents)), columns=['category', 'parent'])


def get_category_id_name_map(file: Path) -> Dict[str, str]:
    tree = ET.parse(file)
    root = tree.getroot()
    id_name_map: Dict[str, str] = {}
    for child in root:
        category_path = child.find("path")
        category_path_list = []
        for category in category_path:
            id_name_map[category.find("id").text] = category.find("name").text
    return id_name_map


def print_results(model, input_path, k):
    num_records, precision_at_k, recall_at_k = model.test(input_path, k)
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

    print("records\t{}".format(num_records))
    print("Precision@{}\t{:.3f}".format(k, precision_at_k))
    print("Recall@{}\t{:.3f}".format(k, recall_at_k))
    print("F1@{}\t{:.3f}".format(k, f1_at_k))
    print()


def stringify_dict(a: Dict[str, Any]) -> str:
    return str(a).replace("{", "").replace("}", "").replace("'", "").replace(":", "-") \
        .replace(",", "_").replace(" ", "")


def create_training_data_sc() -> None:
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace")
    mapping_file = base / "datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml"
    query_file = base / "datasets/train.csv"
    fasttext_base = base / "queries"
    min_queries = [100, 1000]

    for mq in min_queries:
        print(f"Minimum number of queries per label: {mq}")
        query_df = read_query_file(query_file)
        category_df = read_category_file(mapping_file)
        query_df = query_df[query_df["category"].isin(category_df["category"].values)]
        category_counts = query_df.groupby("category").size().reset_index(name="count")
        category_counts = category_counts[category_counts["count"] < mq]
        print(f"Unique categories in query file before roll-up: {query_df['category'].nunique()}")
        while category_counts.shape[0] > 0:
            category_counts = category_counts.merge(category_df, on="category", how="left")
            categories = category_counts["category"].tolist()
            parents = category_counts["parent"].tolist()
            query_df["category"].replace(categories, parents, inplace=True)
            category_counts = query_df.groupby("category").size().reset_index(name="count")
            category_counts = category_counts[category_counts["count"] < mq]

        print(f"Unique categories in query file after roll-up: {query_df['category'].nunique()}")
        query_df = query_df[query_df["category"].isin(category_df["category"]).values]
        query_df["category"] = "__label__" + query_df["category"]
        query_df["query"] = query_df["query"].apply(lambda x: normalize_text(x))
        write_dataframe_as_text(query_df, fasttext_base / f"categories_{mq}.fasttext", label_column="category",
                                text_column="query")


def stratify_split_sc() -> None:
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/queries")

    for i in [100, 1000]:
        print(f"Processing categories: {i}")
        data_file = base / f"categories_{i}.fasttext"
        df = load_fasttext_as_dataframe(data_file)
        train, test = stratify_dataframe(df, split_size=50000)
        assess_random_baseline(train, base / f"random_baseline_categories_{i}.txt")
        write_dataframe_as_text(train, base / f"train_{i}.fasttext")
        write_dataframe_as_text(test, base / f"test_{i}.fasttext")


def classification_sc() -> None:
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/queries")

    for num_cat in [100, 1000]:
        for epoch in [25, 50]:
            for wordNgrams in [1, 2]:
                a = {
                    "epoch": epoch,
                    "lr": 0.1,
                    "dim": 100,
                    "minn": 0,
                    "maxn": 3,
                    "wordNgrams": wordNgrams
                }
                train_data = base / f"train_{num_cat}.fasttext"
                test_data = base / f"test_{num_cat}.fasttext"
                model_file = base / f"fasttext_{num_cat}-{stringify_dict(a)}"

                print(f"Classification - minimum number of instances per label: {num_cat}")
                print("Parameters used for training:")
                pprint(a)
                print(f"Training data: {train_data}")
                print(f"Testing data: {test_data}")
                print(f"Model file: {model_file}")

                model = fasttext.train_supervised(str(train_data), **a)
                for k in [1, 3, 5]:
                    print(f"Test metrics at k={k}:")
                    print_results(model, str(test_data), k)
                model.save_model(str(model_file))

    print(f"Autotuning on train_100.fasttext")
    train_data = str(base / "train_100.fasttext")
    test_data = str(base / "test_100.fasttext")
    model_file = str(base / "fasttext_100_autotuned")
    model = fasttext.train_supervised(input=train_data, autotuneValidationFile=test_data)
    for k in [1, 3, 5]:
        print(f"Test metrics at k={k}:")
        print_results(model, str(test_data), k)
    model.save_model(model_file)


if __name__ == "__main__":
    classification_sc()
