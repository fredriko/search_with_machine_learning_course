import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit


def load_fasttext_as_dataframe(fasttext_file: Path) -> pd.DataFrame:
    c: List[Dict[str, Any]] = []
    with fasttext_file.open(mode="r") as fh:
        for line in fh.readlines():
            parts = line.strip().split()
            c.append({
                "label": parts[0],
                "text": " ".join(parts[1:])
            })
    return pd.DataFrame(c)


def drop_infrequent_labels(df: pd.DataFrame, min_num_labels: int = 0, label_column: str = "label") -> pd.DataFrame:
    label_counts = df[label_column].value_counts()
    return df[df[label_column].isin(label_counts.index[label_counts.ge(min_num_labels)])]


def stratify_dataframe(df: pd.DataFrame, label_column: str = "label", split_size: int = 10000) -> Tuple[
    pd.DataFrame, pd.DataFrame]:
    """
    Stratifies and shuffles a Pandas dataframe based on the label_column and split_size.

    :param df: The dataframe to stratify and shuffle.
    :param label_column: The target column.
    :param split_size: The size of each split, in number of rows in the dataframe.
    :return: Two dataframes of equal size (split_size), the first of which is intended as training data and
    the second as test data.
    """
    num_datapoints = df.shape[0]
    n_splits = int(num_datapoints / split_size)

    df = drop_infrequent_labels(df, n_splits, label_column=label_column)

    sss = StratifiedShuffleSplit(n_splits=n_splits, train_size=split_size, test_size=split_size, random_state=4711)
    train_index, test_index = list(sss.split(df, df[label_column]))[0]
    return df.iloc[train_index], df.iloc[test_index]


def assess_random_baseline(df: pd.DataFrame, report_file: Path, text_column: str = "text",
                           label_column: str = "label") -> None:
    """
    Assesses a stratified random baseline across the label_column labels for the data in df.

    :param df: The dataframe containing the labels and data.
    :param report_file: The file to which the classification report will be written.
    :param text_column: The name of the column containing the text in df.
    :param label_column: The name of the column containing the labels in df.
    :return: None.
    """
    clf = DummyClassifier(strategy="stratified", random_state=4711)
    clf.fit(df[text_column], df[label_column])
    r = classification_report(df[label_column], clf.predict(df[text_column]), labels=df[label_column].unique(),
                              output_dict=True)

    with report_file.open(mode="w") as fh:
        fh.write(f"Random baseline for classification\nNumber of classes: {len(r) - 3}\n\n")
        fh.write(f"Macro average\n{r['macro avg']}\n\n")
        fh.write(f"Micro average\n{r['weighted avg']}\n")


def write_dataframe_as_text(df: pd.DataFrame, file: Path, label_column: str = "label",
                            text_column: str = "text") -> None:
    with file.open(mode="w") as fh:
        for _, r in df.iterrows():
            fh.write(f"{r[label_column]} {r[text_column]}\n")


def load_leaf_hierarchy_map(mapping_file: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Get the mapping from leaf id in fasttext label form to the hierarchy ending in the leaf, e.g.,
    __label__abcat0013003': [('Best Buy', 'cat00000'), ('Gift Center', 'abcat0010000'), ('Teens', 'abcat0013000'),
                             ('Computing', 'abcat0013003')]

    :param mapping_file: The file containing the mapping information.
    :return: The mapping as described above.
    """
    tree = ET.parse(mapping_file)
    root = tree.getroot()
    leaf_hierarchy_map: Dict[str, Any] = {}
    for child in root:
        category_path = child.find("path")
        category_path_list = []
        for category in category_path:
            category_path_list.append((category.find("name").text, category.find("id").text))
        leaf_hierarchy_map[f"__label__{category_path_list[-1][1]}"] = category_path_list
    return leaf_hierarchy_map


def drop_infrequent_labels_sc() -> None:
    print(f"Running drop_infrequent_labels_sc()")
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/categories")
    fasttext_data = base / "output.fasttext"
    training_data = base / "training_data"
    testing_data = base / "testing_data"
    random_baseline = base / "random_baseline"
    df = load_fasttext_as_dataframe(fasttext_data)
    for min_num_instances in [50, 100, 200]:
        print(f"Filtering on min_num_instances: {min_num_instances}")
        training = training_data.with_suffix(f".min_num_instances_{min_num_instances}.fasttext")
        testing = testing_data.with_suffix(f".min_num_instances_{min_num_instances}.fasttext")
        baseline = random_baseline.with_suffix(f".min_num_instances_{min_num_instances}.txt")
        df_ = drop_infrequent_labels(df, min_num_instances)
        print(
            f"Min num instances: {min_num_instances}: original: {df.shape[0]} - lost {df.shape[0] - df_.shape[0]} instances")
        df_train, df_test = stratify_dataframe(df_)
        write_dataframe_as_text(df_train, training)
        write_dataframe_as_text(df_test, testing)
        assess_random_baseline(df_train, baseline)


def stratify_data_sc() -> None:
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/categories")
    fasttext_data = base / "output.fasttext"
    training_data = base / "training_data2.fasttext"
    testing_data = base / "testing_data2.fasttext"
    random_baseline = base / "random_baseline2.txt"

    df = load_fasttext_as_dataframe(fasttext_data)
    df_train, df_test = stratify_dataframe(df)
    write_dataframe_as_text(df_train, training_data)
    write_dataframe_as_text(df_test, testing_data)
    assess_random_baseline(df_train, random_baseline)


def assess_random_baseline_sc() -> None:
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/categories")
    training_data_lvl_2 = base / "training_data2.level_2.fasttext"
    random_baseline_lvl_2 = base / "random_baseline_level_2.txt"
    training_data_lvl_3 = base / "training_data2.level_3.fasttext"
    random_baseline_lvl_3 = base / "random_baseline_level_3.txt"

    df_train_lvl_2 = load_fasttext_as_dataframe(training_data_lvl_2)
    assess_random_baseline(df_train_lvl_2, random_baseline_lvl_2)

    df_train_lvl_3 = load_fasttext_as_dataframe(training_data_lvl_3)
    assess_random_baseline(df_train_lvl_3, random_baseline_lvl_3)


def prune_hierarchy_sc() -> None:
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace")
    mapping_file = base / "datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml"
    training_data = base / "categories/training_data2.fasttext"
    testing_data = base / "categories/testing_data2.fasttext"
    leaf_hierarchy_map = load_leaf_hierarchy_map(mapping_file)
    df_training = load_fasttext_as_dataframe(training_data)
    df_testing = load_fasttext_as_dataframe(testing_data)
    for level in [2, 3]:
        level_map = {}
        for label, hierarchy in leaf_hierarchy_map.items():
            if len(hierarchy) > level:
                level_map[label] = f"__label__{hierarchy[level][1]}"
            else:
                level_map[label] = f"__label__{hierarchy[-1][1]}"

        def map_label(lbl: str) -> str:
            return level_map.get(lbl, "__label__not_available")

        df_tr = df_training.copy()
        df_te = df_testing.copy()
        df_tr["new_label"] = df_tr["label"].apply(lambda x: map_label(x))
        df_te["new_label"] = df_te["label"].apply(lambda x: map_label(x))
        tr_file = training_data.with_suffix(f".level_{level}.fasttext")
        te_file = testing_data.with_suffix(f".level_{level}.fasttext")
        print(f"Writing training data to file: {tr_file}")
        write_dataframe_as_text(df_tr, tr_file, label_column="new_label")
        print(f"Writing testing data to file: {te_file}")
        write_dataframe_as_text(df_te, te_file, label_column="new_label")


if __name__ == "__main__":
    assess_random_baseline_sc()
