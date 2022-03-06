from pathlib import Path

import fasttext

if __name__ == "__main__":
    targets = [
        "red", "black", "orange", "white", "gray/black",  # colors
        "sony", "apple", "canon", "nikon", "dell",  # brands
        "32", "an", "the", "4", "to",  # numbers and stopwords
        "inch", "cm", "oz", "gb", "mb",  # measurements
        "camera", "gps", "mp3", "iphone", "playstation"  # products
    ]
    base = Path("/Users/fredriko/PycharmProjects/search_with_machine_learning_course/workspace/titles")
    training_data = base / "titles.txt"
    model_file = base / "model_file"
    kwargs = {
        "input": str(training_data),
        "epoch": 100,
        "ws": 6,
        "minn": 0,
        "maxn": 0,
        "dim": 150,
        "model": "skipgram"
    }

    for min_count in [25]:
        print(f"Training with min_count: {min_count}")
        kwargs["minCount"] = min_count
        model = fasttext.train_unsupervised(**kwargs)
        model.save_model(str(model_file))
        for target in targets:
            print(f"Target: {target}")
            nns = model.get_nearest_neighbors(target, 10)
            for nn in nns:
                print(f"{nn[1]} -- {round(nn[0], 3)}")
            print("\n")
