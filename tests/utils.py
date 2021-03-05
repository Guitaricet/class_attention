import os

GLOVE_TMP_PATH = "glove.glove"
CLASS_NAMES = {
    "ARTS",
    "ARTS & CULTURE",
    "BLACK VOICES",
    "BUSINESS",
    "COLLEGE",
    "COMEDY",
    "CRIME",
    "CULTURE & ARTS",
    "DIVORCE",
    "EDUCATION",
    "ENTERTAINMENT",
    "ENVIRONMENT",
    "FIFTY",
    "FOOD & DRINK",
    "GOOD NEWS",
    "GREEN",
    "HEALTHY LIVING",
    "HOME & LIVING",
    "IMPACT",
    "LATINO VOICES",
    "MEDIA",
    "MONEY",
    "PARENTING",
    "PARENTS",
    "POLITICS",
    "QUEER VOICES",
    "RELIGION",
    "SCIENCE",
    "SPORTS",
    "STYLE",
    "STYLE & BEAUTY",
    "TASTE",
    "TECH",
    "THE WORLDPOST",
    "TRAVEL",
    "WEDDINGS",
    "WEIRD NEWS",
    "WELLNESS",
    "WOMEN",
    "WORLD NEWS",
    "WORLDPOST",
}


def make_glove_file():
    # if os.path.exists(GLOVE_TMP_PATH):
    #     raise RuntimeError("glove_tmp_path exists")

    with open(GLOVE_TMP_PATH, "w") as f:
        used_names = set()
        for class_name in CLASS_NAMES:
            for word in class_name.split(" "):
                word = word.lower()
                if word in used_names:
                    continue

                f.write(f"{word} 42 42 42\n")
                used_names.add(word)


def delete_glove_file():
    os.remove(GLOVE_TMP_PATH)
