from oct_segmenter import CONFIG, CONFIG_FILE_PATH, MODELS_INDEX_MAP, MODELS_TABLE_ASCII

def list_models():
    models_len = len(MODELS_INDEX_MAP)
    print(f"There are {models_len} choices for the models:\n")
    print(MODELS_TABLE_ASCII)
    print()
    val = input("Press <enter> to keep the current choice[*], or type selection number:")
    if val == "":
        return

    try:
        model_index = int(val)
    except ValueError:
        print(f"Input should be an int between 0 and {models_len-1}")
        exit(1)

    if model_index >= models_len:
        print(f"Input should be an int between 0 and {models_len-1}")
    else:
        update_default_model(MODELS_INDEX_MAP[model_index])

def update_default_model(model_name) -> None:
    CONFIG["User"]["model_dir"] = model_name
    with open(CONFIG_FILE_PATH, "w") as config_file:
        CONFIG.write(config_file)
