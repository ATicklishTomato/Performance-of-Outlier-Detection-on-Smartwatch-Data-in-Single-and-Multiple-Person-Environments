from datetime import datetime

current_files = {
    "single_gmm": None,
    "multiple_gmm": None,
    "single_dbscan": None,
    "multiple_dbscan": None,
    "subject_mean_std": None,
}


def initialize_files():
    for key in current_files.keys():
        current_files[key] = open(f"./results/{key}-{str(datetime.now()).split(' ')[0]}.csv", "a+")


def close_files():
    for key in current_files.keys():
        current_files[key].close()


def write_to_file(key, values: list):
    current_files[key].write(",".join([str(x) for x in values]) + "\n")
    current_files[key].flush()

def write_bulk_to_file(key, text: str):
    current_files[key].write(text)
    current_files[key].flush()
