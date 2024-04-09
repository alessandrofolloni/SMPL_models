import json
from typing import Any


def json_load(path):
    # type: (str) -> Any
    """
    Load a JSON file from the specified path and return the parsed data.

    :param path: the path to the JSON file to load
    :return: the parsed data from the JSON file
    """
    with open(path, 'r') as f:
        return json.load(f)


def json_dump(obj, path):
    # type: (Any, str) -> None
    """
    Dump the specified object to a JSON file at the specified path.

    :param obj: the object to dump to the JSON file.
    :param path: the path to the JSON file to create and write to.
    """
    with open(path, 'w') as f:
        json.dump(obj, f)
