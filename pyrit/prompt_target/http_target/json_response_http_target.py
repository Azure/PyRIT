

def _fetch_key(data: dict, key: str) -> str:
    """
    Credit to @Mayuraggarwal1992
    Fetches the answer from the HTTP JSON response based on the path.

    Args:
        data (dict): HTTP response data.
        key (str): The key path to fetch the value.

    Returns:
        str: The fetched value.
    """
    pattern = re.compile(r"([a-zA-Z_]+)|\[(\d+)\]")
    keys = pattern.findall(key)
    for key_part, index_part in keys:
        if key_part:
            data = data.get(key_part, None)
        elif index_part and isinstance(data, list):
            data = data[int(index_part)] if len(data) > int(index_part) else None
        if data is None:
            return ""
    return data
