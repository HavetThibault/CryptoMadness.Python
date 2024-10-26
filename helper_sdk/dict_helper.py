import hashlib


def hash_to_int8(value):
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16) % (10 ** 8)