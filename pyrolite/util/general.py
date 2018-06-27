def flatten_dict(d):
    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    sk = subkey # key + "." + subkey
                    sv = subvalue
                    yield  sk, sv
            else:
                yield key, value

    return dict(items())

def swap_item(list: list, pull: str, push: str):
    return [push if i == pull else i for i in list]
