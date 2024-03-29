def is_model(model):
    try:
        methods = dir(model)
        if "fit" in methods:
            return True
        else:
            return False
    except Exception:
        return False
