from display import Display


def plot_(output, *args):
    return Display(output).build(*args)


def get_info_(bn, as_df):
    return Display(output=None).get_info(bn, as_df)
