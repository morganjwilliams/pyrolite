from pandas import DataFrame, Series


def b_s_x2_to_df(B, s, X2, index, names, add_uncertainties, add_X2):
    lambdas = DataFrame(
        B,
        index=index,
        columns=names,
        dtype="float32",
    )
    if add_uncertainties:
        lambdas.loc[:, [n + "_" + chr(963) for n in names]] = s
    if add_X2:
        lambdas["X2"] = X2
    return lambdas


def b_s_x2_to_series(B, s, X2, names, add_uncertainties, add_X2):
    lambdas = Series(
        B,
        index=names,
        dtype="float32",
    )
    if add_uncertainties:
        for i, n in enumerate(names):
            lambdas.loc[n + "_" + chr(963)] = s[i]
    if add_X2:
        lambdas["X2"] = X2
    return lambdas
