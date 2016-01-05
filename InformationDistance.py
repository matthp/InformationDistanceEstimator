from CTW import CTW

# Estimates the information distance between x and y
# x and y are arrays containing only ones or zeros (binary) and must be the same length


def estimate_info_distance(x, y, context_length):

    # Return -1 if the lengths of the strings do not match
    if len(x) != len(y):
        return -1

    # Estimate the entropy of X using CTW
    model_x = CTW(context_length)
    model_x.present_bit_string_and_update(x)
    entropy_x = -model_x.compute_log_probability() / len(x)
    model_x = []

    # Estimate the entropy of Y using CTW
    model_y = CTW(context_length)
    model_y.present_bit_string_and_update(y)
    entropy_y = -model_y.compute_log_probability() / len(y)
    model_y = []

    # Estimate the entropy of X conditioned on Y using CTW
    model_xy = CTW(context_length)
    model_xy.present_bit_string_and_update_with_side_information(x, y)
    entropy_xy = -model_xy.compute_log_probability() / len(x)
    model_xy = []

    # Estimate the entropy of Y conditioned on X using CTW
    model_yx = CTW(context_length)
    model_yx.present_bit_string_and_update_with_side_information(y, x)
    entropy_yx = -model_yx.compute_log_probability() / len(y)
    model_yx = []

    # Compute the estimated normalized information distance between X and Y
    return max([entropy_xy, entropy_yx]) / max([entropy_x, entropy_y])
