def gaussian_missing_msg():
    return missing_msg("log")


def chronus_missing_msg():
    return missing_msg("out")


def missing_msg(output_type):
    msg = f" cannot be found in the given .{output_type} file. This is an invalid .{output_type} file.\n Please use a different file."
    return msg
