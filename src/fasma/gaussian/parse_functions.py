from fasma.core import messages as msg
import re


def overlay_search(file_keyword_trie, file_lines, overlay_string: str) -> int:
    """
    Find and return the line number of given overlay in the Gaussian Link section of the .log file.
    :param file_keyword_trie: the KeyWordTrie object of the current file
    :param file_lines: all the lines of this current file
    :param overlay_string: the desired overlay with a "/"  (overlay string for overlay 9 would be "9/")
    :raise ValueError: the given overlay string cannot be found in the current .log file.
    :return: the line number of given overlay in the Gaussian Link section of the .log file.
    """
    overlay_lines = file_keyword_trie.find(overlay_string)

    if not overlay_lines:
        raise ValueError("Overlay " + overlay_string + msg.gaussian_missing_msg())
    ignore_above = file_keyword_trie.find("#")[0]
    temp_line = file_lines[ignore_above - 1]
    while "----" not in temp_line:
        ignore_above += 1
        temp_line = file_lines[ignore_above - 1]
    for line_num in overlay_lines:
        if line_num <= ignore_above:
            continue
        temp_line = file_lines[line_num - 1]
        temp_line = " ".join(temp_line.split())

        for x in range(len(overlay_string)):
            if temp_line[x] != overlay_string[x]:
                break
            if x + 1 == len(overlay_string):
                return line_num


def find_iop(file_keyword_trie, file_lines, overlay: str, iops: list) -> list:
    """
    Find and return a list containing values associated with the overlay and iops being searched for
    in the order of the given iops list.
    :param file_lines: all the lines of this current file
    :param overlay: the overlay number containing the desired iops
    :param iops: a list containing all the iops being search for
    :raise ValueError: the given file isn't a CAS calculation
    :raise IndexError: the given file is an invalid CAS calculation
    :raise ValueError: the given file doesn't contain a valid SCF type
    :raise IndexError: the given file isn't a Pop calculation
    :return: a list containing values associated with the iops being searched for in order of the given iops list
    """
    overlay_line = file_lines[overlay_search(file_keyword_trie, file_lines, overlay + "/") - 1]
    list_of_words = "".join(overlay_line.split())
    list_of_words = re.split(r"[/,]", list_of_words)
    iop_dict = {}
    for word in list_of_words:
        if "=" in word:
            current_iop_arr = word.split("=")
            iop_dict[current_iop_arr[0]] = current_iop_arr[1]
    list_of_iops = []

    for i, current_iop in enumerate(iops):
        retrieval = iop_dict.get(current_iop)
        if retrieval is None:
            if overlay != "3":
                if current_iop == "42" and overlay == "9":
                    raise ValueError("This is not a TD calculation.")
                elif current_iop == "41" and overlay == "9":
                    raise ValueError("This is not a TD calculation.")
                    #raise IndexError("This TD calculation has only one state.")
                elif current_iop == "32" and overlay == "9":
                    raise ValueError("This is calculation doesn't contain a PDM. Missing IOP " + str(overlay) + "/" + str(current_iop) + ".")
                elif i == 0:
                    raise ValueError("This is not a CAS calculation.")
                elif current_iop == "19" and overlay == "9":
                    list_of_iops.append(1)
                else:
                    raise IndexError("This is not a valid CAS calculation. Missing IOP " + str(overlay) + "/" + str(current_iop) + ".")
            else:
                if current_iop == "116":
                    raise ValueError("This file doesn't contain an SCF type and is invalid. Please choose a different file.")
                if current_iop == "33":
                    raise IndexError("This is not a POP calculation.")
        else:
            list_of_iops.append(int(retrieval))
    return list_of_iops


def replace_d(file_lines, start_line: int, end_line: int, old_string: str = "D", new_string: str = "E"):
    """
    Gaussian Log files display numbers as "0.100000D+01". This has the same numerical value as 0.100000E+01.
    replace_d by default will replace all D with E within a line number range to ensure future conversion
    of certain strings to floats will go smoothly.
    Method can be modified for use in other string replacement problems.
    :param file_lines: all the lines of this current file
    :param start_line: the starting line number (based of log file)
    :param end_line: the ending line number (based of log file and non-inclusive)
    :param old_string: the string being replaced (Default = "D")
    :param new_string: the string that will replace old_string (Default = "E")
    :return:
    """
    for x in range(start_line - 1, end_line - 1):
        line = file_lines[x]
        file_lines[x] = line.replace(old_string, new_string)