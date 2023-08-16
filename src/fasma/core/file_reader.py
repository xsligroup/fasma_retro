from fasma.core.keyword_trie import KeywordTrie as kt
import warnings


def check_line(line, current_kt, current_lines):
    current_lines.append(line)
    temp_line = line.strip().split()
    valid_line = False
    for current_word in temp_line:
        if not current_word.isnumeric():
            if current_word.isalpha() or (
                    current_word.__contains__("=") or current_word.__contains__(":")):
                valid_line = True
                break
    if valid_line:
        for current_word in temp_line:
            current_kt.insert(current_word, len(current_lines))


def read_gaussian(filename):
    try:
        with open(filename, "r") as f:
            list_of_key_tries = []
            list_of_lines_list = []
            current_kt = kt()
            current_lines = []
            for line in f:
                # Check for read-in coordinates (iop 1/29 = 6 or 7) to standardize text format
                if "Redundant internal coordinates found in file" in line or "Z-Matrix found in chk file" in line:
                    line = "Symbolic Z-matrix:"
                if "Recover connectivity data from disk" in line:
                    line = ""
                check_line(line, current_kt, current_lines)
                if "Normal termination" in line:
                    list_of_key_tries.append(current_kt)
                    list_of_lines_list.append(current_lines)
                    current_kt = kt()
                    current_lines = []
            if len(current_lines) > 0:
                list_of_key_tries.append(current_kt)
                list_of_lines_list.append(current_lines)
    except OSError:
        raise OSError("The file with the given path cannot be opened. Please try again.")
    else:
        for current_line_list in list_of_lines_list:
            if "Normal termination" not in current_line_list[-1]:
                warnings.warn("This file was not terminated normally. Check if this is the intended .log file.")
    return list_of_key_tries, list_of_lines_list


def read_chronus(filename):
    try:
        with open(filename, "r") as f:
            list_of_key_tries = []
            list_of_lines_list = []
            current_kt = kt()
            current_lines = []
            parsed = False
            found = False
            start = 0
            job_complete = False
            job_complete_counter = 0
            for line in f:
                if job_complete:
                    job_complete_counter += 1
                temp_line = line
                # Checks if file has passed user input section of Chronus .out file
                if not found and "Input File" in line:
                    found = True
                    start = len(current_lines) + 2
                # Uppercase lines for standardization in user input section
                if not parsed and found and start <= len(current_lines):
                    temp_line = temp_line.upper()
                    # Checks if arrived at the end of user input section
                    if "====" in line:
                        parsed = True
                check_line(temp_line, current_kt, current_lines)
                if "ChronusQ Job Ended" in line:
                    job_complete = True
                if job_complete_counter == 2:
                    list_of_key_tries.append(current_kt)
                    list_of_lines_list.append(current_lines)
                    current_kt = kt()
                    current_lines = []
            if len(current_lines) > 0:
                list_of_key_tries.append(current_kt)
                list_of_lines_list.append(current_lines)
    except OSError:
        raise OSError("The file with the given path cannot be opened. Please try again.")
    else:
        for current_line_list in list_of_lines_list:
            if "ChronusQ Job Ended" not in current_line_list[-3]:
                warnings.warn("This file was not terminated normally. Check if this is the intended .out file.")
    return list_of_key_tries, list_of_lines_list


def read(filename):
    if filename.endswith('.log'):
        list_of_key_tries, list_of_lines_list = read_gaussian(filename)
        file_type = "Gaussian"
    elif filename.endswith('.out'):
        list_of_key_tries, list_of_lines_list = read_chronus(filename)
        file_type = "ChronusQ"
    return list_of_key_tries, list_of_lines_list, file_type






