from fasma.core import boxes as bx
from fasma.gaussian import parse_basic
from fasma.gaussian import parse_td
from fasma.gaussian import parse_cas
from fasma.gaussian import parse_pop


def parse(file_keyword_trie, file_lines):
    basic = parse_basic.get_basic(file_keyword_trie, file_lines)
    spectra = parse_td.check_td(basic, file_keyword_trie, file_lines)
    if spectra is None:
        spectra = parse_cas.check_cas(basic, file_keyword_trie, file_lines)
    pop = parse_pop.check_pop(basic, file_keyword_trie, file_lines, False)
    box = bx.Box(basic_data=basic, spectra_data=spectra, pop_data=pop)
    return box

