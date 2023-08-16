class KeywordTrie:
    # Maybe implement compressed trie for further efficiency in the future.
    """
    A dictionary that uses a trie to store important keyword strings along with
    the line-numbers that these strings and their prefixes appear in.

    Attributes:
        root: the root node of the dictionary.
    """
    class HashTrieNode:
        """
        An object that represents a single character in an important keyword.

        Attributes:
            pointers: a dictionary with a character as a key and another HashTrieNode as a value
            line_number: a set containing all the line numbers (based on vim) that contains the current prefix
                (all the characters from root node to current node)
        """
        def __init__(self):
            """
            Constructs a HashTrieNode object
            """
            self.pointers = {}
            self.line_number = set()

    def __init__(self):
        """
        Initializes the root node of a HashTrieMap dictionary
        """
        self.root = self.HashTrieNode()

    def find(self, key: str):
        """
        Returns a list of line numbers of the lines that contain every keyword (separated by whitespace)
        of the given key string (order-insensitive).

        For example, "hamburger soda" and "soda hamburger" will return the same list of line numbers.

        :param key: a string containing all the keywords desired in a line (separated by whitespace)

        :return: a list of line numbers of the lines with all the desired keywords in the given key string
        """
        key = key
        temp_string = key.split()
        intersection_set = None

        for current_word in temp_string:
            current_set = self.find_helper(current_word)

            if not current_set:
                return None

            if not intersection_set:
                intersection_set = current_set
            else:
                intersection_set = intersection_set & current_set

        return sorted(intersection_set)

    def find_helper(self, keyword: str):
        """
        Returns a set of line numbers of the lines that contains the given keyword.

        :param keyword: the keyword desired in a line

        :return: a list of line numbers of lines with the desired keyword
        """
        if keyword is None:
            raise ValueError("Cannot find null value.")

        current_node = self.root

        if len(keyword) != 0:
            for current_char in keyword:
                current_node = current_node.pointers.get(current_char)

                if current_node is None:
                    return None

        return current_node.line_number

    def insert(self, key: str, value: int):
        """
        Inserts a keyword into the HashTrieMap dictionary.

        :param key: the keyword being added

        :param value: the line number where the keyword appears in

        :return:
        """
        if key is None or value is None:
            raise ValueError("Cannot insert null values.")

        current_node = self.root

        for current_char in key:
            if current_char not in current_node.pointers.keys():
                current_node.pointers.update({current_char: self.HashTrieNode()})

            current_node = current_node.pointers.get(current_char)
            current_node.line_number.add(value)
