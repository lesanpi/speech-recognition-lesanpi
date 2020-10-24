class TextTransform:
    """Caracteres a Enteros y viceversa."""
    def __init__(self):
        map_str = """
        ' 0
        <SPACE> 1
        a 2
        b 3
        c 4
        d 5
        e 6
        f 7
        g 8
        h 9
        i 10
        j 11
        k 12
        l 13
        m 14
        n 15
        o 16
        p 17
        q 18
        r 19
        s 20
        t 21
        u 22
        v 23
        w 24
        x 25
        y 26
        z 27
        """
        self.char_map = {}
        self.index_map = {}

        for line in map_str.strip().split('\n'):
            char, index = line.split()
            self.char_map[char] = index
            self.index_map[index] = char
        self.index_map[1] = ' '

    def text_to_int(self, text):
        int_sequence = []
        for c in text:
            if c == ' ':
                char = self.char_map['']
            else:
                char = self.char_map[c]
            int_sequence.append(char)
        return int_sequence

    def int_to_text(self, labels):
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('',' ')
