import re

class SentenceTranslator:

    def __init__(self):
        self.words = []

    def add_word(self, word):

        if len(self.words) > 0:
            if self.words[-1] == word:
                return

        self.words.append(word)

    def combine_affixes(words):

        hasil = []

        i = 0

        while i < len(words):

            word = words[i]

            # awalan
            if word.startswith("awalan-"):

                prefix = word.replace("awalan-", "")

                if i + 1 < len(words):
                    combined = prefix + words[i + 1]
                    hasil.append(combined)
                    i += 2
                    continue

            # partikel
            elif word.startswith("partikel-"):

                particle = word.replace("partikel-", "")

                if i + 1 < len(words):
                    combined = particle + words[i + 1]
                    hasil.append(combined)
                    i += 2
                    continue
            # akhiran
            elif word.startswith("akhiran-"):

                suffix = word.replace("akhiran-", "")

                if len(hasil) > 0:
                    hasil[-1] = hasil[-1] + suffix

            else:
                hasil.append(word)

            i += 1

        return hasil

    def get_sentence(self):

        result = []

        for word in self.words:

            result.append(self.clean_word(word))

        sentence = " ".join(result)

        sentence = re.sub(r'\s+', ' ', sentence)

        return sentence.strip()

    def reset(self):

        self.words = []