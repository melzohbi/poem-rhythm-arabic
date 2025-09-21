import re
import random
import numpy as np
import pandas as pd

TANWEEN = ["ً", "ٌ", "ٍ"]
HARAKAT = ["َ", "ُ", "ِ"]
ROUNDED_ZERO = ["۟"]
SUKUN = ["ْ"]
SHADDA_CHAR = ["ّ"]
DAGGER = ["ٰ"]
TASHKEEL = HARAKAT + SUKUN + TANWEEN + SHADDA_CHAR + DAGGER + ROUNDED_ZERO
HAMZA_WASL = ["ٱ"]
MADDAH = ["آ"]
ALL_CHARS = ["ا", "أ", "إ", "ؤ", "ئ", "ء", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ", "ر", "ز", "س",
             "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق", "ك", "م", "ن", "ه", "و", "ي", "ى", "ة", "ل", " "]

ILLA = ["ا", "و", "ي", "ى"]

ALIF_HAMZA = ["أ", "إ"] + HAMZA_WASL

MOON_LETTERS = ["أ", "إ", 'ب', 'ج', 'ح', 'خ',
                'ع', 'غ', 'ف', 'ق', 'ك', 'م', 'ه', 'و', 'ي']
SUN_LETTERS = ['ت', 'ث', 'د', 'ذ', 'ر', 'ز',
               'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ل', 'ن']

ALL_ARABIC_CHARS = ALL_CHARS + TASHKEEL + HAMZA_WASL + MADDAH

EXEPTION_WORDS = {
    "هذا": "هَاْذَاْ",
    "هذه": "هَاْذِهِيْ",
    "هذي": "هَاْذِيْ",
    "هذان": "هَاْذَاْنِ",
    "هذين": "هَاْذَيْنِ",
    "ذلك": "ذَاْلِكَ",
    "ذلكما": "ذَاْلِكُمَاْ",
    "ذلكم": "ذَاْلِكُمْ",

    "هؤلاء": "هَاْؤُلَاْءِ",

    "اللهم": "ٱلْلَاْهُمْمَ",

    "اله": "إِلَاْهَ",
    "الهي": "إِلَاْهِيْ",
    "إلهنا": "إِلَاْهُنَاْ",
    "الهكم": "إِلَاْهُكُمْ",
    "الههم": "إِلَاْهُهُمْ",
    "الههن": "إِلَاْهُهُنْنَ",
    "رحمن": "رَحْمَاْنُ",

    "طاوس": "طَاْوُوْسُ",
    "داود": "دَاْوُوْدُ",
    "طه": "طَاْهَاْ",
    "اسمعيل": "إِسْمَاعِيْلُ",
}

SPECIAL_ZIYADAH = {
    "مائتين": "مِا۟ئَتَيْنِ",
    "مائتان": "مِا۟ئَتَاْنِ",
    "اولئك": "أُو۟لَاْئِكَ",
    "اولئكم": "أُو۟لَاْئِكُمْ",
    "اولو": "أُو۟لُوْ",
    "اولاء": "أُو۟لَاْءِ",
    "اولي": "أُو۟لِيْ",
}


EXEPTION_WORDS_AL = {
    "الذي": "ٱلْلَذِيْ",
    "الذين": "ٱلْلَذِيْنَ",
    "التي": "ٱلْلَتِيْ",
    "الاله": "ٱلْإِلَاْهُ",
    "الرحمن": "ٱلْرَحْمَاْنُ",
    "الله": "ٱلْلَاْهُ",
    "الاله": "ٱلْإِلَاْهُ",
}

FAMOUS_UNAMBIGUOUS_WORDS = {
    "على": "عَلَىْ",
    "او": "أَوْ",
    "عليه": "عَلَيْهِ",
    "عليها": "عَلَيْهَاْ",
    "عليهم": "عَلَيْهِمْ",
    "عليك": "عَلَيْكَ",
    "عليكم": "عَلَيْكُمْ",
    "حتى": "حَتَّىْ",
    "اذا": "إِذَاْ",
    "له": "لَهُوْ",
    "لها": "لَهَاْ",
    "لك": "لَكَ",
    "به": "بِهِيْ",

    "لكن": "لَاْكِنْ",
    "لكنه": "لَاْكِنْنَهُوْ",
    "لكنني": "لَاْكِنْنَنِيْ",
    "لكنك": "لَاْكِنْنَكَ",
    "لكنها": "لَاْكِنْنَهَاْ",
    "لكنهما": "لَاْكِنْنَهُمَاْ",
    "لكنهم": "لَاْكِنْنَهُمْ",
    "لكنهن": "لَاْكِنْنَهُنْنَ",

    "ههنا": "هَاْهُنَاْ",
    "هكذا": "هَاْكَذَاْ",
}

AHRUF_ATF = ["و", "ف"]
AHRUF_JAR = ["ك", "ب", "ل"]
RABT = AHRUF_ATF + AHRUF_JAR


def replace_special_ziyadah(sentence):
    words = sentence.split()
    RABAT_DICTIONARY = {**SPECIAL_ZIYADAH, **EXEPTION_WORDS}
    BIG_DICTIONARY = {**RABAT_DICTIONARY, **
                      EXEPTION_WORDS_AL, **FAMOUS_UNAMBIGUOUS_WORDS}

    for i, word in enumerate(words):
        untash = strip_all_tashkeel(word, replace_hamza=True)

        if untash in BIG_DICTIONARY:
            words[i] = BIG_DICTIONARY[untash]

        elif untash[1:] in RABAT_DICTIONARY and untash[0] in RABT:

            if untash[0] in (AHRUF_ATF or AHRUF_JAR[0]):
                words[i] = untash[0] + HARAKAT[0] + \
                    RABAT_DICTIONARY[untash[1:]]
            else:
                words[i] = untash[0] + HARAKAT[2] + \
                    RABAT_DICTIONARY[untash[1:]]

        elif (ILLA[0] + untash[1:]) in EXEPTION_WORDS_AL and untash[0] == AHRUF_JAR[-1]:
            words[i] = AHRUF_JAR[-1] + HARAKAT[2] + \
                EXEPTION_WORDS_AL[ILLA[0] + untash[1:]][1:]

        elif untash[1:] in FAMOUS_UNAMBIGUOUS_WORDS and untash[0] in AHRUF_ATF:
            words[i] = untash[0] + HARAKAT[0] + \
                FAMOUS_UNAMBIGUOUS_WORDS[untash[1:]]

    return ' '.join(words)


def expand_tanween(text):
    pattern = fr'({TANWEEN[0]})(?=[اى])'
    result = re.sub(pattern, HARAKAT[0] + 'نْ', text)
    pattern = fr'({TANWEEN[1]})(?:(?:$)|(?=\s))'
    result = re.sub(pattern, HARAKAT[1] + 'نْ', result)
    pattern = fr'({TANWEEN[2]})(?:(?:$)|(?=\s))'
    result = re.sub(pattern, HARAKAT[2] + 'نْ', result)
    return result


def expand_tanween_2(text):
    for tanween, harakat in zip(TANWEEN, HARAKAT):
        pattern = fr'({tanween})(?:(?:$)|(?=\s))'
        text = re.sub(pattern, harakat + 'نْ', text)
    return text


def expand_shaddah(text):
    pattern = fr'([{"".join(ALL_CHARS[:-1])}])({SHADDA_CHAR[0]})'
    result = re.sub(pattern, r'\1' + SUKUN[0] + r'\1', text)
    return result


def add_missing_sukuns(st):
    result = []
    i = 0
    while i < len(st):
        if not is_mushakkal(st, i) and st[i] in ALL_CHARS[:-1]:
            result.append(st[i] + SUKUN[0])
        else:
            result.append(st[i])
        i += 1
    return ''.join(result)


def remove_unvocalized_harfs(st):
    return ''.join(char for i, char in enumerate(st) if is_mushakkal(st, i) != ROUNDED_ZERO[0])


def remove_rounded_zero(text):
    # remove all chars that are followed by a rounded zero, remove the rounded zero as well
    pattern = fr'[{"".join(ALL_CHARS[:-1])}]{ROUNDED_ZERO[0]}'
    result = re.sub(pattern, '', text)
    return result


def replace_maddah(st):
    st = st.replace("آ", "أَاْ")
    return st


def is_mutaharik(sentence, index):
    if index == len(sentence) - 1:
        return False
    if sentence[index] in ALL_CHARS[:-1] and sentence[index+1] in HARAKAT:
        return True
    return False


def is_sakin(sentence, index):
    if index == len(sentence) - 1:
        return False
    if sentence[index] in ALL_CHARS[:-1] and sentence[index+1] in SUKUN:
        return True
    return False


def is_mushakkal(sentence, index):
    if index < len(sentence) - 1:
        if sentence[index] in ALL_CHARS[:-1] and sentence[index+1] in TASHKEEL:
            return sentence[index+1]
    return False


def id_unvoc_harfs(text):
    pattern = r'(?:وْاْ|واْ|وْا|وا)(?=\s|$)'
    pattern_2 = fr'(?:{TANWEEN[0]}{ILLA[0]})'
    pattern_3 = fr'(?:{TANWEEN[0]}{ILLA[-1]})'

    result = re.sub(pattern, 'وْا' + ROUNDED_ZERO[0], text)
    result = re.sub(pattern_2, TANWEEN[0] + ILLA[0] + ROUNDED_ZERO[0], result)
    result = re.sub(pattern_3, TANWEEN[0] + ILLA[-1] + ROUNDED_ZERO[0], result)

    # handle special cases in unvocalized harfs
    for word in result.split():

        un_tash_word = strip_all_tashkeel(word)

        # special cases with no case endings
        if un_tash_word in SPECIAL_ZIYADAH:
            result = result.replace(word, SPECIAL_ZIYADAH[un_tash_word])

        # special cases with case endings
        if un_tash_word == "عمرو":
            for i, char in enumerate(word):
                tash = is_mushakkal(word, i)
                if char == "ر" and tash:
                    result = result.replace(
                        word, "عَمْر" + tash + "و" + ROUNDED_ZERO[0])
                    break
            else:
                result = result.replace(
                    word, "عَمْر" + random.choice(HARAKAT + SUKUN + TANWEEN) + "و" + ROUNDED_ZERO[0])
        if un_tash_word == "مائة":
            if word[-1] in (HARAKAT + SUKUN + TANWEEN):
                result = result.replace(word, "مِا۟ئَة" + word[-1])
            else:
                result = result.replace(
                    word, "مِا۟ئَة" + random.choice(HARAKAT + SUKUN + TANWEEN))
        if un_tash_word == "أولات":
            if word[-1] in (HARAKAT + SUKUN):
                result = result.replace(word, "أُو۟لَاْت" + word[-1])
            else:
                result = result.replace(
                    word, "أُو۟لَاْت" + random.choice(HARAKAT + SUKUN + TANWEEN))

    return result


def replace_hamza_wasl(text):
    pattern = fr'(?:(?:^)|(?<=\s|[{"".join(HARAKAT)}])){ILLA[0]}(?=.{SUKUN[0]})'
    pattern_al = r'(?:(?:^)|(?<=\s))(?:ال)'
    pattern_al_2 = fr'[{"".join(HARAKAT)}](?:ال)(?=[{"".join(ALL_CHARS[:-1])}]{SHADDA_CHAR[0]})'

    result = re.sub(pattern, HAMZA_WASL[0], text)
    result = re.sub(pattern_al, HAMZA_WASL[0] + 'ل', result)
    result = re.sub(pattern_al_2, HAMZA_WASL[0] + 'ل', result)
    return result


def replace_hamza_kasra(text):
    text = text.replace('إْ', 'إِ')
    pattern = fr'إ(?![{"".join(HARAKAT)}])'
    result = re.sub(pattern, 'إِ', text)
    return result


def replace_alif_hamza(text):
    pattern = fr'[{"".join(ALIF_HAMZA)}]'
    result = re.sub(pattern, "ا", text)
    return result


def strip_all_tashkeel(text, replace_hamza=False):
    pattern = fr'[{"".join(TASHKEEL)}]'
    result = re.sub(pattern, '', text)
    if replace_hamza:
        result = replace_hamza_wasl(result)
    return result


def some_tashkeel(text, p=0.2):
    # Find all the Tashkeel in the text

    # remove rounded zeros from text and replace hamza wasl ALWAYS
    text = re.sub(ROUNDED_ZERO[0], '', text)
    text = re.sub(HAMZA_WASL[0], 'ا', text)

    # random 0 or 1 remove all sukun
    if random.randint(0, 1):
        text = re.sub(SUKUN[0], '', text)

    # get a geometric distribution of the number of Tashkeel to remove
    tashkeel_positions = [(m.start(), m.group())
                          for m in re.finditer(f'[{"".join(TASHKEEL)}]', text)]

    # get a geometric distribution of the number of Tashkeel to remove
    num_to_keep = np.random.geometric(p) - 1

    # Ensure that the number of Tashkeel to keep is less than the number of Tashkeel in the text
    num_to_keep = min(num_to_keep, len(tashkeel_positions))

    num_to_remove = len(tashkeel_positions) - num_to_keep

    # Randomly select which Tashkeel to remove
    positions_to_remove = random.sample(tashkeel_positions, num_to_remove)

    # Remove the selected Tashkeel from the text
    for pos, tashkeel in sorted(positions_to_remove, reverse=True):
        text = text[:pos] + text[pos+1:]

    return text


# def some_tashkeel_2(text, p=0.5):
#     # Find all the Tashkeel in the text
#     tashkeel_positions = [(m.start(), m.group())
#                           for m in re.finditer(f'[{"".join(TASHKEEL)}]', text)]

#     # get a random number of Tashkeel to keep based on a bell curve probability distribution with a mean of p
#     num_to_keep = np.random.binomial(len(tashkeel_positions), p)

#     # # Ensure that the number of Tashkeel to keep is less than the number of Tashkeel in the text
#     # num_to_keep = min(num_to_keep, len(tashkeel_positions))

#     num_to_remove = len(tashkeel_positions) - num_to_keep

#     # Randomly select which Tashkeel to remove
#     positions_to_remove = random.sample(tashkeel_positions, num_to_remove)

#     # Remove the selected Tashkeel from the text
#     for pos, tashkeel in sorted(positions_to_remove, reverse=True):
#         text = text[:pos] + text[pos+1:]

#     return text


def extract_tashkeel(text, case_endings=True, tatweel=False):

    if not case_endings:
        # look behind a space for a tashkeel character and remove it
        text = re.sub(
            fr'([^{"".join(ALL_CHARS[:-1])}])(?:(?:$)|(?=\s))', '', text)

    # Create a regex pattern that matches any Tashkeel character
    pattern = fr'([{"".join(ALL_CHARS[:-1])}])'

    replacement = ' ـ' if tatweel else ' '
    tashkeel_sequence = re.sub(pattern, replacement, text)

    pattern = fr'({SHADDA_CHAR[0]})([{"".join(TASHKEEL)}])'
    tashkeel_sequence = re.sub(
        pattern, r'\1' + replacement + r'\2', tashkeel_sequence)

    result = ''.join(tashkeel_sequence)

    # remove double spaces or a single space at the begining or the end of the string
    result = re.sub(r'\s+', ' ', result).strip()

    # Join the extracted Tashkeel characters into a single string
    return result


def extract_tashkeel_2(text):

    # Create a regex pattern that matches any Tashkeel character
    pattern = fr'[{"".join(TASHKEEL + [" "])}]'

    # Use re.findall to extract all Tashkeel characters in the order they appear
    tashkeel_sequence = re.findall(pattern, text)

    tashkeel_with_tatweel = ''.join(
        f'ـ{t}' if t != " " else " " for t in tashkeel_sequence)

    # Join the extracted Tashkeel characters into a single string
    return ''.join(tashkeel_with_tatweel)


def remove_non_arabic_chars(text):
    pattern = f'[^{"".join(ALL_ARABIC_CHARS)}]'
    result = re.sub(pattern, '', text)
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def fix_priority_errors(text):
    tanween_after_alif_pattern = re.compile(f"([اى])({TANWEEN[0]})")
    harakah_after_shaddah_pattern = re.compile(
        f"([{''.join(HARAKAT + TANWEEN)}])({SHADDA_CHAR[0]})")
    text = tanween_after_alif_pattern.sub(
        lambda m: f"{m.group(2)}{m.group(1)}", text)
    text = harakah_after_shaddah_pattern.sub(
        lambda m: f"{m.group(2)}{m.group(1)}", text)
    return text


def add_harakah_before_madd(text):
    pattern = fr'(?<=[{"".join(ALL_CHARS[:-1] + SHADDA_CHAR)}])({ILLA[0]})(?:(?=$)|(?=[{"".join(ALL_CHARS[:-2]+SUKUN) + " "}])|(?=ل[{"".join(HARAKAT)}]))'
    result = re.sub(pattern, HARAKAT[0] + ILLA[0] + SUKUN[0], text)
    pattern = fr'(?<=[{"".join(ALL_CHARS[1:-1] + SHADDA_CHAR)}])({ILLA[1]})(?:(?=$)|(?=[{"".join(ALL_CHARS[1:]+SUKUN)}]))'
    result = re.sub(pattern, HARAKAT[1] + ILLA[1] + SUKUN[0], result)
    pattern = fr'(?<=[{"".join(ALL_CHARS[1:-1] + SHADDA_CHAR)}])({ILLA[2]})(?:(?=$)|(?=[{"".join(ALL_CHARS[1:]+SUKUN)}]))'
    result = re.sub(pattern, HARAKAT[2] + ILLA[2] + SUKUN[0], result)
    pattern = fr'(?<=[{"".join(ALL_CHARS[:-1] + SHADDA_CHAR)}])({ILLA[3]})(?:(?=$)|(?=[{"".join(HARAKAT) + " "}]))'
    result = re.sub(pattern, HARAKAT[0] + ILLA[3] + SUKUN[0], result)
    return result


def add_madd_silah(text):
    pattern = fr'(?<=[{"".join(HARAKAT)}])(ه)({HARAKAT[0]}) '
    result = re.sub(pattern, r"\1\2" + ILLA[0] + SUKUN[0] + " ", text)

    pattern = fr'(?<=[{"".join(HARAKAT)}])(ه)({HARAKAT[1]}) '
    result = re.sub(pattern, r"\1\2" + ILLA[1] + SUKUN[0] + " ", result)

    pattern = fr'(?<=[{"".join(HARAKAT)}])(ه)({HARAKAT[2]}) '
    result = re.sub(pattern, r"\1\2" + ILLA[2] + SUKUN[0] + " ", result)

    return result


def add_ishbaa_hm(text):
    if random.randint(0, 1):
        text = re.sub(r'هُمْ( ?)', r'هُمُوْ\1', text)
        text = re.sub(r'كُمْ( ?)', r'كُمُوْ\1', text)
    return text


def process_hamza_wasl(text):
    # Case 0: hamza wasl followed by a sun letter followed by a sukun, remove lam
    pattern = fr'{HAMZA_WASL[0]}لْ([{"".join(SUN_LETTERS)}]{SUKUN[0]})'
    text = re.sub(pattern, HAMZA_WASL[0] + r'\1', text)

    # Case 1: hamza wasl at the beginning of the text, replace it with alif
    if text.startswith(HAMZA_WASL[0]):
        text = "أ" + HARAKAT[0] + text[1:]

    # Case 2: if any HARAKAT followed by zero or one space followed by a hamza wasl, remove hamza wasl
    pattern = fr'([{"".join(HARAKAT)}])( ?){HAMZA_WASL[0]}'
    text = re.sub(pattern, r'\1\2', text)

    # Case 3: if HARAKAT is followed by a Madd followed by a sukun then a space and wasl, remove ILLA, SUKUN and Hamza Wasl
    patterns = [
        f"{HARAKAT[0]}{ILLA[0]}{SUKUN[0]}",
        f"{HARAKAT[1]}{ILLA[1]}{SUKUN[0]}",
        f"{HARAKAT[2]}{ILLA[2]}{SUKUN[0]}",
        f"{HARAKAT[0]}{ILLA[3]}{SUKUN[0]}"
    ]

    # Combine patterns with space and HAMZA_WASL[0]
    combined_patterns = [f"{pattern}" + " " + HAMZA_WASL[0]
                         for pattern in patterns]

    # Create a regex pattern to match any of the combined patterns
    regex_pattern = '|'.join(combined_patterns)

    # Find all matches
    matches = list(re.finditer(regex_pattern, text))

    # Remove the matched characters
    for match in reversed(matches):
        start, end = match.span()
        text = text[:start+1] + " " + text[end:]

    # Case 4: SUKUN followed by a space followed by a hamza wasl, remove the SUKUN and the hamza wasl and put a HARAKAT[0] instead before the space
    pattern = fr'{SUKUN[0]} {HAMZA_WASL[0]}'
    text = re.sub(pattern, HARAKAT[0] + " ", text)

    pattern = fr'{SUKUN[0]}{HAMZA_WASL[0]}'
    # replace with fatha just to get the correct beat, but it may be any harakah
    text = re.sub(pattern, HARAKAT[0], text)

    return text


def process_line(line):
    line = remove_non_arabic_chars(line)
    line = fix_priority_errors(line)
    line = replace_hamza_wasl(line)
    line = replace_hamza_kasra(line)
    line = id_unvoc_harfs(line)
    # line = remove_unvocalized_harfs(line)
    line = add_harakah_before_madd(line)
    line = add_missing_sukuns(line)
    # line = expand_shaddah(line)
    untash = strip_all_tashkeel(line)
    return line, untash


def convert_to_phones(line):
    line = remove_non_arabic_chars(line)
    line = replace_special_ziyadah(line)
    line = remove_rounded_zero(line)
    line = expand_tanween_2(line)
    line = expand_shaddah(line)
    line = replace_maddah(line)
    line = add_madd_silah(line)
    line = add_ishbaa_hm(line)
    line = process_hamza_wasl(line)
    return line


def convert_to_beat(line):
    # check chars, if it is followed by a harakat, put 1 if it is followed by a sukun put 0
    result = []
    i = 0
    while i < len(line):
        if is_mutaharik(line, i):
            result.append("1")
        elif is_sakin(line, i):
            result.append("0")
        elif line[i] == " ":
            result.append(",")
        i += 1
    return ''.join(result)


if __name__ == "__main__":
    # open a csv file using pandas and convert column "processed" to beat and write to a new column and save the file
    # line = "وَلَوْ أُنْثَىْ فَلَهُ جَبْرُهُمَاْ إِنْ لَمْ يَمْرَضْ ٱلْسَّيِّدُ مَرَضًا۟ مَخُوْفًا۟"
    # processed = convert_to_phones(line)
    # print(processed)
    from tqdm import tqdm
    tqdm.pandas()
    pass
    df = pd.read_csv("/datasets/new/tash_train.csv")
    df["phones"] = df["lines"].progress_apply(convert_to_phones)
    df["binary"] = df["phones"].progress_apply(convert_to_beat)
    df.to_csv(
        "/datasets/new/proc_tash_train.csv", index=False)
