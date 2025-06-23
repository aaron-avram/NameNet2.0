"""
File containing functionality for parsing a txt file
"""

def parse_txt(file: str):
    """
    Parse text file into a list
    """
    f = open(file=file, mode='r', encoding='UTF8')
    out = [s[:-1] for s in f.readlines()]
    f.close()
    return out
