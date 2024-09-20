from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import xml.etree.ElementTree as ET
import teanga
import glob
import tqdm

detokenizer = TreebankWordDetokenizer()
detokenizer.ENDING_QUOTES = [
        (re.compile(r"([^' ])\s('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1\2 "),
        (re.compile(r"([^' ])\s('[sS]|'[mM]|'[dD]|') "), r"\1\2 "),
        (re.compile(r"([^'\s])\s(\'\')"), r"\1\2"),
        (
            re.compile(r"(\'\')\s([.,:)\]>};%])"),
            r"\1\2",
        ),  # Quotes followed by no-left-padded punctuations.
        (re.compile(r"''"), '"'),
    ]

def find_spans(text, tokens, offset=0):
    spans = []
    start = 0
    last_start = 0
    for token in tokens:
        if token == '``':
            token = "\""
        elif token == "''":
            token = "\""
        elif token == ". ...":
            token = "...."
        elif token.startswith("( ") and token.endswith(" )"):
            token = "(" + token[2:-2] + ")"
        else:
            if "''" in token:
                token = token.replace("''", "\"")
            if "``" in token:
                token = token.replace("``", "\"")
        last_start = start
        start = text.find(token, start)
        if start == -1:
            raise ValueError("Token not found in text: " + token + 
                             " in " + text[last_start:last_start+50])
        end = start + len(token)
        spans.append((start + offset, end + offset))
        start = end
    return spans



def convert_ufsac_file(file_path, mode, has_document_id=False):
    print(file_path)
    tree = ET.parse(file_path)

    root = tree.getroot()

    corpus = teanga.Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("tokens", layer_type="span", base="text")
    corpus.add_layer_meta("lemmas", layer_type="seq", base="tokens", 
                          data="string")
    corpus.add_layer_meta("pos", layer_type="seq", base="tokens", 
                          data="string")
    corpus.add_layer_meta("wn16_key", layer_type="element", base="tokens", 
                          data="string")
    corpus.add_layer_meta("wn30_key", layer_type="element", base="tokens",
                          data="string")
    if mode == "doc" or mode == "para":
        corpus.add_layer_meta("sentence", layer_type="div", base="text")
    if mode == "doc":
        corpus.add_layer_meta("paragraph", layer_type="div", base="text")
    if has_document_id:
        corpus.add_layer_meat("document_id")

    n_paragraphs = 0
    n_sentences = 0
    n_tokens = 0

    for document in tqdm.tqdm(root.findall('document')):
        if has_document_id:
            document_id = document.attrib['id']
        i = 0
        offset = 0
        token_spans = []
        lemmas = []
        pos = []
        wn16_key = []
        wn30_key = []
        sentences = []
        paragraphs = []
        text = ""

        for paragraph in document.findall('paragraph'):
            n_paragraphs += 1

            paragraphs.append(offset)
            for sentence in paragraph.findall('sentence'):
                n_sentences += 1
                sentences.append(offset)
                tokens = []
                for token in sentence.findall('word'):
                    n_tokens += 1
                    tokens.append(token.attrib['surface_form'])
                    lemmas.append(token.attrib.get('lemma', 
                                                   token.attrib['surface_form']))
                    pos.append([i, token.attrib['pos']])
                    if 'wn16_key' in token.attrib:
                        wn16_key.append([i, token.attrib['wn16_key']])
                    if 'wn30_key' in token.attrib:
                        wn30_key.append([i, token.attrib['wn30_key']])
                    i += 1
                if text:
                    text += " "
                    offset += 1
                text += detokenizer.detokenize(tokens)
                token_spans.extend(find_spans(text[offset:], tokens, offset))
                offset = len(text)
                if mode == "sent":
                    document = corpus.add_doc(text)
                    document.tokens = token_spans
                    document.lemmas = lemmas
                    document.pos = pos
                    document.wn16_key = wn16_key
                    document.wn30_key = wn30_key
                    if has_document_id:
                        document.document_id = document_id
                    text = ""
                    token_spans = []
                    lemmas = []
                    pos = []
                    wn16_key = []
                    wn30_key = []
                    sentences = []
                    paragraphs = []
                    offset = 0
            if mode == "para":
                document = corpus.add_doc(text)
                document.tokens = token_spans
                document.lemmas = lemmas
                document.pos = pos
                document.wn16_key = wn16_key
                document.wn30_key = wn30_key
                document.sentence = sentences
                if has_document_id:
                    document.document_id = document_id
                text = ""
                token_spans = []
                lemmas = []
                pos = []
                wn16_key = []
                wn30_key = []
                sentences = []
                paragraphs = []
            else:
                text += "\n"
                offset += 1
        
        if mode == "doc":
            document = corpus.add_doc(text)
            document.tokens = token_spans
            document.lemmas = lemmas
            document.pos = pos
            document.wn16_key = wn16_key
            document.wn30_key = wn30_key
            document.paragraph = paragraphs
            document.sentence = sentences
            if has_document_id:
                document.document_id = document_id

    print("Paragraphs:", n_paragraphs)
    print("Sentences:", n_sentences)
    print("Tokens:", n_tokens)

    with open(file_path[:-4] + ".yaml", "w") as f:
        corpus.to_yaml(f)

if __name__ == "__main__":
    convert_ufsac_file("ufsac-public-2.1/masc.xml", "sent")
    #convert_ufsac_file("ufsac-public-2.1/omsti.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/raganato_ALL.xml", "para")
    convert_ufsac_file("ufsac-public-2.1/raganato_semeval2007.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/raganato_semeval2013.xml", "para")
    convert_ufsac_file("ufsac-public-2.1/raganato_semeval2015.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/raganato_senseval2.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/raganato_senseval3.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/semcor.xml", "doc")
    convert_ufsac_file("ufsac-public-2.1/semeval2007task17.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/semeval2007task7.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/semeval2013task12.xml", "para")
    convert_ufsac_file("ufsac-public-2.1/semeval2015task13.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/senseval2_lexical_sample_test.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/senseval2_lexical_sample_train.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/senseval2.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/senseval3task1.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/senseval3task6_test.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/senseval3task6_train.xml", "sent")
    #convert_ufsac_file("ufsac-public-2.1/trainomatic.xml", "sent")
    convert_ufsac_file("ufsac-public-2.1/wngt.xml", "sent")


