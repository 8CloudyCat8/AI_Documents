from rutermextract import TermExtractor
term_extractor = TermExtractor()
text = "Текст для разбора"
for term in term_extractor(text):
    print(term.normalized, term.count)