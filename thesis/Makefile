MAIN   = main
LATEX  = xelatex
BIBTEX = bibtex
FLAGS  = -interaction=nonstopmode -halt-on-error

.PHONY: all clean distclean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex chapters/*.tex references.bib thesis_refs.bib
	$(LATEX) $(FLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	       $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	       $(MAIN).fls $(MAIN).fdb_latexmk
	rm -f chapters/*.aux

distclean: clean
	rm -f $(MAIN).pdf
