MAIN = slides
LATEXMK = latexmk
FLAGS = -xelatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all:
	$(LATEXMK) $(FLAGS) $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f *.nav *.snm *.vrb
