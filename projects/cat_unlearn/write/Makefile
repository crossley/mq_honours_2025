# Makefile to compile main.tex into a PDF

TEXFILE = main
PDF = $(TEXFILE).pdf

all: $(PDF)

$(PDF): $(TEXFILE).tex
	pdflatex $(TEXFILE).tex

clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk

.PHONY: all clean

