PDF=main.pdf
TEX=main.tex

.PHONY: pdf clean

pdf:
	@if [ ! -f neurips_2025.sty ]; then \
		echo "Missing neurips_2025.sty. Download https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip and place neurips_2025.sty in report/."; \
		exit 1; \
	fi
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode -halt-on-error $(TEX); \
	else \
		if ! command -v pdflatex >/dev/null 2>&1; then \
			echo "Missing LaTeX compiler. Install latexmk or pdflatex, then rerun make pdf."; \
			exit 1; \
		fi; \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
		bibtex main || true; \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
		pdflatex -interaction=nonstopmode -halt-on-error $(TEX); \
	fi

clean:
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -C; \
	fi
	rm -f *.aux *.bbl *.blg *.fdb_latexmk *.fls *.log *.out *.toc *.synctex.gz $(PDF)
