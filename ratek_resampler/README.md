# A study of the codec2 700C Rate K resampler

Originally in codec2-dev/doc/ratek_resampler dr-papr branch.  Source
code to run simulations cited in this study can by found in that branch.

# Basic setup:

```
$ sudo apt install texmaker texlive-bibtex-extra
$ texmaker ratek_resampler.tex &
```

1. Options -> Config Texmaker -> Quick Build -> select 2nd option: "PDFLatex+BibLatex+PDFLatex2+ViewPDF"
1. Open `ratek_resampler.tex`
1. F1 to build document.

# Latex/EPS figures from Octave:

Gotchas:
1. Watch out for any control characters like & or _ in figure text, this will cause the build step to choke.
1. In `/path/to/ratek_resampler` Optionally clean out any older .eps files of same name (e.g. previously rendered with encapsulated Postscript).
1. In texmaker, build PDF from `ratek_resampler.tex`, not the figure `.tex` file (e.g. testepslatex.tex), you might end up in the latter if an error was encountered.

For nice rendering on your monitor, there may be some non-standard font and line size defaults set up in `.octaverc`, e.g.:
```
set(0, "defaulttextfontsize", 24)  % title
set(0, "defaultaxesfontsize", 24)  % axes labels
set(0, "defaultlinelinewidth", 2)
```
For Latex/EPS plots, save these defaults, and set up fontsize 10, line size 0.5 before plotting:

```
textfontsize = get(0,"defaulttextfontsize");
linewidth = get(0,"defaultlinelinewidth");
set(0, "defaulttextfontsize", 10);
set(0, "defaultaxesfontsize", 10);
set(0, "defaultlinelinewidth", 0.5);
```

Render your figure then print in EPS Latex using:

```
octave:1> print("testepslatex","-depslatex","-S300,300");
```

After plotting, the defaults can be restored:

```
set(0, "defaulttextfontsize", textfontsize);
set(0, "defaultaxesfontsize", textfontsize);
set(0, "defaultlinelinewidth", linewidth);
```

Latex code to insert figure:
```
\begin{figure}[h]
\caption{Test Octave -depslatex}
\begin{center}
\input testepslatex.tex
\end{center}
\end{figure}
```