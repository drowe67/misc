Some small projects on miscellaneous topics.

| Date | Project | Description |
| --- | --- | --- |
| Feb 2020 | covid19 | Generating my own custom plots using Python to interpret COVID-19, see http://www.rowetel.com/?p=7016 |
| Jan 2022 | emimap | Mapping Electromagnetic Interference (EMI) around Adelaide |
| April 2022 | ah_timer | Timing lithium cell discharge to estimate AH capacity |
| Mar 2023 | ldpc_even | Proving an expression used in LDPC decoders to determine the probability that a sequence of bits has an even number of ones |
| April 2023 | gt_sun | Derivation of the expression for estimating receiver G/T from power measurements obtained by pointing an antenna at the sun and quiet sky |
| April 2023 | ratek_resampler | A study of the codec2 700C Rate K resampler |
| May 2023 | rect_ft | Fourier Transforms of rectangular functions |
| May 2023 | emi | HF EMI study: Qualitative discussion of EMI on lower HF bands, expressions for PWM noise and noise blanking, experimental EMI measurements |
| July 2023 | freedv_low | Low SNR FreeDV waveform study |
| Sep 2023 | folded_balun | Folded balun (Pawsey Stub) |
| Oct 2023 | manifold | Experiments with non-linear dependencies with Codec 2 spectral magnitude vectors |
| Jan 2024 | mlquant | ML Quantisation of Vocoder Features |
| Feb 2024 | radio_ae | Autoencoder for transmission of vocoder features over radio channels |
| April 2024 | hf_modems | Comparisoon of VARA to codec2 raw data waveforms |

I build the Latex documents using Texmaker:

```
$ sudo apt install texmaker texlive-bibtex-extra texlive-science
$ texmaker your_doc.tex &
```

1. Options -> Config Texmaker -> Quick Build -> select 2nd option: "PDFLatex+BibLatex+PDFLatex2+ViewPDF"
1. F1 to build document.
