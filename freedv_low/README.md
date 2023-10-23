# WP4000 FreeDV low SNR Study

| File | Description |
| ---- | ---- |
| freedv_low.tex | Latex report source |
| acquisition.m | Octave simulation code used to develop and test algorithms |
| equaliser.m | Octave simulation code used to develop and test equalisation and multipath algorithms |
| ofdm_lib.m | Library of shared OFDM functions |
| load_const.m | #define like constants file shared by simulations |
| doppler_spread.m | Generates Doppler spreading samples |
| ofdm_c.sh | Generate BER and PER curve data for comparison from existing OFDM C implementations |

1. Building Report
   ```
   $ sudo apt install texmaker texlive-bibtex-extra texlive-science
   $ texmaker your_doc.tex &
   ```

   1. Options -> Config Texmaker -> Quick Build -> select 2nd option: "PDFLatex+BibLatex+PDFLatex2+ViewPDF"
   1. F1 to build document.

1. Running equaliser sim on a single Eb/No point (AWGN channel, Eb/No = 4 dB), lots of plots
   ```
   octave:369> equaliser; run_single(10000,'awgn',EbNodB=4);
   ```
   MPP channel, Eb/No = 7 dB:
   ```
   octave:371> equaliser; run_single(10000,'mpp',7)
   ```
   More examples in `equaliser.m`.

1. Running acquisition sim spot checks (good way to check code is OK)
   ```
   octave:362> acquisition; spot_checks
   ```

1. Run acq sim on a single point:
   ```
   acquisition; run_single(1E4,'awgn',10,'norand')
   ```
   `norand` switches off randomisation of time and freq and makes Dtmax point on RHS of Fig 2 clearer

1. Testing time between false detections on noise only:
   ```
   acquisition; run_single(1E4,'awgn',-100,'Pthresh',3E-5')
   ```
   Compare `Tdet/Tnoise` to `Tnoise (theory)`.  Setting Eb/No =-100dB ensures the signal is swamped by noise.
   
