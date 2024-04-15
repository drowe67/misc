# HF Modem Comparisons

In this doc the  term "codec2" refers to the codec 2 HF raw data waveforms (modes).  FreeDATA uses the codec2 waveforms are being used to implement a HF data system.

# VARA Comments

1. An OFDM modem like codec2. VARA uses Turbo codes which in terms of performance are equivalent to the LPDC codes used by codec2.
1. A VARA "speed level" is equivalent to a codec2 "mode" or "waveform"
1. The general frame design [2] is similar to codec2, several "modem frames" concatenated to make a "packet" (codec 2 terminology).  Each modem frame has is it's own sync/EQ symbols.
1. VARA uses a fixed duration data frame for all speed levels.  The number of symbols per frame is fixed, and the modulation (and perhaps the code rate) varies to move up and down the bit rate/SNR curve.
1. The length of codec2 packets is ad hoc - we design the packet length based largely on our suite of available LDPC codes, and SNR.
1. The fixed length DATA and ARQ frames is an interesting design choice.  This resembles a TDMA time slot based approach.  FreeDATA uses an asynchronous design.  In general synchronous designs are more stable with less bugs. There may be some advantages to fixed length frames, e.g. a simplified state machine with transitions based on time rather than say when a fwd link packet is received.
1. For VARA Ts=0.0024, Rs=42, Nc=52 (see glossary of [4])
1. The length of the Cyclic Prefix (CP) is a bit unclear, it's quoted as 3ms at the top of the document, 2.66ms in Section 5.1 of [1], but the figure is marked 5.33ms.  The figure has the CP in the wrong position (CP should be in front of the symbol).  Lets assume Tcp=0.00267, with Ts'=0.02133.
1. A 2.66ms CP is not very long, and may have problems with high delay spread paths.  However it means a low overhead: Lcp=0.52dB.  In some codec2 modes Tcp is a much greater proportion of Ts, in order to balance fast fading performance and high delay spread operation.
1. Use of DPSK at low SNRs is interesting, codec2 using coherent PSK at all PSKs.  Quite a big drop in performance with DPSK over PSK.  The codec2 approach is to reduce the number of carriers as SNR drops, and keep Eb/No about the same.  This keeps the energy for the pilots about constant too, so we can enjoy coherent PSK.
1. VARA has separate pilot symbols for synchronisation and EQ. The EQ pilot distribution is interesting. Stepping them in frequency is a good idea, avoids any persistent notches.
1. It is possible to est the channel for all carriers from pilots in a subset of the carriers (e.g. the least squares approach in [4]). So the frequency response of the VARA EQ is the same as the symbols rate - it may be capable of handling quite fast fading.
1. In contrast codec2 uses the same pilots for sync and EQ, and they are only present at the start and end of each modem frame (like the DPSK frame design in VARA).  Thus the time between pilots (Ns) sets the freq response to fast fading channels, and means we need a small Ns (and relatively high Lp). The codec2 modes use a small number of symbols for the EQ of each carrier on or adjacent to the current carrier frequency (hard to say if this is a pro or con).
1. Nc=52, Np=11 (sync), 5 pilots/row, so Np=10 (EQ). 52*11=572 symbols in a frame. Nsync=52+50=102. So I think Lp = 10*log10((572-102)/572)=0.85dB.
1. codec2 has two additional overheads (a) a unique word to make synchronisation more reliable, and (b) pre-and post amble(an additional overhead) for reliable acquisition.  It is unclear if VARA has a pre-amble/unique word.
1. Both VARA and codec2 reserve a few bytes for a checksum.
1. Hard to say which approach works better. Some simulations and perhaps tests would be needed to compare he merits of the two approaches.  Key is difference (in dB) from ideal results, and fast fading/high delay-spread performance (does it break down).
1. The ACK frame is a dual carrier FSK signal (so two FSK signals sent at the same time).  This is uncommon, FSK is often chosen as it's PAPR is 0dB.  Using two parallel FSK carriers means a 3dB RMS power hit, which the author acknowledges.
1. The data frames have a PAPR of 6dB, so 3dB lower power than the ACK frames.  This is an interesting way of making the ACK frame more robust, although what really counts is the PER over various channel types and speed levels, which is set by Eb/No rather than RMS Power.
1. The VARA author uses a different standard for measuring PAPR, probably the peak/average of the real signal rather than complex/bandpass/PEP power.  However this can be compared with the codec2 PAPR metric by reference to the RMS power values.  Using the codec2 method for measuring PAPR, codec2 waveforms have a PAPR of around 4dB, VARA data 6dB, VARA ACK 3dB.
1. The 5 different ACK packet types are interesting, the requirement for not clashing with another sessions ACK is mentioned.
1. The VARA modem has at +/-50 Hz acquisition range, and can handle drift of 0.5 Hz/s, similar to codec2.

1. Waveform overheads (using codec2/datac1 as an example, ignoring preamble and UW overheads):

   |       | VARA  | datac1 |
   | ----- | ----- | -----  |   
   | Lp    |  0.85 |  1.38  |
   | Lcp   |  0.52 |  0.97  |
   | Total |  1.37 |  2.35  |

   Which suggests VARA is 1dB more efficicent. Added to this is the implementation loss, e.g. the difference from ideal AWGN/MP performance due to non-ideal synchronisation/equalisation (measured on AWGN and multipath channels such as MPP, MPD).  A great deal of DSP performance is also due to "tuning" - basically the skill and experience of the author.  For example two different implementations of the same paper "standard" may result in very different results.

# References

1. VARA Specification.pdf, Revision 2.0.0 April 5, 2018
2. VaraHF v4.0 Levels.pdf
3. VARA_HF_Modem.ppt
4. misc/freedv_low.pdf, OFDM modem analysis, glossary

# TODO

1. Find up to date VARA docs, files in this repo are labelled 2018
1. Find or measure VARA PER (or throughput) versus SNR for various speed levels on AWGN/MPP,MPD channels.  Unfortunately throughput tests measure combined waveform and protocol performance.
1. Measure PAPR of VARA signals.

