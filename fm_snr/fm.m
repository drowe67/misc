% fm.m
% David Rowe Dec 2014/2025
%
% Analog FM Octave simulation.
%
% usage:
%
%   octave:16> fm; run_fm_curves("snr_cnr")
%   octave:17> fm; run_fm_measured("snr_rx")

1;

function fm_states = analog_fm_init(fm_states)
  pkg load signal;

  % FM modulator constants

  Fs = fm_states.Fs; FsOn2 = Fs/2;  
  fm_max = fm_states.fm_max;                 % max modulation freq
  fd = fm_states.fd;                         % (max) deviation
  fm_states.m = fd/fm_max;                   % modulation index
  fm_states.Bfm = Bfm = 2*(fd+fm_max);       % Carson's rule for FM signal bandwidth
  fm_states.tc = tc = 50E-6;
  fm_states.prede = [1 -(1 - 1/(tc*Fs))];    % pre/de emp filter coeffs
  fm_states.ph_dont_limit = 0;               % Limit rx delta-phase
  
  % Select length of filter to be an integer number of symbols to
  % assist with "fine" timing offset estimation.  Set Ts to 1 for
  % analog modulation.

  Ts = fm_states.Ts;
  desired_ncoeffs = 200;
  ncoeffs = floor(desired_ncoeffs/Ts+1)*Ts;

  % "coarse" timing offset is half filter length, we have two filters.
  % This is the delay the two filters introduce, so we need to adjust
  % for this when comparing tx to trx bits for BER calcs.

  fm_states.nsym_delay = ncoeffs/Ts;

  % input filter gets rid of excess noise before demodulator, as too much
  % noise causes atan2() to jump around, e.g. -pi to pi.  However this
  % filter can cause harmonic distortion at very high SNRs, as it knocks out
  % some of the FM signal spectra.  This filter isn't really required for high
  % SNRs > 20dB.

  fc = (Bfm/2)/(FsOn2);
  fm_states.bin  = firls(ncoeffs,[0 fc*(1-0.05) fc*(1+0.05) 1],[1 1 0.01 0.01]);

  % demoduator output filter to limit us to fm_max (e.g. 3kHz)

  fc = fm_max/(FsOn2);
  fm_states.bout = firls(ncoeffs,[0 0.95*fc 1.05*fc 1], [1 1 0.01 0.01]);
endfunction


function tx = analog_fm_mod(fm_states, mod)
  Fs = fm_states.Fs;
  fc = fm_states.fc; wc = 2*pi*fc/Fs;
  fd = fm_states.fd; wd = 2*pi*fd/Fs;
  nsam = length(mod);

  if fm_states.pre_emp
    mod = filter(fm_states.prede,1,mod);
    mod = mod/max(mod);           % AGC to set deviation
  end

  tx_phase = 0;
  tx = zeros(1,nsam);

  for i=0:nsam-1
    w = wc + wd*mod(i+1);
    tx_phase = tx_phase + w;
    tx_phase = tx_phase - floor(tx_phase/(2*pi))*2*pi;
    tx(i+1) = exp(j*tx_phase);
  end
endfunction


function [rx_out rx_bb] = analog_fm_demod(fm_states, rx)
  Fs = fm_states.Fs;
  fc = fm_states.fc; wc = 2*pi*fc/Fs;
  fd = fm_states.fd; wd = 2*pi*fd/Fs;
  nsam = length(rx);
  t = 0:(nsam-1);

  rx_bb = rx .* exp(-j*wc*t);      % down to complex baseband
  rx_bb = filter(fm_states.bin,1,rx_bb);

  % differentiate first, in rect domain, then find angle, this puts
  % signal on the positive side of the real axis

  rx_bb_diff = [ 1 rx_bb(2:nsam) .* conj(rx_bb(1:nsam-1))];
  rx_out = atan2(imag(rx_bb_diff),real(rx_bb_diff));

  % limit maximum phase jumps, to remove static type noise at low SNRs
  if !fm_states.ph_dont_limit
    rx_out(find(rx_out > wd)) = wd;
    rx_out(find(rx_out < -wd)) = -wd;
  end
  rx_out *= (1/wd);

  if fm_states.output_filter
    rx_out = filter(fm_states.bout,1,rx_out);
  end
  if fm_states.de_emp
    rx_out = filter(1,fm_states.prede,rx_out);
  end
endfunction


function sim_out = analog_fm_test(sim_in)
  nsam      = sim_in.nsam;
  CNdB      = sim_in.CNdB;
  verbose   = sim_in.verbose;

  Fs = fm_states.Fs = 96000;  
  fm_max = sim_out.fm_max = fm_states.fm_max = 3E3;
  fd = fm_states.fd = 2.5E3;
  fm_states.fc = 24E3;

  fm_states.pre_emp = pre_emp = sim_in.pre_emp;
  fm_states.de_emp  = de_emp = sim_in.de_emp;
  fm_states.Ts = 1;
  fm_states.output_filter = 1;
  fm_states = analog_fm_init(fm_states);
  sim_out.Bfm = fm_states.Bfm;

  Bfm = fm_states.Bfm;
  m = fm_states.m; tc = fm_states.tc;
  t = 0:(nsam-1);
 
  fm = 1000; wm = 2*pi*fm/fm_states.Fs;
  
  % Theoretical from Carson
  A = 1;
  Gfm = 3*(A^2/2)*m^2;
  Gfm_dB = 10*log10(Gfm);
  
  printf("fd: %5.2f fm: %5.2f Beta: %5.2f A: %3.2f Gfm_dB: %5.2f\n", fd, fm_max, m, A, Gfm_dB);

  % start simulation

  for ne = 1:length(CNdB)

    % work out the variance we need to obtain our C/N in the bandwidth
    % of the message signal (equivalent to SSB). The gaussian generator 
    % randn() generates noise with a bandwidth of Fs

    aCNdB = CNdB(ne);
    CN = 10^(aCNdB/10);
    variance = Fs/(CN*fm_max);
     
    % FM Modulator -------------------------------

    mod = A*sin(wm*t);
    tx = analog_fm_mod(fm_states, mod);
    
    % Channel ---------------------------------

    noise = sqrt(variance/2)*(randn(1,nsam) + j*randn(1,nsam));
    rx = tx + noise;

    % FM Demodulator

    [rx_out rx_bb] = analog_fm_demod(fm_states, rx);

    % notch out test tone

    w = 2*pi*fm/Fs; beta = 0.99;
    rx_notch = filter([1 -2*cos(w) 1],[1 -2*beta*cos(w) beta*beta], rx_out);

    % measure power with and without test tone to determine S+N and N

    settle = 1000;             % filter settling time, to avoid transients
    nsettle = nsam - settle;

    sinad = (rx_out(settle:nsam) * rx_out(settle:nsam)')/nsettle;
    nad = (rx_notch(settle:nsam) * rx_notch(settle:nsam)')/nsettle;

    snr = (sinad-nad)/nad;
    sim_out.snrdB(ne) = 10*log10(snr);
   
    snr_theory_dB = Gfm_dB + aCNdB;
    fx = 1/(2*pi*tc); W = fm_max;
    I = (W/fx)^3/(3*((W/fx) - atan(W/fx)));
    I_dB = 10*log10(I);

    sim_out.snr_theorydB(ne) = snr_theory_dB;
    sim_out.snr_theory_pre_dedB(ne) = snr_theory_dB + I_dB;
   
    if verbose > 1
      printf("modn index: %2.1f Bfm: %4.0f Hz\n", m, Bfm);
    end

    if verbose > 0
      printf("C/N: %4.1f SNR: %4.1f dB THEORY: %4.1f dB or with pre/de: %4.1f dB\n", 
      aCNdB, 10*log10(snr), snr_theory_dB, snr_theory_dB+I_dB);
    end

    if verbose > 1
      figure(1)
      subplot(211)
      plot(20*log10(abs(fft(rx))))
      title('FM Modulator Output Spectrum');
      axis([1 length(tx) 0 100]);
      subplot(212)
      Rx_bb = 20*log10(abs(fft(rx_bb)));
      plot(Rx_bb)
      axis([1 length(tx) 0 100]);
      title('FM Demodulator (baseband) Input Spectrum');

      figure(2)
      subplot(211)
      plot(rx_out(settle:nsam))
      axis([1 4000 -1 1]) 
      subplot(212)
      Rx = 20*log10(abs(fft(rx_out(settle:nsam))));
      plot(Rx(1:10000))
      axis([1 10000 0 100]);
   end

  end

endfunction


function run_fm_curves(epslatex="")
  sim_in.nsam    = 96000;
  sim_in.verbose = 1;
  sim_in.pre_emp = 0;
  sim_in.de_emp  = 0;
  sim_in.CNdB = 0:2:30;

  sim_out = analog_fm_test(sim_in);

  if length(epslatex)
    [textfontsize linewidth] = set_fonts(20);
  end

  figure(1); clf;
  plot(sim_in.CNdB, sim_out.snrdB,"r;FM Simulated;");
  hold on;
  plot(sim_in.CNdB, sim_out.snr_theorydB,"g;FM Theory;");
  plot(sim_in.CNdB, sim_in.CNdB,"b; SSB Theory;");
  hold off;
  grid("minor");
  xlabel("FM demod input CNR (dB)");
  ylabel("FM demod output SNR (dB)");
  legend("boxoff"); legend('location','southeast');
  axis([0 30 0 30]);
  if length(epslatex)
    print_eps_restore(epslatex,"-S300,200",textfontsize,linewidth);
  end

  % C/No curves

  Bfm_dB = 10*log10(sim_out.Bfm);
  Bssb_dB = 10*log10(3000);

  figure(2); clf;
  plot(sim_in.CNdB + Bfm_dB, sim_out.snrdB,"r;FM Simulated;");
  hold on;
  plot(sim_in.CNdB + Bfm_dB, sim_out.snr_theorydB,"g;FM Theory;");
  plot(sim_in.CNdB + Bssb_dB, sim_in.CNdB,"b; SSB Theory;");
  hold off;
  grid("minor");
  xlabel("FM demod input C/No (dB)");
  ylabel("FM demod output S/N (dB)");
  legend("boxoff");

endfunction


function run_fm_single
  sim_in.nsam    = 96000;
  sim_in.verbose = 2;
  sim_in.pre_emp = 0;
  sim_in.de_emp  = 0;

  sim_in.CNdB   = 20;
  sim_out = analog_fm_test(sim_in);
end

% Experimental results from Tibor Bece
function run_fm_measured(epslatex="")
  fd=2500; fm=3000; A=0.6;
  NF_dB=5;
  Rx_dBm = -127:-104;
  SNR_dB = [4.2 -2.2 0.3 2.0 4.2 5.9 7.2 8.6 9.4 10.6 11.5 12.5 13.6 15.2 16.1 17.1 18.2 19.2 20.0 20.9 22.0 22.8 23.8 24.7];
  assert(length(Rx_dBm) == length(SNR_dB))

  N_dBm = -174 + NF_dB + 10*log10(fm);
  CNR_dB = Rx_dBm - N_dBm;
  m = fd/fm;
  Gfm = 3*(A^2/2)*m^2;
  Gfm_dB = 10*log10(Gfm);
  SNR_theory_dB =  CNR_dB + Gfm_dB
  printf("fd: %5.2f fm: %5.2f Beta: %5.2f A: %3.2f Gfm_dB: %5.2f\n", fd, fm, m, A, Gfm_dB);
  if length(epslatex)
    [textfontsize linewidth] = set_fonts(20);
  end

  figure(1); clf;
  plot(Rx_dBm, SNR_dB,"r;FM Measured;");
  hold on;
  plot(Rx_dBm, SNR_theory_dB,"g;FM Theory;");
  hold off;
  grid("minor");
  xlabel("Rx input level (dBm)");
  ylabel("FM demod output SNR (dB)");
  legend("boxoff"); legend('location','southeast');
  axis([-127 -100 0 30]);
  if length(epslatex)
    print_eps_restore(epslatex,"-S300,200",textfontsize,linewidth);
  end

endfunction

function [textfontsize linewidth] = set_fonts(font_size=12)
  textfontsize = get(0,"defaulttextfontsize");
  linewidth = get(0,"defaultlinelinewidth");
  set(0, "defaulttextfontsize", font_size);
  set(0, "defaultaxesfontsize", font_size);
  set(0, "defaultlinelinewidth", 0.5);
end

function restore_fonts(textfontsize,linewidth)
  set(0, "defaulttextfontsize", textfontsize);
  set(0, "defaultaxesfontsize", textfontsize);
  set(0, "defaultlinelinewidth", linewidth);
end

function print_eps(fn,sz)
  print(fn,sz,"-depslatex");
  printf("printing... %s\n", fn);
end

function print_eps_restore(fn,sz,textfontsize,linewidth)
  print_eps(fn,sz);
  restore_fonts(textfontsize,linewidth);
end
