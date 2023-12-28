% ml_fbf.m
%
% David Rowe 2023
%
% Frame by Frame stepping for ML Experiment, generating plots for report
%
% Usage:
%   Make sure codec2-dev is compiled with the -DDUMP option - see README.md for
%    instructions.
%   $ ./ratek_resampler.sh ml_test_231126
%   (generates big_dog model & dump files, "big_dog_b.f32", big_dog_y.f32", "big_dog_y_hat.f32")
%   octave:14> ratek3_fbf("big_dog",61)

function ml_fbf(samname, f)
  more off;

  Fs = 8000; resampler = 'spline'; Lhigh = 80; F0high = (Fs/2)/Lhigh;
  K=20; Kst = 0; Ken = K-1;
  epslatex = 0; N=320;

  % load up text files dumped from c2sim ---------------------------------------

  sn_name = strcat(samname,"_sn.txt");
  Sn = load(sn_name);
  sw_name = strcat(samname,"_sw.txt");
  Sw = load(sw_name);
  model_name = strcat(samname,"_model.txt");
  model = load(model_name);
  [frames tmp] = size(model);
  phase_name = strcat(samname,"_phase.txt");
  if (file_in_path(".",phase_name))
    phase = unwrap(load(phase_name),pi,2);
  endif

  rate_K_sample_freqs_kHz = mel_sample_freqs_kHz(K);

  b = load_f32(strcat(samname,"_b.f32"),K+2);
  y = load_f32(strcat(samname,"_y.f32"),Lhigh-1);
  y_hat = load_f32(strcat(samname,"_y_hat.f32"),Lhigh-1);

  rate_Lhigh_sample_freqs_kHz = (F0high:F0high:(Lhigh-1)*F0high)/1000;
  if epslatex, [textfontsize linewidth] = set_fonts(); end

  % Keyboard loop --------------------------------------------------------------

  k = ' '; 
  do
    if epslatex, [textfontsize linewidth] = set_fonts(); end

    s = [ Sn(2*f-1,:) Sn(2*f,:) ];
    figure(1); clf; plot(s); axis([1 length(s) -20000 20000]);

    Wo = model(f,1); F0 = Fs*Wo/(2*pi); L = model(f,2);
    Am = model(f,3:(L+2)); AmdB = 20*log10(Am);
    Am_freqs_kHz = (1:L)*Wo*4/pi;
    rate_L_sample_freqs_kHz = ((1:L)*F0)/1000;
       
    figure(2); clf;

    % present everything in de-eph domain
    BdB = b(f,1:K) + de_emph(rate_K_sample_freqs_kHz);
    YdB = y(f,:) + de_emph(rate_Lhigh_sample_freqs_kHz);
    YdB_hat = y_hat(f,:) + de_emph(rate_Lhigh_sample_freqs_kHz);

    % Synthesise phase  using Hilbert Transform
    AmdB_hat = interp1([0 rate_Lhigh_sample_freqs_kHz 4], [0 YdB_hat 0], rate_L_sample_freqs_kHz, "spline", 0);
    phase_hat = synth_phase_from_mag(rate_Lhigh_sample_freqs_kHz, YdB_hat, Fs, Wo, L, 0);

    Am_hat = 10.^(AmdB_hat/20);
    subplot(211);
    [orig_lo orig_mid orig_hi] = synth_time(Wo, L, Am, phase(f,:), N);
    plot_time(orig_lo, orig_mid, orig_hi);
    subplot(212);
    [synth_lo synth_mid synth_hi] = synth_time(Wo, L, Am_hat, phase_hat, N);
    plot_time(synth_lo, synth_mid, synth_hi);

    if epslatex == 2
      fn = sprintf("%s_f%d_ml_bandpass.tex",samname,f);
      print(fn,"-depslatex","-S300,250");
      printf("printing... %s\n", fn);
    end

    figure(4); clf;
    subplot(211);
    plot_time_all(orig_lo, orig_mid, orig_hi);
    subplot(212);
    plot_time_all(synth_lo, synth_mid, synth_hi);
  
    if epslatex == 2
      fn = sprintf("%s_f%d_ml_all.tex",samname,f);
      print(fn,"-depslatex","-S300,250");
      printf("printing... %s\n", fn);
      restore_fonts(textfontsize,linewidth);
      epslatex = 0;
    end

    figure(3); clf;
    hold on;
    plot((0:255)*4000/256, Sw(f,:),";Sw;");
    plot(rate_K_sample_freqs_kHz*1000, BdB, ';dim K=20 B;c-');    
    plot((1:Lhigh-1)*F0high, YdB,";dim K=80 Y;g-");
    plot((1:Lhigh-1)*F0high, YdB_hat,";dim K=80 Y hat;r-");
    axis([0 Fs/2 0 70]); legend('boxoff');
    hold off;
 
    fn = sprintf("%s_f%d_ml.tex",samname,f);
    if epslatex ==1, print_eps_restore(fn,"-S300,250",textfontsize,linewidth); epslatex=0; end

    % interactive menu ------------------------------------------

    printf("\rframe: %d  menu: n-nxt b-bck q-qt p-printFig3 t-printFig2&4", f);
    fflush(stdout);
    k = kbhit();

    if k == 'n', f = f + 1; end
    if k == 'b', f = f - 1; end
    if k == 'p', epslatex=1; end
    if k == 't', epslatex=2; end

  until (k == 'q')
  printf("\n");

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

function print_eps_restore(fn,sz,textfontsize,linewidth)
  print(fn,"-depslatex",sz);
  printf("printing... %s\n", fn);
  restore_fonts(textfontsize,linewidth);
end

% synth time domain waveform, broken into three frequency bands
function [lo mid hi] = synth_time(Wo, L, Am, phase, N)
  lo = zeros(1,N);
  mid = ones(1,N);
  hi = ones(1,N);
  t=0:N-1;
  for m=1:round(L/4)
    lo += Am(m)*exp(j*(Wo*m*t + phase(m)));
  end
  for m=round(L/4)+1:round(L/2)
    mid += Am(m)*exp(j*(Wo*m*t + phase(m)));
  end
  for m=round(L/2)+1:L
    hi += Am(m)*exp(j*(Wo*m*t + phase(m)));
  end
endfunction

function plot_time(lo, mid, hi)
  y_off = -5000;
  hold on;
  plot(real(lo),sprintf('g;%s;',papr(lo)));
  plot(y_off + real(mid),sprintf('r;%s;',papr(mid)));
  plot(2*y_off + real(hi),sprintf('b;%s;',papr(hi)));
  legend("boxoff")
  hold off;
endfunction

function plot_time_all(lo, mid, hi)
  all = lo+mid+hi;
  plot(real(all),sprintf('g;%s;',papr(all)));
  legend("boxoff")
endfunction

function HpredB = de_emph(fkHz)
    w = pi*fkHz/4;
    Hpre = 2-2*cos(w);
    HpredB = -10*log10(Hpre);
endfunction
 