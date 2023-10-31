% ratek3_fbf.m
%
% David Rowe 2023
%
% Rate K Experiment 3
%
% Interactive Octave script to explore frame by frame operation of rate K
% resampling, VQ, and various keyboard controlled options.  Companion to
% ratek3_batch
%
% Usage:
%   Make sure codec2-dev is compiled with the -DDUMP option - see README.md for
%    instructions.
%   $ ~/codec2-dev/build_linux/src/c2sim ~/codec2-dev/raw/big_dog.raw --hpf --phase0 --dump big_dog
%   octave:14> ratek3_fbf("big_dog",61)

function ratek3_fbf(samname, f, vq_stage1_f32="", vq_stage2_f32="", K=79, Nb=100)
  more off;

  newamp_700c; melvq;
  Fs = 8000; resampler = 'spline'; Lhigh = 80; vq_en = 0; all_en = 0;
  amp_pf_en = 0; eq = 0; Kst = 0; Ken = K-1; pre_en = 0; w_en = 0;
  epslatex = 0; N=320;

  rateK_en = 0; if K==20, rateK_en =1; end

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

  % optionally load up VQ

  if length(vq_stage1_f32)
    vq_en = 1;
    vq_stage1 = load_f32(vq_stage1_f32,K);
    vq(:,:,1)= vq_stage1; 
    [M tmp] = size(vq_stage1); printf("stage 1 vq size: %d\n", M);
    nvq = 1; mbest_depth = 1;
    if length(vq_stage2_f32)
      vq_stage2 = load_f32(vq_stage2_f32,K);
      vq(:,:,2)= vq_stage2; 
      [M tmp] = size(vq_stage2); printf("stage 2 vq size: %d\n", M);
      nvq++; mbest_depth = 5;
    end
    w = ones(1,K);
  end

  % precompute filters at rate Lhigh. Note range of harmonics is 1:Lhigh-1, as
  % we don't use Lhigh-th harmonic as it's on Fs/2

  h = zeros(Lhigh, Lhigh);
  F0high = (Fs/2)/Lhigh;
  for m=1:Lhigh-1
    h(m,:) = generate_filter(m,F0high,Lhigh,Nb);
    plot((1:Lhigh-1)*F0high,h(m,1:Lhigh-1));
  end
  
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

    % optionally apply pre-emphasis
    if pre_en
      p = 1 - cos(Wo*(1:L)) + j*sin(Wo*(1:L));
      PdB = 20*log10(abs(p));
      AmdB += PdB;
    end

    % resample from rate L to rate Lhigh (both linearly spaced)
    AmdB_rate_Lhigh = interp1([0 Am_freqs_kHz 4], [0 AmdB 0], rate_Lhigh_sample_freqs_kHz, "spline", "extrap");
    
    % Filter at rate Lhigh, y = F(R(a)). Note we filter in linear energy domain, and Lhigh are linearly spaced

    YdB = zeros(1,Lhigh-1);
    for m=1:Lhigh-1
      Am_rate_Lhigh = 10.^(AmdB_rate_Lhigh/20);
      Y = sum(Am_rate_Lhigh.^2 .* h(m,1:Lhigh-1));
      YdB(m) = 10*log10(Y);
    end
    
    if rateK_en
      % Resample to rate K, optional EQ
      B = interp1(rate_Lhigh_sample_freqs_kHz, YdB, rate_K_sample_freqs_kHz, "spline", "extrap");
    end

    Eq = 0; if rateK_en, B_hat = B; else YdB_hat = YdB; end
    if vq_en
      if rateK_en
        amean = sum(B(Kst+1:Ken+1))/(Ken-Kst+1);
        target = zeros(1,K);
        target(Kst+1:Ken+1) = B(Kst+1:Ken+1)-amean;
        [res target_ ind] = mbest(vq, target, mbest_depth);
        B_hat = target_; B_hat(Kst+1:Ken+1) += amean;
      else
        amean = sum(YdB(Kst+1:Ken+1))/(Ken-Kst+1);
        target = zeros(1,K);
        target(Kst+1:Ken+1) = YdB(Kst+1:Ken+1)-amean;
        [res target_ ind] = mbest(vq, target, mbest_depth);
        YdB_hat = target_; YdB_hat(Kst+1:Ken+1) += amean;
      end

      Eq = sum((target-target_).^2)/(Ken-Kst+1);
    end
       
    figure(2); clf;
    % Synthesise phase  using Hilbert Transform
    if rateK_en
      AmdB_hat = interp1([0 rate_K_sample_freqs_kHz 4], [0 B_hat 0], rate_L_sample_freqs_kHz, "spline", 0);
      phase_hat = synth_phase_from_mag(rate_K_sample_freqs_kHz, B_hat, Fs, Wo, L, 0);
    else
      AmdB_hat = interp1([0 rate_Lhigh_sample_freqs_kHz 4], [0 YdB_hat 0], rate_L_sample_freqs_kHz, "spline", 0);
      phase_hat = synth_phase_from_mag(rate_Lhigh_sample_freqs_kHz, YdB_hat, Fs, Wo, L, 0);
    end
    Am_hat = 10.^(AmdB_hat/20);
    subplot(211);
    [orig_lo orig_mid orig_hi] = synth_time(Wo, L, Am, phase(f,:), N);
    plot_time(orig_lo, orig_mid, orig_hi);
    subplot(212);
    [synth_lo synth_mid synth_hi] = synth_time(Wo, L, Am_hat, phase_hat, N);
    plot_time(synth_lo, synth_mid, synth_hi);

    if epslatex == 2
      fn = sprintf("%s_f%d_k%d_bandpass.tex",samname,f,K);
      print(fn,"-depslatex","-S300,250");
      printf("printing... %s\n", fn);
    end

    figure(4); clf;
    subplot(211);
    plot_time_all(orig_lo, orig_mid, orig_hi);
    subplot(212);
    plot_time_all(synth_lo, synth_mid, synth_hi);
  
    if epslatex == 2
      fn = sprintf("%s_f%d_k%d_all.tex",samname,f,K);
      print(fn,"-depslatex","-S300,250");
      printf("printing... %s\n", fn);
      restore_fonts(textfontsize,linewidth);
      epslatex = 0;
    end

    figure(3); clf;
    hold on;
    plot((0:255)*4000/256, Sw(f,:),";Sw;");
    le = sprintf("Eq %3.2fdB*dB", Eq);
    if rateK_en
      plot(rate_K_sample_freqs_kHz*1000, B, ';rate K=20 B;g-');    
      plot(rate_K_sample_freqs_kHz*1000, B_hat, ";rate K=20 B hat;r-");
      plot(rate_K_sample_freqs_kHz*1000, B - B_hat, sprintf(";%s;bk-",le));
    else
      plot((1:Lhigh-1)*F0high, YdB,";rate K=80 Y;g-");
      plot((1:Lhigh-1)*F0high, YdB_hat,";rate K=80 Y hat;r-");
      plot((1:Lhigh-1)*F0high, YdB - YdB_hat, sprintf(";%s;bk-",le));
    end
    axis([0 Fs/2 -10 80]); legend('boxoff');
    hold off;
 
    fn = sprintf("%s_f%d_k%d.tex",samname,f,K);
    if epslatex ==1, print_eps_restore(fn,"-S300,250",textfontsize,linewidth); epslatex=0; end

    % interactive menu ------------------------------------------

    printf("\rframe: %d  menu: n-nxt b-bck q-qt v-vq[%d] p-printFig3 t-printFig2&4", f, vq_en);
    fflush(stdout);
    k = kbhit();

    if k == 'n', f = f + 1; end
    if k == 'b', f = f - 1; end
    if k == 'v', vq_en = mod(vq_en+1,2); end
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
