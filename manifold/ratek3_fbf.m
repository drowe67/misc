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
%   $ ~/codec2-dev/build_linux/c2sim ~/codec2-dev/raw/big_dog.raw --hpf --dump big_dog
%   octave:14> ratek3_fbf("big_dog",61)

function ratek3_fbf(samname, f, vq_stage1_f32="", vq_stage2_f32="")
  more off;

  newamp_700c; melvq;
  Fs = 8000; Nb = 20; K = 20; resampler = 'spline'; Lhigh = 80; vq_en = 0; all_en = 0;
  amp_pf_en = 0; eq = 0; Kst = 0; Ken = K-1; pre_en = 0; w_en = 0;
  
  % load up text files dumped from c2sim ---------------------------------------

  sn_name = strcat(samname,"_sn.txt");
  Sn = load(sn_name);
  sw_name = strcat(samname,"_sw.txt");
  Sw = load(sw_name);
  model_name = strcat(samname,"_model.txt");
  model = load(model_name);
  [frames tmp] = size(model);
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

  if length(vq_stage1_f32)
    printf("training mic EQ...\n");
    % microphone equaliser (closed form solution)
    ratek3_batch; B=ratek3_batch_tool(samname,'K',20);
    q = mean(B-mean(B,2)) - mean(vq_stage1);
    q = max(q,0);
  end
  
  % Keyboard loop --------------------------------------------------------------

  k = ' '; 
  do
    s = [ Sn(2*f-1,:) Sn(2*f,:) ];
    figure(1); clf; plot(s); axis([1 length(s) -20000 20000]);

    Wo = model(f,1); F0 = Fs*Wo/(2*pi); L = model(f,2);
    Am = model(f,3:(L+2)); AmdB = 20*log10(Am);
    Am_freqs_kHz = (1:L)*Wo*4/pi;
    
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
    
    % Resample to rate K, optional EQ
    B = interp1(rate_Lhigh_sample_freqs_kHz, YdB, rate_K_sample_freqs_kHz, "spline", "extrap");
    if eq
      B -= q;
    end
    
    Eq = 0; B_hat = B;
    if vq_en
      amean = sum(B(Kst+1:Ken+1))/(Ken-Kst+1);
      target = zeros(1,K);
      target(Kst+1:Ken+1) = B(Kst+1:Ken+1)-amean;
      [res target_ ind] = mbest(vq, target, mbest_depth);
      B_hat = target_; B_hat(Kst+1:Ken+1) += amean;
      Eq = sum((target-target_).^2)/(Ken-Kst+1);
      gmin = amean; best_i = ind(1);

      figure(2); clf; hold on;
      for i=1:nvq
        plot(rate_K_sample_freqs_kHz*1000, vq(ind(i),:,i),'b;vq;');
      end
      plot(rate_K_sample_freqs_kHz*1000, target,'g;target;');
      plot(rate_K_sample_freqs_kHz*1000, q,'c;EQ;');
      hold off; axis([0 4000 -30 30]);  
    end
    
    YdB_ = interp1([0 rate_K_sample_freqs_kHz 4], [0 B 0], rate_Lhigh_sample_freqs_kHz, "spline", 0);
    nzero = floor(rate_K_sample_freqs_kHz(Kst+1)*1000/F0high);
    YdB_(1:nzero) = 0;
    
    % Optional amplitude post filtering
    if amp_pf_en
      [YdB_ SdB] = amplitude_postfilter(rate_Lhigh_sample_freqs_kHz, YdB_, Fs, F0high);
    end  
   
    figure(3); clf;
    hold on;
    l = sprintf(";rate %d AmdB;g+-", L);
    plot((0:255)*4000/256, Sw(f,:),";Sw;");
    plot(rate_K_sample_freqs_kHz*1000, B, ';rate K B;b+-');    
    plot(rate_K_sample_freqs_kHz*1000, B_hat, ";rate K B hat;c+-");

    Lmin = round(200/F0high); Lmax = floor(3700/F0high);
    %plot((1:Lhigh-1)*F0high, YdB_,";rate Lhigh Y hat;r--");
    le = sprintf("Eq %3.2fdB*dB", Eq);
    plot(rate_K_sample_freqs_kHz*1000, B - B_hat, sprintf(";%s;bk+-",le));
    axis([0 Fs/2 -10 80]);
    hold off;

    % interactive menu ------------------------------------------

    printf("\rframe: %d  menu: n-nxt b-bck q-qt v-vq[%d] e-mic_eq[%d]", f, vq_en, eq);
    fflush(stdout);
    k = kbhit();

    if k == 'n', f = f + 1; end
    if k == 'b', f = f - 1; end
    if k == 'v', vq_en = mod(vq_en+1,2); end
    if k == 'a', all_en = mod(all_en+1,2); end
    if k == 'e', eq = mod(eq+1,2); end
    if k == 'f', amp_pf_en = mod(amp_pf_en+1,2); end
    if k == 'p', pre_en = mod(pre_en+1,2); end
    if k == 'w', w_en = mod(w_en+1,2); end

  until (k == 'q')
  printf("\n");

endfunction
