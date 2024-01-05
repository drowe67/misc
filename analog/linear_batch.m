% linear_batch.m
%
% David Rowe 2023
%
% Batch processing tool for misc/analog project - performs linear DSP processing

1;

function mel = ftomel(fHz)
  mel = floor(2595*log10(1+fHz/700)+0.5);
endfunction

function rate_K_sample_freqs_kHz = mel_sample_freqs_kHz(K)
  mel_start = ftomel(200); mel_end = ftomel(3700);
  step = (mel_end-mel_start)/(K-1);
  mel = mel_start:step:mel_end;
  rate_K_sample_freqs_Hz = 700*((10 .^ (mel/2595)) - 1);
  rate_K_sample_freqs_kHz = rate_K_sample_freqs_Hz/1000;
endfunction

function fHz = warp(k, K)
  mel_start = ftomel(200); mel_end = ftomel(3700);
  step = (mel_end-mel_start)/(K-1);
  mel = mel_start + (k-1)*step;
  fHz = 700*((10 .^ (mel/2595)) - 1);
endfunction

function k = warp_inv(fHz, K)
  mel_start = ftomel(200); mel_end = ftomel(3700);
  step = (mel_end-mel_start)/(K-1);
  mel = ftomel(fHz) - step;
  k = (ftomel(fHz) - mel_start)/step + 1;
endfunction

function h = generate_filter(m,f0,L,Nb)
  g = zeros(1,L);
  fb = m*f0;
  b = warp_inv(fb,Nb);
  st = round(warp(b-1,Nb)/f0); st = max(st,1);
  en = round(warp(b+1,Nb)/f0); en = min(en,L);
  %printf("fb: %f b: %f warp(b-1): %f warp(b+1): %f st: %d en: %d\n", fb, b, warp(b-1,Nb), warp(b+1,Nb), st, en);
  for k=st:m-1
    g(k) = (k-st)/(m-st);
  end
  g(m) = 1;
  for k=m+1:en
    g(k) = (en-k)/(en-m);
  end
  g_sum = sum(g);
  h = g/g_sum;
endfunction

% normalises the energy in AmdB_rate2 to be the same as AmdB_rate1
function AmdB_rate2_hat = norm_energy(AmdB_rate1, AmdB_rate2)
  c = sum(10 .^ (AmdB_rate1/10))/sum(10 .^ (AmdB_rate2/10));
  AmdB_rate2_hat = AmdB_rate2 + 10*log10(c);
end

% Synthesised phase0 model using Hilbert Transform
function phase0 = synth_phase_from_mag(mag_sample_freqs_kHz, mag_dB, Fs, Wo, L)
  Nfft=512;
  sample_freqs_kHz = (Fs/1000)*[0:Nfft/2]/Nfft;  % fft frequency grid (nonneg freqs)
  Gdbfk = interp1([0 mag_sample_freqs_kHz 4], [0 mag_dB 0], sample_freqs_kHz, "spline", "extrap");

  [phase_ht s] = mag_to_phase(Gdbfk, Nfft);
  phase0 = zeros(1,L);
  for m=1:L
    b = round(m*Wo*Nfft/(2*pi));
    phase0(m) = phase_ht(b);
  end
end

% Returns a file of b vectors, used for input to PyTorch ML inference tool
function linear_batch_ml_in(samname, varargin)
  more off;

  Fs = 8000; max_amp = 160; resampler='spline'; Lhigh=80; max_amp = 160;
  
  K=20; Nb = 20; B_out_fn = ""; Y_out_fn = "";

  i = 1;
  while i<=length(varargin)
    if strcmp(varargin{i},"B_out")
      B_out_fn = varargin{i+1}; i++;
    elseif strcmp(varargin{i},"Y_out") 
      Y_out_fn = varargin{i+1}; i++;
    elseif strcmp(varargin{i},"Nb")
      Nb = varargin{i+1}; i++;
    else
      printf("\nERROR unknown argument: %s\n", varargin{i});
      return;
    end
    i++;      
  end  
  printf("B_out_fn: %s Y_out_fn: %s \n", B_out_fn, Y_out_fn);

  model_name = strcat(samname,"_model.bin");
  model = load_codec2_model(model_name);
  [frames tmp] = size(model);
  rate_K_sample_freqs_kHz = mel_sample_freqs_kHz(K);
  B = zeros(frames,K);
  YdB = zeros(frames,Lhigh-1);
 
  % precompute filters at rate Lhigh. Note range of harmonics is 1:Lhigh-1, as
  % we don't use Lhigh-th harmonic as it's on Fs/2

  h = zeros(Lhigh, Lhigh);
  F0high = (Fs/2)/Lhigh;
  for m=1:Lhigh-1
    h(m,:) = generate_filter(m,F0high,Lhigh,Nb);
  end
  rate_Lhigh_sample_freqs_kHz = (F0high:F0high:(Lhigh-1)*F0high)/1000;
  sum_Eq = 0; nEq = 0;

  for f=1:frames
    Wo = model(f,1); F0 = Fs*Wo/(2*pi); L = model(f,2);
    Am = model(f,3:(L+2)); AmdB = 20*log10(Am);
    rate_L_sample_freqs_kHz = ((1:L)*F0)/1000;
       
    % preemphasis 
    w = Wo*(1:L);
    Hpre = 2-2*cos(w);
    AmdB += 10*log10(Hpre);

    % resample from rate L to rate Lhigh (both linearly spaced)
    AmdB_rate_Lhigh = interp1([0 rate_L_sample_freqs_kHz 4], [0 AmdB 0], rate_Lhigh_sample_freqs_kHz, "spline", "extrap");
    AmdB_rate_Lhigh = norm_energy(AmdB, AmdB_rate_Lhigh);
 
    % Filter at rate Lhigh, y = F(R(a)). Note we filter in linear energy domain, and Lhigh are linearly spaced
    for m=1:Lhigh-1
      Am_rate_Lhigh = 10.^(AmdB_rate_Lhigh/20);
      Y = sum(Am_rate_Lhigh.^2 .* h(m,1:Lhigh-1));
      YdB(f,m) = 10*log10(Y);
    end
    
    % Resample from rate Lhigh to rate K b=R(Y), note K are non-linearly spaced (warped freq axis)
    B(f,:) = interp1(rate_Lhigh_sample_freqs_kHz, YdB(f,:), rate_K_sample_freqs_kHz, "spline", "extrap");
    B(f,:) = norm_energy(YdB(f,:), B(f,:));

    printf("%d/%d %3.0f%%\r", f,frames, (f/frames)*100);
  end
  printf("\n");
  
  % Optionally write a y .f32 file as this is used for energy normalisation during inference, shape of y_hat is
  % inferred from b. Also useful for measuring SD
  if length(Y_out_fn)
    fy = fopen(Y_out_fn,"wb");
    for f=1:frames
      %just pass a dummy vector with the same energy as y_hat to check inference is not cheating
      %e = sum(10.^(YdB(f,:)/10));
      %Yfloat = [10*log10(e) zeros(1,Lhigh-2)];
      Yfloat = YdB(f,:);
      fwrite(fy, Yfloat, "float32");
    end
    fclose(fy);
  end

  % write b to a .f32 file for input to ML inference
  if length(B_out_fn)
    fb = fopen(B_out_fn,"wb");
    for f=1:frames
      Bfloat = B(f,:);
      fwrite(fb, Bfloat, "float32");
      fwrite(fb,[model(f,1) model(f,end)], "float32");
    end
    fclose(fb);
  end

endfunction

% From y_hat outputs files of {Am} and {arg[Hm]}, used for ML inference output processing
function linear_batch_ml_out(samname, varargin)
  more off;

  Fs = 8000; max_amp = 160; resampler='spline'; Lhigh=80; max_amp = 160;
  
  Y_in_fn = ""; A_out_fn = ""; H_out_fn = ""; 

  i = 1;
  while i<=length(varargin)
    if strcmp(varargin{i},"Y_in") 
      Y_in_fn = varargin{i+1}; i++;
    elseif strcmp(varargin{i},"Y_hat_in") 
      Y_hat_in_fn = varargin{i+1}; i++;
    elseif strcmp(varargin{i},"A_out") 
      A_out_fn = varargin{i+1}; i++;
    elseif strcmp(varargin{i},"H_out")
      H_out_fn = varargin{i+1}; i++;
    else
      printf("\nERROR unknown argument: %s\n", varargin{i});
      return;
    end
    i++;      
  end  
  printf("Y_in_fn: %s Y_hat_in_fn: %s A_out_fn: %s H_out_fn: %s\n", Y_in_fn, Y_hat_in_fn, A_out_fn, H_out_fn);

  model_name = strcat(samname,"_model.bin");
  model = load_codec2_model(model_name);
  [frames tmp] = size(model);
  YdB = zeros(frames,Lhigh-1); YdB_ = zeros(frames,Lhigh-1);
  
  fy = fopen(Y_in_fn,"rb");
  fy_hat = fopen(Y_hat_in_fn,"rb");
 
  % just measure squared error between 200 and 3700 Hz
  se_low = round(Lhigh*200/(Fs/2));
  se_high = round(Lhigh*3700/(Fs/2));
  mse_sum = 0;

  F0high = (Fs/2)/Lhigh;
  rate_Lhigh_sample_freqs_kHz = (F0high:F0high:(Lhigh-1)*F0high)/1000;

  for f=1:frames
    Wo = model(f,1); F0 = Fs*Wo/(2*pi); L = model(f,2);
    Am = model(f,3:(L+2)); AmdB = 20*log10(Am);
    rate_L_sample_freqs_kHz = ((1:L)*F0)/1000;
       
    YdB = fread(fy, Lhigh-1, "float32")';
    YdB_(f,:) = fread(fy_hat, Lhigh-1, "float32");
    se = (YdB - YdB_(f,:)).^2;  
    mse_sum += mean(se(se_low:se_high));
 
    % back to rate L
    AmdB_ = interp1([0 rate_Lhigh_sample_freqs_kHz 4], [0 YdB_(f,:) 0], rate_L_sample_freqs_kHz, "spline", "extrap");
    AmdB_ = norm_energy(YdB_(f,:), AmdB_);
 
    % de-emphasis 
    w = Wo*(1:L);
    Hpre = 2-2*cos(w);
    AmdB_ -= 10*log10(Hpre);
    Am_(f,1:L) = 10.^(AmdB_/20);
    
    if length(H_out_fn)
      H(f,1:L) = synth_phase_from_mag(rate_Lhigh_sample_freqs_kHz, YdB_(f,:), Fs, Wo, L);
    end

    printf("%d/%d %3.0f%%\r", f,frames, (f/frames)*100);
  end

  sd = mse_sum/frames;
  printf("\nSD = %5.2f dB^2 %5.2f dB\n", sd, sqrt(sd));
  
  fam = fopen(A_out_fn,"wb");
  for f=1:frames
    Afloat_ = zeros(1,max_amp);
    L = model(f,2);
    Afloat_(2:L+1) = Am_(f,1:L);
    fwrite(fam, Afloat_, "float32");
  end
  fclose(fam);
   
  max_amp = 160;
  fhm = fopen(H_out_fn,"wb");
  for f=1:frames
    Hfloat = zeros(1,2*max_amp);
    L = model(f,2);
    for m=1:L
      Hfloat(2*m+1) = cos(H(f,m));
      Hfloat(2*m+2) = sin(H(f,m));
    end
    fwrite(fhm, Hfloat, "float32");
  end
  fclose(fhm);

endfunction
