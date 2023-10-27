% equaliser.m
% David Rowe Oct 2023
%
% Prototyping acquisition improvements for WP4000 Low SNR study. 

1;
pkg load signal statistics;
qpsk;
ofdm_lib;

% 700D/E acq test

function sim_out = acq_test_Ct(sim_in)
    rand('seed',1);
    randn('seed',1);

    load_const;
    sim_out.Ct = [];

    [tx_bits tx] = ofdm_modulator(Ns,Nc,Nd,M,Ncp,Winv,nbitsperframe,nframes,nsymb);
 
    for ne = 1:length(EbNovec)
      rx = tx;
      
      % work out noise power -------------
      EbNodB = sim_in.EbNovec(ne);
      EsNodB = EbNodB + 10*log10(bps);
      EsNo = 10^(EsNodB/10);
      variance = 1/EsNo;
      noise = sqrt(variance*0.5/M)*(randn(1,nsam) + j*randn(1,nsam));
      rx_noise = rx + noise;

      % correlate at various time offsets

      % set up pilot samples, scale such that Ct_max = 1
      p = zeros(M,1); p(:,1) = tx(Ncp+1:Ncp+M); assert(length(p) == M);
      Ct_max = 1; p_scale = Ct_max*sqrt(p'*p)/(p'*p); p *= p_scale;

      Ct = zeros(1,nsymb);
      r = zeros(M,1);
      for s=1:nsymb
          st = (s-1)*(M+Ncp)+1;
          en = st+M+Ncp-1;
          r(:,1) = rx_noise(st+Ncp:en);
          Ct(s) = r'*p/sqrt(r'*r);
      end
  
      if verbose == 2
          % 700D/E Ct algorithm (fixed thresh)
          figure(1); clf; plot(Ct,'+'); mx = 1.2; axis([-mx mx -mx mx]);
          figure(2); clf; plot(abs(Ct)); axis([0 length(Ct) 0 mx]);
      end

      sim_out.Ct = [sim_out.Ct; Ct];
    end
endfunction

% Demo of potential issue with fixed threshold based 700D/E correlation algorithm
% 
% usage: 
%   octave:3> acquisition; threshold_700d_demo()

function threshold_700d_demo(epslatex=0)
    sim_in.verbose     = 1;
    sim_in.nbits       = 1E4;
    sim_in.EbNovec     = [0 10 100];
    sim_in.ch          = 'awgn';
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
  
    sim_out = acq_test_Ct(sim_in);
    if epslatex
        [textfontsize linewidth] = set_fonts();
    end

    figure(1); clf;
    subplot(211);
    plot(sim_out.Ct(1,:),'b+;Eb/No 0 dB;');
    axis([-0.5 1 -0.5 0.5]); legend('boxoff');
    subplot(212);
    plot(sim_out.Ct(2,:),'g+;Eb/No 10 dB;');
    axis([-0.5 1 -0.5 0.5]); legend('boxoff');
    if epslatex
        fn = "acq_ct_scatter.tex";
        print(fn,"-depslatex","-S300,300");
        printf("printing... %s\n", fn);
        restore_fonts(textfontsize,linewidth);
    end
endfunction


function sim_out = acq_test(sim_in)
  rand('seed',1);
  randn('seed',1);

  load_const;
  Pthresh = 3E-5;
  fmin_Rs = floor(-200/Rs);
  fmax_Rs = ceil(200/Rs);
  %fmin_Rs = fmax_Rs = 0;

  t_tol_syms = 1;
  f_tol_Rs = 0.5;
  timeStep = 2^(-5);
  freqStep = 0.5;
  if isfield(sim_in,"Pthresh")
      Pthresh = sim_in.Pthresh;
  end
  epslatex = 0;
  if isfield(sim_in,"epslatex")
    epslatex = sim_in.epslatex;
  end
  rand_freq = 1; rand_time = 1;
  if isfield(sim_in,"norand")
    rand_freq = 0; rand_time = 0;
  end

  nFreqSteps = length(fmin_Rs:freqStep:fmax_Rs);
  nTimeSteps = nsymb/timeStep;
 
  [tx_bits tx P] = ofdm_modulator(Ns,Nc,Nd,M,Ncp,Winv,nbitsperframe,nframes,nsymb);

  % Fixed timing offset applied to entire simulation, this puts target timing offset in
  % the middle of the search window which makes simulation easier as we don't get modulo
  % Ns peaks in Dt at either end of the search window (which are both correct)
  timeOffsetSymbols = Ns/2;
  if isfield(sim_in,"timeOffsetSymbols")
    timeOffsetSymbols = sim_in.timeOffsetSymbols;
  end
  timeOffsetSamples = (M+Ncp)*timeOffsetSymbols;
  tx = [zeros(1,timeOffsetSamples) tx];
  nsam = length(tx);

  rx = tx .* exp(j*ch_phase);
  hf_en = 0;
  if !strcmp(ch,'awgn')
    hf_en = 1;
    [spread1 spread2 hf_gain path_delay_samples] = gen_hf(ch,Fs,nsam);
    printf("HF channel simulation...\n");
    x = path_delay_samples;
    rx_hf = hf_gain*spread1(1:nsam).*rx;
    rx_hf(x+1:end) += hf_gain*spread2(1:nsam-x).*rx(1:nsam-x);
    rx = rx_hf;
  end

  % set up pilot samples, scale such that Dt_max = 1
  assert(Nd == 1); % TODO handle diversity
  p=zeros(M,1);
  for c=1:Nc+2
    p += P(c)*exp(j*w(c)*(0:M-1)).';
  end
  
  for ne = 1:length(EbNovec)
    
    % work out noise power -------------
    EbNodB = sim_in.EbNovec(ne);
    EsNodB = EbNodB + 10*log10(bps);
    EsNo = 10^(EsNodB/10);
    variance = 1/EsNo;
    noise = sqrt(variance*0.5/M)*(randn(1,nsam) + j*randn(1,nsam));
    rx_noise = rx + noise;

    % random additive sine wave
    if isfield(sim_in,"sineci") || isfield(sim_in,"sinein") 
      if isfield(sim_in,"sineci"), sin_rms = std(tx)/(10^(sim_in.sineci/10)); end
      if isfield(sim_in,"sinein"), sin_rms = (10^(sim_in.sinein/10))*sqrt(variance); end
      
      for n=1:Fs:length(rx_noise)
        omega_sin = (w(Nc+2)-w(1))*rand(1,1) + w(1);
        st = n; en = min(length(rx_noise),st+Fs-1);
        rx_noise(st:en) += sin_rms*exp(j*omega_sin*(0:(en-st)));
      end
    end

    % Sample Dt on grid of time and freq steps
   
    Dt = zeros(nFreqSteps,nTimeSteps);
    f_target_log =zeros(1,nTimeSteps); t_target_log = zeros(1,nTimeSteps);
    r = zeros(M,1);
    for s=1:timeStep:nsymb

      st = (s-1)*(M+Ncp)+1;
      en = st+M+Ncp-1;
      t_ind = s/timeStep-1/timeStep+1; % as s starts at 1, annoying Octave off by 1 issues :(

      % random freq offset every modem frame
      if mod(s,Ns) == 1
        f_target_Rs = 0.0;
        if rand_freq 
          f_target_Rs = fmin_Rs + rand(1,1)*(fmax_Rs-fmin_Rs);
        end
        omega = 2*pi*f_target_Rs*Rs/Fs;
      end
      f_target_log(t_ind) = f_target_Rs;

      % random target timing offset +/-Rs/2 for every modem frame
      if mod(s,Ns) == 1
        t_offset_Rs = 0.0;
        if rand_time
          t_offset_Rs = 0.5 - rand(1,1);
        end
        t_offset_ind = round(t_offset_Rs*(M+Ncp));
      end
      t_target_log(t_ind) = Ns*floor(s/Ns)+1 + timeOffsetSymbols + t_offset_Rs;
      if (s > 2) && (s < (nsymb-1))
        st -= t_offset_ind; en -= t_offset_ind;
      end

      for f_Rs=fmin_Rs:freqStep:fmax_Rs;
        f_Hz = f_Rs*Rs;
        omega_hat = 2*pi*f_Hz/Fs;
        r(:,1) = exp(j*(omega-omega_hat)*(0:M-1)).*rx_noise(st+Ncp:en);
        f_ind = f_Rs/freqStep + ceil(nFreqSteps/2);
        Dt(f_ind,t_ind) = r'*p;
      end
    end
 
    % count successful acquisitions
    Ncorrect = 0; Nfalse = 0; Ndetect = 0;
    Dtmax_log = [];
    t_max_log = []; f_fine_log = [];
    stats_log = [];
    for s=1:Ns:nsymb

      % threshold estimation for this modem frame
      st_ind = s/timeStep - 1/timeStep + 1;
      en_ind = st_ind + Ns/timeStep - 1;
      Dt_col = reshape(Dt(:,st_ind:en_ind),nFreqSteps*Ns/timeStep,1);

      sigma_est = std(Dt_col);
 
      % Determine threshold
      Dthresh = sigma_est*sqrt(-log(Pthresh));

      sigma = sqrt(variance/M);
      if verbose == 3
        printf("signal_noise: %f sigma_est: %fn", sigma, sigma_est);
      end

      % Search for maxima over time and freq samples for this modem frame
      Dtmax = 0;
      for t_syms=s:timeStep:s+Ns-timeStep
        for f_Rs=fmin_Rs:freqStep:fmax_Rs;
          t_ind = t_syms/timeStep-1/timeStep+1;
          f_ind = f_Rs/freqStep + ceil(nFreqSteps/2);
          if abs(Dt(f_ind,t_ind)) > abs(Dtmax)
            Dtmax = Dt(f_ind,t_ind);
            t_max = t_syms;
            f_max = f_Rs;
          end
        end
      end
      
      t_ind = s/timeStep-1/timeStep+1;

      f_fine_max = 0;
      if abs(Dtmax) > Dthresh

        % Evaluate candidate

        Ndetect++;
        t_error = t_max-t_target_log(t_ind);
        f_error = f_max-f_target_log(t_ind);
        correct = 0;
        if (abs(t_error) < t_tol_syms) && (abs(f_error) < f_tol_Rs)
          Ncorrect++;
          correct = 1;
        else
          Nfalse++;
        end
 
        % Attempt to refine freq estimate at chosen timing est
        
        % Reconstruct freq and time offset (messy - not the best simulation design!)
        omega = 2*pi*f_target_log(t_ind)*Rs/Fs;
        t_offset_Rs = t_target_log(t_ind) - Ns*floor(t_max/Ns)-1 - timeOffsetSymbols;
        t_offset_ind = round(t_offset_Rs*(M+Ncp));
        st = (t_max-1)*(M+Ncp)+1;
        en = st+M+Ncp-1;
        if (t_max > 2) && (t_max < (nsymb-1))
          st -= t_offset_ind; en -= t_offset_ind;
        end

        % Correlate over a fine freq grid
        fmin_fine_Rs = f_max - freqStep;
        fmax_fine_Rs = f_max + freqStep;
        Dtmax1 = 0;
        for f_Rs=fmin_fine_Rs:0.01:fmax_fine_Rs
          f_Hz = f_Rs*Rs;
          omega_hat = 2*pi*f_Hz/Fs;
          r(:,1) = exp(j*(omega-omega_hat)*(0:M-1)).*rx_noise(st+Ncp:en);
          Dt1 = r'*p;
          if abs(Dt1) > abs(Dtmax1)
            Dtmax1 = Dt1;
            f_fine_max = f_Rs;
          end
        end
        
        if correct
          f_fine_error = f_fine_max-f_target_log(t_ind);
          f_fine_log = [f_fine_log f_fine_error];
        end
        stats_log = [stats_log; t_target_log(t_ind) correct t_error f_error];
      end

      if verbose == 2
        printf("t_targ: %6.2f t_max: %6.2f f_targ: %5.2f f_max: %5.2f f_fine_max: %5.2f Dtmax: %5.2f Dthresh: %5.2f\n", 
               t_target_log(t_ind), t_max, f_target_log(t_ind), f_max, f_fine_max, abs(Dtmax), Dthresh);
      end

      Dtmax_log = [Dtmax_log Dtmax];
      t_max_log = [t_max_log t_max];
    end

    % mean time between detections (either true of false), for noise only case Tdet = Tnoise
    Tdet = (nsymb*(Ts+Tcp))/Ndetect;
    if verbose >= 1
        nTimeStepPerFrame = Ns/timeStep;
        nSamplesPerModemFrame = nFreqSteps*nTimeStepPerFrame;
        % prob of 1 or more detections per second if we just have noise at the input
        PnoisePerFrame = 1 - binopdf(0,nSamplesPerModemFrame,Pthresh);
        Tf = (Ts+Tcp)*Ns;
        printf("EbNodB: %4.0f Nfr: %4d Ndet: %4d Ncor: %4d Nfls: %4d Pcor: %4.3f Pfls: %4.3f Tdet/Tnoise: %4.3f Pnoise %4.3f Tnoise (theory): %4.2f f_freq std: %4.3f\n", 
                EbNodB, nframes, Ndetect, Ncorrect, Nfalse, Ncorrect/nframes, Nfalse/nframes, Tdet, PnoisePerFrame, Tf/PnoisePerFrame,std(f_fine_log));
     end

    sim_out.Pcorrect(ne) = Ncorrect/nframes;
    sim_out.Pfalse(ne) = Nfalse/nframes;
    sim_out.Tdet(ne) = Tdet;
    % note we just store latest here - only usable for single points
    sim_out.f_fine_log = f_fine_log;

    if verbose == 2
      if epslatex
        [textfontsize linewidth] = set_fonts();
      end

      % Plot PDFs - note sigma will only be of last modem frame, so YMMV with non-stationary
      % channels like MPP
      figure(1); clf;
      [h x]=hist(abs(Dt(:)),20);
      plot(x,h/trapz(x,h),'b;Histogram $|D_t|$;'); 
      sigma_r = sigma_est/sqrt(2);
      p_rayleigh = (x./(sigma_r*sigma_r)).*exp(-(x.^2)/(2*sigma_r*sigma_r));
      hold on; 
      plot(x,p_rayleigh,'g;Rayleigh PDF;'); 
      plot([Dthresh Dthresh],[0 max(p_rayleigh)*0.5],'r;Dthresh;')
      hold off;
      legend('boxoff');

      if epslatex
        fn = "acq_dt_hist.tex";
        print(fn,"-depslatex","-S300,250");
        printf("printing... %s\n", fn);
      end

      % 2D scatter plot
      figure(2); clf;
      plot(Dt(:),'+');
      if isfield(sim_in,"Pthresh")
        circle = Dthresh*exp(j*(0:0.001*2*pi:2*pi));
        hold on; plot(circle); hold off;
      end

      if epslatex
        fn = "acq_dt_scatter.tex";
        print(fn,"-depslatex","-S300,250");
        printf("printing... %s\n", fn);
        restore_fonts(textfontsize,linewidth);
      end

      % 3D scatter plot
      figure(3); clf;
      [nn cc] = hist3([real(Dt(:)) imag(Dt(:))],[25 25]);
      mesh(cc{1},cc{2},nn);
      hidden('off')
      if isfield(sim_in,"Pthresh")
        circle = Dthresh*exp(j*(0:0.001*2*pi:2*pi));
        hold on; plot3(real(circle),imag(circle),zeros(1,length(circle)),'r','linewidth',3); hold off;
      end
      
      figure(4); clf; plot_specgram(real(rx_noise),Fs,0,2500);

      % |Dt| against time
      figure(5); clf; 
      stem(t_max_log*(Ts+Tcp),abs(Dtmax_log)/(Nc+2));
      hold on; plot([0 nsymb*(Ts+Tcp)],[Dthresh Dthresh]/(Nc+2),'r--'); hold off;
      ylabel('$|D_{tmax}|$'); axis([0 nsymb*(Ts+Tcp) 0 max(abs(Dtmax_log)/(Nc+2))]);

      % Delta time and freq
      figure(6); clf;
      if length(stats_log)
        subplot(211); 
        plot(stats_log(:,1)*(Ts+Tcp),stats_log(:,3),'+');
        ylabel('$\Delta$ T (syms)');
        axis([0 nsymb*(Ts+Tcp) -Ns/2 Ns/2]);
        subplot(212);
        plot(stats_log(:,1)*(Ts+Tcp),stats_log(:,4),'+'); ylabel('$\Delta$ Freq (Rs)');
        axis([0 nsymb*(Ts+Tcp) fmin_Rs-0.001 fmax_Rs+0.001]); xlabel('Time(s)');
      end
      
      % Histogram of fine freq errors
      figure(7); clf;
      if length(f_fine_log)
        hist(f_fine_log);
      end
    end

  end
endfunction

% run test with a single EbNo point, lots of plots.  Useful for development/debug
% usage: 
%  acquisition; run_single(nbits=1E4,ch='mpp')
function run_single(nbits = 1000, ch='awgn',EbNodB=100, varargin)
    sim_in.verbose     = 2;
    sim_in.nbits       = nbits;
    sim_in.EbNovec     = EbNodB;
    sim_in.ch          = ch;
    sim_in.ch_phase    = 0;
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
    sim_in.Pthresh     = 3E-5;
 
    i = 1;
    while i <= length(varargin)
      if strcmp(varargin{i},"ch_phase")
        sim_in.ch_phase = varargin{i+1}; i++;
      elseif strcmp(varargin{i},"Nd")
        sim_in.Nd = varargin{i+1}; i++;
      elseif strcmp(varargin{i},"bitsperframe")
        sim_in.nbitsperframe = varargin{i+1}; i++;
      elseif strcmp(varargin{i},"bitsperpacket")
        sim_in.nbitsperpacket = varargin{i+1}; i++;    
      elseif strcmp(varargin{i},"epslatex")
        sim_in.epslatex = 1;    
      elseif strcmp(varargin{i},"Pthresh")
        sim_in.Pthresh = varargin{i+1}; i++;    
      elseif strcmp(varargin{i},"timeOffsetSymbols")
        sim_in.timeOffsetSymbols = varargin{i+1}; i++;    
      elseif strcmp(varargin{i},"norand")
        sim_in.norand = 1;    
      elseif strcmp(varargin{i},"sinecidB")
        % carrier to sinusoidal interferer ratio 
        sim_in.sineci = varargin{i+1}; i++;   
      elseif strcmp(varargin{i},"sineindB")
        % sinusoidal interferer to noise ratio
        sim_in.sinein = varargin{i+1}; i++;   
      else
        printf("\nERROR unknown argument: %s\n", varargin{i});
        return;
      end
      i++; 
    end

    acq_test(sim_in);
endfunction

% usage: acquisition; run_curves(0.1)
function run_curves(runtime_scale=0,epslatex=0)
    sim_in.verbose     = 1;
    sim_in.ch_phase    = 0;
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
    sim_in.Pthresh     = 3E-5;

    % run long enough to get sensible results, using rec from ITUT F.1487
    Fd_min = 1; run_time_s = 3000/Fd_min; Rb = 100;
    % default 0.1*run_time_s actually gives results close to theory
    run_time_s *= runtime_scale;
    sim_in.nbits = floor(run_time_s*Rb);
    printf("runtime (s): %f\n", run_time_s);

    EbNovec = 0:10; sim_in.EbNovec = EbNovec;
    sim_in.ch = 'awgn'; awgn = acq_test(sim_in);
    sim_in.ch = 'mpp'; mpp = acq_test(sim_in);
    sim_in.ch = 'mpd'; mpd = acq_test(sim_in);
    sim_in.ch = 'notch'; notch = acq_test(sim_in);

    if epslatex
      [textfontsize linewidth] = set_fonts();
    end

    figure(8); clf; hold on;
    plot(EbNovec,awgn.Pcorrect,'b+-;Pcorrect AWGN;');
    plot(EbNovec,awgn.Pfalse,'b+-;Pfalse AWGN;');
    plot(EbNovec,mpp.Pcorrect,'g+-;Pcorrect MPP;');
    plot(EbNovec,mpp.Pfalse,'g+-;Pfalse MPP;');
    plot(EbNovec,mpd.Pcorrect,'r+-;Pcorrect MPD;');
    plot(EbNovec,mpd.Pfalse,'r+-;Pfalse MPD;');
    plot(EbNovec,notch.Pcorrect,'c+-;Pcorrect Notch;');
    plot(EbNovec,notch.Pfalse,'c+-;Pfalse Notch;');
    hold off; axis([min(EbNovec) max(EbNovec) 0 1]); grid;
    xlabel('Eb/No (dB)'); legend('boxoff'); legend('location','east');

    if epslatex
      fn = "acq_curves.tex";
      print(fn,"-depslatex","-S350,250");
      printf("printing... %s\n", fn);
      restore_fonts(textfontsize,linewidth);
    end
   
endfunction

% acquisition performance with additive sine wave interferer
% usage: acquisition; run_curves_sin
function run_curves_sin(nbits=1E5, epslatex=0)
    sim_in.verbose     = 1;
    sim_in.ch_phase    = 0;
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
    sim_in.Pthresh     = 3E-5;
    sim_in.nbits       = nbits;
       
    sim_in.EbNovec = 4; sim_in.ch = 'awgn'; 
    awgn.P = [];
    ci_vec = -3:1:10;
    for n = 1:length(ci_vec)
      sim_in.sineci = ci_vec(n);
      sim_out = acq_test(sim_in);
      awgn.Pcorrect(n) = sim_out.Pcorrect;
      awgn.Pfalse(n) = sim_out.Pfalse;
    end

    sim_in.EbNovec = 7; sim_in.ch = 'mpp'; 
    for n = 1:length(ci_vec)
      sim_in.sineci = ci_vec(n);
      sim_out = acq_test(sim_in);
      mpp.Pcorrect(n) = sim_out.Pcorrect;
      mpp.Pfalse(n) = sim_out.Pfalse;
    end

    if epslatex
      [textfontsize linewidth] = set_fonts();
    end

    figure(8); clf; hold on;
    plot(ci_vec,awgn.Pcorrect,'b+-;Pcorrect AWGN;');
    plot(ci_vec,awgn.Pfalse,'b+-;Pfalse AWGN;');
    plot(ci_vec,mpp.Pcorrect,'g+-;Pcorrect MPP;');
    plot(ci_vec,mpp.Pfalse,'g+-;Pfalse MPP;');
    hold off; axis([min(ci_vec) max(ci_vec) 0 1]); grid;
    xlabel('C/I (dB)'); legend('boxoff'); legend('location','east');

    if epslatex
      fn = "acq_curves_sin.tex";
      print(fn,"-depslatex","-S350,250");
      printf("printing... %s\n", fn);
      restore_fonts(textfontsize,linewidth);
    end
   
endfunction

% Histograms of fine freq error
% usage: acquisition; fine_freq_hist
function fine_freq_hist(nbits=1E5, epslatex=0)
    sim_in.verbose     = 1;
    sim_in.ch_phase    = 0;
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
    sim_in.Pthresh     = 3E-5;
    sim_in.nbits       = nbits;
       
    sim_in.EbNovec = 100; sim_in.ch = 'awgn';
    awgn_out1 = acq_test(sim_in);
    sim_in.EbNovec = 2; sim_in.ch = 'awgn';
    awgn_out2 = acq_test(sim_in);
    sim_in.EbNovec = 7; sim_in.ch = 'mpp'; 
    mpp_out = acq_test(sim_in);

    if epslatex
      [textfontsize linewidth] = set_fonts();
    end

    figure(8); clf; hold on;
    [h1 x]=hist(awgn_out1.f_fine_log);
    %h1 = h1/trapz(x,h);
    plot(x,h1,'r;AWGN $E_b/N_0=100$ dB;'); 
    [h2 x]=hist(awgn_out2.f_fine_log);
    %h2 = h2/trapz(x,h);
    plot(x,h2,'b;AWGN $E_b/N_0=2$ dB;'); 
    [h3 x]=hist(mpp_out.f_fine_log);
    %h3 = h3/trapz(x,h);
    plot(x,h3,'g;MPP $E_b/N_0=7$ dB;'); 
    hold off; grid; axis([-0.2 0.2 0 max([h1 h2 h3])]);
    xlabel('Freq Error ($R_s$)'); legend('boxoff');

    if epslatex
      fn = "acq_fine_hist.tex";
      print(fn,"-depslatex","-S350,250");
      printf("printing... %s\n", fn);
      restore_fonts(textfontsize,linewidth);
    end
   
endfunction

% automated spot checks for acquisition (candidates for ctests in C implementation)
function spot_checks(nbits=1E4)
    sim_in.verbose     = 1;
    sim_in.nbits       = nbits;
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
 
    passes = tests = 0;
    sim_in.ch = 'awgn'; sim_in.EbNovec = 2;  sim_out = acq_test(sim_in);
    printf("---- AWGN Pcorrect: %4.2f Pfalse: %4.2f", sim_out.Pcorrect, sim_out.Pfalse);  
    tests++;
    if sim_out.Pcorrect > 0.8, printf(" [PASS]\n"); passes++; else printf(" [FAIL]\n"); end

    sim_in.ch = 'mpp'; sim_in.EbNovec = 7; sim_out = acq_test(sim_in);
    printf("---- MPP Pcorrect: %4.2f Pfalse: %4.2f", sim_out.Pcorrect, sim_out.Pfalse);  
    tests++;
    if sim_out.Pcorrect > 0.6, printf(" [PASS]\n");  passes++; else printf(" [FAIL]\n"); end
 
    sim_in.ch = 'awgn'; sim_in.EbNovec = -100;  sim_out = acq_test(sim_in);
    printf("---- Noise only Tdet: %4.2f", sim_out.Tdet);  
    tests++;
    if sim_out.Tdet > 1.5, printf(" [PASS]\n");  passes++; else printf(" [FAIL]\n"); end

    sim_in.ch = 'awgn'; sim_in.EbNovec = 4;  sim_in.sineci = 3; sim_out = acq_test(sim_in);
    printf("---- Sine interferer C/I = 3dB Pcorrect: %4.2f"), sim_out.Pcorrect;  
    tests++;
    if sim_out.Pcorrect > 0.8, printf(" [PASS]\n");  passes++; else printf(" [FAIL]\n"); end

    printf("\nPASSED %d/%d\n",passes,tests)
endfunction

% checking our scale parameter mapping for Rayleigh
function test_rayleigh
  N = 1E6;
  % total power in noise is sigma_n^2
  sigma_n = 0.5*sqrt(2);
  noise = (sigma_n/sqrt(2))*(randn(1,N) + j*randn(1,N));
  [h x] = hist(abs(noise),50);
  sigma = sigma_n/sqrt(2);
  p = (x./(sigma*sigma)).*exp(-(x.^2)/(2*sigma*sigma));
  figure(1); clf;
  plot(x,h/trapz(x,h),'b;histogram;');
  hold on;
  plot(x,p,'g;pdf;');
  hold off; grid;
end

% Exploring inner term of Dt summation, comparing derived sinc() form to
% initial form with summation
function sweep_q_inner
  Fs=8000; Rs=50; M=Fs/Rs;
  log_q = []; log_Dt = [];
  n = 0:M-1;
  for q=-5:0.1:5
    Dt = sum(exp(j*2*pi*q*n/M));
    log_q = [log_q q];
    log_Dt = [log_Dt Dt];
  end
  figure(1); clf; plot(log_q,abs(log_Dt));
  hold on;
  Dt_sinc = M*sinc(log_q);
  plot(log_q, abs(Dt_sinc));
  hold off;
endfunction

% Exploring complete expression for Dt, sweeping across freq as a function of q. 
% Note PcPd terms don't cancel except at omega=0 
function sweep_q(Nc=16, epslatex=0)
  Fs=8000; Rs=50; M=Fs/Rs;
  P = barker_pilots(Nc); 
  
  log_q = []; log_Dt = [];
  n = 0:M-1;
  for f=-5*Rs:0.1*Rs:5*Rs
    Dt = 0;
    for c=1:Nc
      for d=1:Nc
        omega = 2*pi*f/Fs;
        q = d - c - M*omega/(2*pi);
        Dt += P(c)*P(d)*sum(exp(j*2*pi*q*n/M));
      end
    end
    log_q = [log_q q];
    log_Dt = [log_Dt Dt];
  end
  if epslatex
      [textfontsize linewidth] = set_fonts();
  end
  figure(1); clf; plot(log_q,abs(log_Dt)/(M*Nc));
  xlabel('Freq Offset ($R_s$)'); ylabel('$|D_t|$');
  axis([-5 5 0 1]); grid;
  if epslatex
      fn = "acq_dt_q.tex";
      print(fn,"-depslatex","-S300,250");
      printf("printing... %s\n", fn);
      restore_fonts(textfontsize,linewidth);
  end
  endfunction

% Helper function to help design UW error thresholds. See also 
% https://www.rowetel.com/wordpress/?p=7467
% Source: codec2/octave/ofdm_helper.m
function ofdm_determine_bad_uw_errors(Nuw, epslatex=0)
  if epslatex, [textfontsize linewidth] = set_fonts(); end
  figure(1); clf;
   
  % Ideally the 10% and 50% BER curves are a long way apart
  
  % Pc(n), prob of n errors with correct acquistion
  Pc = binocdf(0:Nuw,Nuw,0.1);
  plot(0:Nuw, 1-Pc,';P(reject|correct);'); hold on; 

  % Pf(n), probability of n errors with false acquistion
  Pf = binocdf(0:Nuw,Nuw,0.5);
  plot(0:Nuw, Pf,';P(accept|false);'); 
  
  % Find n that minimises 1-Pc(n) and Pf(n)

  [tmp n_min] = min(1-Pc+Pf); 
  n_min -= 1;

  plot([n_min n_min],[0 1],';suggested threshold;'); hold off; grid
  xlabel('n (bit errors in UW)'); legend("boxoff"); axis([0 Nuw, 0 1]);

  if epslatex, print_eps_restore("acq_uw.tex","-S300,250",textfontsize,linewidth); end

  printf("for Nuw = %d, suggest reject error threshold > %d P(rejected|correct): %f P(accepted|false): %f\n", Nuw, n_min, 1-Pc(n_min+1),Pf(n_min));
end
