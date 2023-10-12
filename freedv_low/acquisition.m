% equaliser.m
% David Rowe Oct 2023
%
% Prototyping acquisition improvements for WP4000 Low SNR study. 

1;
pkg load signal;
qpsk;
ofdm_lib;

function sim_out = acq_test(sim_in)
    rand('seed',1);
    randn('seed',1);

    load_const;
    sim_out.Ct = []; sim_out.Dt = [];
    Pthresh = 1E-3;
    if isfield(sim_in,"Pthresh")
       Pthresh = sim_in.Pthresh;
    end
    epslatex = 0;
    if isfield(sim_in,"epslatex")
      epslatex = sim_in.epslatex;
    end

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

        % set up pilot samples, scale sucn that Ct_max = 1
        p = zeros(M,1); p(:,1) = tx(Ncp+1:Ncp+M); assert(length(p) == M);
        Ct_max = 1; p_scale = Ct_max*sqrt(p'*p)/(p'*p); p *= p_scale;

        Ct = zeros(1,nsymb); Dt = zeros(1,nsymb); 
        r = zeros(M,1);
        for s=1:nsymb
            st = (s-1)*(M+Ncp)+1;
            en = st+M+Ncp-1;
            r(:,1) = rx_noise(st+Ncp:en);
            Ct(s) = r'*p/sqrt(r'*r);
            Dt(s) = r'*p;
        end

        sigma_est = std(Dt);
        % remove outliers
        Dt_prime = Dt(find(Dt < 2*sigma_est));
        sigma_prime_est = std(Dt_prime);

        % Determine threshold
        Dthresh = sigma_prime_est*sqrt(-log(Pthresh));

        sigma = sqrt(variance/M);
        if verbose == 2
          printf("signal_noise: %f sigma_est: %f sigma_prime_est: %f\n", sigma, sigma_est,sigma_prime_est);
        end
   
        if verbose == 2
            % 700D/E Ct algorithm (fixed thresh)
            figure(1); clf; plot(Ct,'+'); mx = 1.2; axis([-mx mx -mx mx]);
            figure(2); clf; plot(abs(Ct)); axis([0 length(Ct) 0 mx]);

            if epslatex
              [textfontsize linewidth] = set_fonts();
            end

            % Prototype Dt algorithm (variable threshold)
            figure(3); clf;
            [h x]=hist(abs(Dt),20);
            plot(x,h/trapz(x,h),'b;Histogram $|D_t|$;'); 
            sigma_r = sigma_prime_est/sqrt(2);
            p = (x./(sigma_r*sigma_r)).*exp(-(x.^2)/(2*sigma_r*sigma_r));
            hold on; plot(x,p,'g;Rayleigh PDF;'); hold off;
            legend('boxoff');
            if epslatex
              fn = "acq_dt_hist.tex";
              print(fn,"-depslatex","-S300,250");
              printf("printing... %s\n", fn);
            end

            figure(4); clf; plot(Dt,'+');
            if isfield(sim_in,"Pthresh")
              circle = exp(j*(0:0.001*2*pi:2*pi));
              hold on; plot(Dthresh*circle); hold off;
            end
            
            if epslatex
              fn = "acq_dt_scatter.tex";
              print(fn,"-depslatex","-S300,250");
              printf("printing... %s\n", fn);
              restore_fonts(textfontsize,linewidth);
            end
        end

        % count successful acquisitions
        Ncorrect = 0; Nfalse = 0;
        for s=1:nsymb
          if abs(Dt(s)) > Dthresh
            if mod(s,Ns) == 1
              Ncorrect++;
            else
              Nfalse++;
            end
          end
        end
        if verbose == 1
           printf("EbNodB: %5.0f Pcorrect: %4.3f Pfalse: %4.3f\n", EbNodB, Ncorrect/nframes, Nfalse/nframes);
        end
        sim_out.Pcorrect(ne) = Ncorrect/nframes;
        sim_out.Pfalse(ne) = Nfalse/nframes;
        if ne == 1
          sim_out.Ct = [sim_out.Ct; Ct];
        end
    end
endfunction

function run_single(nbits = 1000, ch='awgn',EbNodB=100, varargin)
    sim_in.verbose     = 2;
    sim_in.nbits       = nbits;
    sim_in.EbNovec     = EbNodB;
    sim_in.ch          = ch;
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
    
    i = 1;
    while i <= length(varargin)
      varargin{i}
      if strcmp(varargin{i},"ch_phase")
        sim_in.ch_phase = varargin{i+1}; i++;
      elseif strcmp(varargin{i},"combining")
        sim_in.combining = varargin{i+1}; i++;
      elseif strcmp(varargin{i},"resampler")
        sim_in.resampler = varargin{i+1}; i++;
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
      else
        printf("\nERROR unknown argument: %s\n", varargin{i});
        return;
      end
      i++; 
    end

    acq_test(sim_in);
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
  
    sim_out = acq_test(sim_in);
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

function test_cases(nbits=1E5,ch='awgn',epslatex=0)
    sim_in.verbose     = 1;
    sim_in.nbits       = nbits;
    sim_in.EbNovec     = [-100 0 10 100];
    sim_in.ch          = ch;
    sim_in.resampler   = "lin2";
    sim_in.ls_pilots   = 1;
    sim_in.Nd          = 1;
    sim_in.combining   = 'mrc';
    sim_in.epslatex    = epslatex;

    acq_test(sim_in);
endfunction

% just inner term of Dt summation
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

% with outer terms, PcPd terms don't cancel except at omega=0,
% the actual reinforce which makes it better athan 
function sweep_q(epslatex=0)
  Fs=8000; Rs=50; M=Fs/Rs;
  Nc = 13; 
  P = 1-2*(rand(1,Nc) > 0.5);
  %P = ones(1,Nc);
  % length 13 barker code
  P=[1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
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

