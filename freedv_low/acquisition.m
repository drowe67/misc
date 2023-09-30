% equaliser.m
% David Rowe Oct 2023
%
% Prototyping acquisition improvements for WP4000 Low SNR study. 

1;
pkg load signal;
qpsk;
ofdm_lib;

function acq_test(sim_in)
    rand('seed',1);
    randn('seed',1);

    load_const;

    [tx_bits tx] = ofdm_modulator(Ns,Nc,Nd,M,Ncp,Winv,nbitsperframe,nframes,nsymb);
    
    rx = tx;
    
    % work out noise power -------------
    EbNodB = sim_in.EbNovec;
    EsNodB = EbNodB + 10*log10(bps);
    EsNo = 10^(EsNodB/10);
    variance = 1/EsNo;
    noise = sqrt(variance*0.5/M)*(randn(1,nsam) + j*randn(1,nsam));
    rx_noise = rx + noise;

    % correlate at various time offsets

    Ct = zeros(1,nsymb); 
    p = zeros(M,1); p(:,1) = tx(Ncp:Ncp+M-1); assert(length(p) == M);
    p'*p
    r = zeros(M,1);
    for s=1:1
      st = (s-1)*(M+Ncp)+1; en = st+M+Ncp-1;
      r(:,1) = rx_noise(st+Ncp:en);
      Ct(s) = r'*p;
      r(1:4)
      p(1:4)
    end
    Ct(1)
    figure(1); clf; plot(Ct,'+');

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
      elseif strcmp(varargin{i},"epslatex_interp")
        sim_in.epslatex_interp = 1;    
      elseif strcmp(varargin{i},"epslatex")
        sim_in.epslatex = 1;    
      else
        printf("\nERROR unknown argument: %s\n", varargin{i});
        return;
      end
      i++; 
    end

    acq_test(sim_in);
endfunction


 