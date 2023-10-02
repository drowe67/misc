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
        sigma = std(rx_noise)
        sqrt(variance*0.5/M)
        % correlate at various time offsets

        % set up pilot samples, scale sucn that Ct_max = 1
        p = zeros(M,1); p(:,1) = tx(Ncp+1:Ncp+M); assert(length(p) == M);
        Ct_max = 1; p_scale = Ct_max*sqrt(p'*p)/(p'*p); p *= p_scale;

        Ct = zeros(1,nsymb); Ct2 = zeros(1,nsymb); 
        r = zeros(M,1);
        for s=1:nsymb
            st = (s-1)*(M+Ncp)+1;
            en = st+M+Ncp-1;
            r(:,1) = rx_noise(st+Ncp:en);
            Ct(s) = r'*p/sqrt(r'*r);
            Ct2(s) = r'*p;
        end
    
        if verbose == 2
            figure(1); clf; plot(Ct,'+'); mx = 1.2; axis([-mx mx -mx mx]);
            figure(2); clf; plot(abs(Ct)); axis([0 length(Ct) 0 mx])
            figure(3); clf; hist(abs(Ct),20);
            figure(4); clf; plot(Ct2,'+'); %mx = 1.2; axis([-mx mx -mx mx]);
            hold on;
            circle = exp(j*(0:0.001*2*pi:2*pi));
            plot(sigma*circle);
            p=1E-7; Cthresh = sigma*sqrt(-log(p));
            plot(Cthresh*circle);
            
            hold off;

        end

        sim_out.Ct = [sim_out.Ct; Ct];
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


 function threshold_demo(epslatex=0)
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


 