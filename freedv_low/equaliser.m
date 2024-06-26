% equaliser.m
% David Rowe Aug 2023
%
% Prototyping equaliser and multipath improvements for WP4000 Low SNR study. 

1;
pkg load signal;
qpsk;

function draw_ellipse(x1,y1,a,b,angle, leg)

  % Generate points on the ellipse
  theta = linspace(0, 2*pi, 100);  % Angle values
  x = x1 + a * cos(theta) * cos(angle) - b * sin(theta) * sin(angle);
  y = y1 + a * cos(theta) * sin(angle) + b * sin(theta) * cos(angle);

  % Plot the ellipse
  plot(x, y, leg);
end

function [tx_bits tx] = ofdm_modulator(Ns,Nc,Nd,M,Ncp,Winv,nbitsperframe,nframes,nsymb)
    printf("Modulate to rate Rs OFDM symbols...\n");
    tx_bits = rand(1,nframes*nbitsperframe) > 0.5; bit = 1;
    tx_symb = [];
    for f=1:nframes
      % set up Nc x Ns array of symbols with pilots
      atx_symb = zeros(Nc,Ns); 
      for c=1:Nc
        atx_symb(c,1) = 1; % TODO make equaliser work with any pilots
        for s=2:Ns
          atx_symb(c,s) = qpsk_mod(tx_bits(bit:bit+1)); bit += 2;
        end
      end
      
      % diversity copy
      tmp = [];
      for d=1:Nd;
        tmp = [tmp; atx_symb];
      end
      atx_symb = tmp;

      % add "wingman" pilots for first and last carriers
      wingman = [1 zeros(1,Ns-1)];
      atx_symb = [wingman; atx_symb; wingman];

      % normalise power when using diversity
      atx_symb *= 1/sqrt(Nd);
      
      tx_symb = [tx_symb atx_symb];
    end
    [r c] = size(tx_symb);
    assert(c == nsymb);
    
    % IDFT to convert to from freq domain rate Rs to time domain rate Fs

    printf("IDFT to time domain...\n");
    nsam = (M+Ncp)*nsymb;
    tx = zeros(1,nsam);
    for s=1:nsymb          
      st = (s-1)*(M+Ncp)+1; en = st+M+Ncp-1;
      tx(st+Ncp:en) = Winv*tx_symb(:,s)/M;
      
      % cyclic prefix
      tx(st:st+Ncp-1) = tx(st+M:en);
    end
endfunction

% Rate Rs modem simulation model -------------------------------------------------------

function sim_out = ber_test(sim_in)
    rand('seed',1);
    randn('seed',1);

    load_const;

    % init HF model

    printf("Generating HF model spreading samples...\n")
    hf_en = 0;
    if strcmp(ch,"mpp") || strcmp(ch,"mpd")
      hf_en = 1;
      % some typical values

      if strcmp(ch,"mpp")
        dopplerSpreadHz = 1.0; path_delay_s = 2E-3;
      else
        dopplerSpreadHz = 2.0; path_delay_s = 4E-3;
      end
      path_delay_samples = round(path_delay_s*Fs);
      
      spread1 = doppler_spread(dopplerSpreadHz, Fs, nsam);
      spread2 = doppler_spread(dopplerSpreadHz, Fs, nsam);

      % normalise power through HF channel
      hf_gain = 1.0/sqrt(var(spread1)+var(spread2));
    end

    % modulator -------------------------
    [tx_bits tx] = ofdm_modulator(Ns,Nc,Nd,M,Ncp,Winv,nbitsperframe,nframes,nsymb);
    
    % channel ---------------------------

    rx = tx;
    rx *= exp(j*ch_phase);
    ch_model = ones(Nc*Nd+2, nsymb).*exp(j*ch_phase);
    
    if hf_en
       printf("HF channel simulation...\n");

      % time domain rate Fs HF channel simulation

      x = path_delay_samples;
      rx_hf = hf_gain*spread1(1:nsam).*rx;
      rx_hf(x+1:end) += hf_gain*spread2(1:nsam-x).*rx(1:nsam-x);
      rx = rx_hf;
      
      if verbose == 2
        % frequency domain rate Rs HF simulation for reference.
        % This is an approximation as spread1 & spread2 are changing
        % slowly across the symbol
      
        for c=1:Nc*Nd+2
          for s=1:nsymb
            st = (s-1)*(M+Ncp)+1; en = st+M+Ncp-1;
            g1 = mean(spread1(st:en)); g2 = mean(spread2(st:en));
            ch_model(c,s) *= hf_gain*(g1 + exp(-j*path_delay_samples*w(c))*g2);
          end
        end
      end
    end

    printf("Noise loop and Rx...\n");
    for ne = 1:length(EbNovec)

        % work out noise power -------------
        EbNodB = EbNovec(ne);
        EsNodB = EbNodB + 10*log10(bps);
        EsNo = 10^(EsNodB/10);
        variance = 1/EsNo;
         
        % variance is noise power, which is divided equally between real and
        % imag components of noise

        noise = sqrt(variance*0.5/M)*(randn(1,nsam) + j*randn(1,nsam));
        rx_noise = rx + noise;

        % DFT to convert from time domain back to rate Rs freq domain
        % We assume "genie" timing, set just after cyclic prefix        

        rx_symb = ones(Nc*Nd+2,nsymb);
        for s=1:nsymb
          st = (s-1)*(M+Ncp)+1; en = st+M+Ncp-1;
          rx_symb(:,s) = Wfwd*rot90(rx_noise(st+Ncp:en),3);
        end
        
        % equaliser ------------------------------------------

        rx_symb_t = Ts*(0:nsymb-1);          % symbol times
        rx_pilots_t = Ts*(0:Ns:nsymb-1);     % pilot times

        % estimate channel by extracting pilots, smoothing (filtering) across time and freq,
        % and interpolating to obtain channel estimates for every symbol        
        rx_pilots = zeros(Nc*Nd+2,nsymb/Ns); 
        rx_ch = zeros(Nc*Nd+2,nsymb);

        for c=2:Nc*Nd+1
          % need to filter each carrier separately

          if sim_in.ls_pilots
            p = 1; local_path_delay_s = 0.004;
            a = local_path_delay_s*Fs;
            A = [1 exp(-j*w(c-1)*a); 1 exp(-j*w(c)*a); 1 exp(-j*w(c+1)*a)];
            P = inv(A'*A)*A'; 
            for s=1:Ns:nsymb
              h = zeros(3,1);
              for c1=-1:1
                h(c1+2) = rx_symb(c+c1,s);
              end
              g = P*h;
              rx_pilots(c,p) = g(1) + g(2)*exp(-j*w(c)*a);
              p++;
            end
          else
            % weighted average across frequency
            for c1=-1:1
              rx_pilots(c,:) += pilot_freq_weights(2+c1)*rx_symb(c+c1,1:Ns:nsymb);   
            end
            rx_pilots(c,:) ./= sum(pilot_freq_weights);
          end
 
          % bunch of options for resampler, takes pilots and estimates channel

          if strcmp(sim_in.resampler,"nearest")
            % use nearest pilot
            for s=0:nsymb-1
              ind = round(s/Ns)+1;
              ind = min(length(rx_pilots),ind); ind = max(1,ind);
              rx_ch(c,s+1) = rx_pilots(c,ind);
            end
          end
          if strcmp(sim_in.resampler,"lin2")
            % Linear interpolation between two nearest pilots, used on 700E and datac1 modes
            % 1.5dB IL AWGN, similar performance to mean4 on HF (1.5dB), trade off I guess
            rx_ch(c,:) = interp1(rx_pilots_t,rx_pilots(c,:),rx_symb_t);
          end
          if strcmp(sim_in.resampler,"mean4")
            % Mean of 4 adjacent pilots, similar time duration to 700D.  Lack of HF response in resampler
            % is quite obvious, about 0.5dB IL AWGN
            filter_delay = 1.5*Ns*Ts;
            h = [1 1 1 1]; h=h./sum(h);
            rx_pilots_filtered = filter(h,1,rx_pilots(c,:));
            rx_ch(c,:) = interp1(rx_pilots_t-filter_delay,rx_pilots_filtered,rx_symb_t,"extrap");
          end
          if strcmp(sim_in.resampler,"sinc")
            % use sinc filter to double the sample rate of the pilots, then linear interpolation
            % note delay and B needed hand tweaking using plot_freq_response above, and run_single to adjust delay
            filter_delay = 3.5*Ns*Ts;
            B=2*pi; n=(-3.5:0.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h);
            rx_pilots2 = zeros(1,2*length(rx_pilots));
            rx_pilots2(1:2:end) = rx_pilots(c,:);
            rx_pilots2_filtered = 2*filter(h,1,rx_pilots2);
            rx_pilots2_t = Ts*(0:Ns/2:nsymb-1);
            rx_ch(c,:) = interp1(rx_pilots2_t-filter_delay,rx_pilots2_filtered,rx_symb_t,"extrap");
          end

        end % for c=2:Nc*Nd+1
      
        % now we have channel estimates perform equalisation of received symbols, discarding
        % padding frames at the end and "wingman" carriers

        nsymb -= Npad*Ns;
        rx_symb_eq = zeros(Nc,nsymb);
        s = 1:nsymb;
        for c=2:Nc+1
          if sim_in.ideal_phase == 1
            for d=0:Nd-1
              rx_symb_eq(c-1,s) += rx_symb(c+d*Nc,s);
            end
          else
            for d=0:Nd-1
              if strcmp(combining,"mrc")
                 % maximum ratio combining
                rx_symb_eq(c-1,s) += rx_symb(c+d*Nc,s).*conj(rx_ch(c+d*Nc,s));
              else 
                % equal ratio combining, just equalise phase
                rx_symb_eq(c-1,s) += rx_symb(c+d*Nc,s).*exp(-j*angle(rx_ch(c+d*Nc,s)));
              end
            end
          end
        end % for c=2:Nc*Nd+1 
        
        if verbose == 2
            figure(3); clf;
            c = 2;
            if isfield(sim_in,"epslatex_interp")
                % just real part for Latex plot
                [textfontsize linewidth] = set_fonts(20);
                hold on;
                plot(rx_symb_t,real(ch_model(c,:)),'b;channel H;');
                plot(rx_pilots_t, real(rx_pilots(c,:)),'ro;pilots;');
                plot(rx_symb_t,real(rx_ch(c,:)),'r-;lin2 channel est;');
                hold off; axis([0 Ts*(nsymb-1) -2 2]); xlabel('time (s)'); ylabel('real');
                fn = "interpolator.tex";
                print(fn,"-depslatex","-S300,300");
                printf("printing... %s\n", fn);
                restore_fonts(textfontsize,linewidth);
            else 
                subplot(211); hold on;
                plot(rx_symb_t,real(ch_model(c,:)),'b-;channel Hn;');
                plot(rx_pilots_t, real(rx_pilots(c,:)),'ro;pilots;');
                plot(rx_symb_t,real(rx_ch(c,:)),'r-;channel est;');
                hold off; axis([0 Ts*(nsymb-1) -2 2]); xlabel('time (s)'); ylabel('real');
                subplot(212); hold on;
                plot(rx_symb_t,imag(ch_model(c,:)),'b-;channel Hn;');
                plot(rx_pilots_t, imag(rx_pilots(c,:)),'ro;pilots;');
                plot(rx_symb_t,imag(rx_ch(c,:)),'r-;channel est;');
                hold off; axis([0 Ts*(nsymb-1) -2 2]); xlabel('time (s)'); ylabel('imag');
            end
        end
 
        % demodulate rx symbols to bits
        nframes -= Npad;
        rx_bits = zeros(1,nframes*nbitsperframe);
        bit = 1;
        for f=1:nframes
          for c=1:Nc
            for s=2:Ns
              arx_symb = rx_symb_eq(c,(f-1)*Ns+s);
              rx_bits(bit:bit+1) = qpsk_demod(arx_symb);
              bit+=2;
            end
          end
        end
       
        % count errors -----------------------------------------

        error_pattern = xor(tx_bits(1:nframes*nbitsperframe), rx_bits);
        nerrors = sum(error_pattern);
        bervec(ne) = nerrors/nbits + 1E-12;

        % assume our FEC code can correct 10% random errors in one packet
        npacket_errors = 0;
        npackets = floor(length(rx_bits)/nbitsperpacket); berperpacket = [];
        for f=1:npackets
          st = (f-1)*nbitsperpacket + 1;
          en = st + nbitsperpacket - 1;
          nerr = sum(error_pattern(st:en));
          berperpacket = [berperpacket nerr/nbitsperpacket];
          if nerr > 0.1*nbitsperpacket
            npacket_errors++;
          end
        end
        pervec(ne) = npacket_errors/npackets + 1E-12;
        if verbose == 2
          figure(6);
          Tpacket=(Ts+Tcp)*Ns*nbitsperpacket/nbitsperframe;
          stem((0:npackets-1)*Tpacket,berperpacket);
          axis([0 npackets*Tpacket 0 0.25]); xlabel('Time (s)'); ylabel('BER/packet'); grid;
          if isfield(sim_in,"epslatex")
              fn = "ber_packet.tex";
              print(fn,"-depslatex","-S300,200");
              printf("printing... %s\n", fn);
          end
        end

        if verbose
          printf("EbNodB: % 5.1f nbits: %7d nerrors: %5d ber: %4.3f npackets %7d nperrors: %5d per: %4.3f\n", 
                 EbNodB, nbits, nerrors, bervec(ne), npackets, npacket_errors, pervec(ne));
          if verbose == 2
            if isfield(sim_in,"epslatex")
                [textfontsize linewidth] = set_fonts(20);
            end
            figure(4); clf;
            plot(rx_symb_eq*exp(j*pi/4),'+','markersize', 10);
            mx = max(abs(rx_symb_eq(:)));
            axis([-mx mx -mx mx]);
            if isfield(sim_in,"epslatex")
                fn = "scatter.tex";
                print(fn,"-depslatex","-S200,200");
                printf("printing... %s\n", fn);
                restore_fonts(textfontsize,linewidth);
            end

            figure(5); clf; plot_specgram(real(rx_noise),Fs,0,2500);
            if isfield(sim_in,"epslatex")
              fn = "spectrogram.tex";
              print(fn,"-depslatex","-S300,200");
              printf("printing... %s\n", fn);
           end
         end
        end
    end

    sim_out.bervec = bervec;
    sim_out.pervec = pervec;
endfunction

#{
  Single point simulation, various command line arguments.
  
  usage:
    octave:13> equaliser; run_single(1000,'awgn',4)
#}

function run_single(nbits = 1000, ch='awgn',EbNodB=100, varargin)
    sim_in.pilot_freq_weights = [0 1 0];
    sim_in.verbose     = 2;
    sim_in.nbits       = nbits;
    sim_in.EbNovec     = EbNodB;
    sim_in.ch          = ch;
    sim_in.resampler   = "lin2";
    sim_in.ch_phase    = 0;
    sim_in.ideal_phase = 0;
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

    sim_qpsk = ber_test(sim_in);
endfunction

function run_curves(itut_runtime=0,epslatex=0)
    max_nbits = 1E5;
    sim_in.verbose = 1;
    sim_in.EbNovec = 0:10;
    sim_in.ch = "awgn";
    sim_in.hf_en   = 0;
    sim_in.resampler = "";
    sim_in.ideal_phase = 0;
    sim_in.ls_pilots = 0;

    % AWGN -----------------------------

    awgn_theory = 0.5*erfc(sqrt(10.^(sim_in.EbNovec/10)));
    sim_in.nbits  = max_nbits;

    sim_in.pilot_freq_weights = [1 1 1];
    sim_in.resampler = "mean4"; awgn_sim_mean12 = ber_test(sim_in);
    sim_in.pilot_freq_weights = [0 1 0];
    sim_in.resampler = "lin2"; awgn_sim_lin2 = ber_test(sim_in);
    sim_in.ls_pilots = 1; awgn_sim_lin2ls = ber_test(sim_in);
     
    % HF -----------------------------

    hf_sim_in = sim_in; 
    hf_sim_in.EbNovec = 0:16;
    hf_sim_in.ch = "mpp";
    hf_sim_in.ideal_phase = 1;
    hf_sim_in.ls_pilots = 0;

    EbNoLin = 10.^(hf_sim_in.EbNovec/10);
    hf_theory = 0.5.*(1-sqrt(EbNoLin./(EbNoLin+1)));

    % run long enough to get sensible results, using rec from ITUT F.1487
    Fd_min = 1; run_time_s = 3000/Fd_min; Rb = 100;
    % default 0.1*run_time_s actually gives results close to theory
    if itut_runtime==0, run_time_s /= 10; end % default 0.1*run_time_s
    hf_sim_in.nbits = floor(run_time_s*Rb);

    hf_sim = ber_test(hf_sim_in);
    hf_sim_in.ideal_phase = 0;
    hf_sim_in.pilot_freq_weights = [1 1 1];
    hf_sim_in.resampler = "mean4"; hf_sim_mean12 = ber_test(hf_sim_in);
    hf_sim_in.pilot_freq_weights = [0 1 0];
    hf_sim_in.resampler = "lin2"; hf_sim_lin2 = ber_test(hf_sim_in);
    hf_sim_in.ls_pilots = 1; hf_sim_lin2ls = ber_test(hf_sim_in);

    hf_sim_in.ch = "mpd";
    hf_sim_in.pilot_freq_weights = [1 1 1]; hf_sim_in.ls_pilots = 0;
    hf_sim_in.resampler = "mean4"; hf_sim_mean12_mpd = ber_test(hf_sim_in);
    hf_sim_in.pilot_freq_weights = [0 1 0];
    hf_sim_in.resampler = "lin2"; hf_sim_lin2_mpd = ber_test(hf_sim_in);
    hf_sim_in.ls_pilots = 1; hf_sim_lin2ls_mpd = ber_test(hf_sim_in);

    if epslatex
        [textfontsize linewidth] = set_fonts();
    end

    % Plot results --------------------

    figure (1); clf;
    semilogy(sim_in.EbNovec, awgn_theory,'r+-;AWGN theory;')
    hold on;
    semilogy(sim_in.EbNovec, awgn_sim_mean12.bervec,'b+-;AWGN sim mean12;')
    semilogy(sim_in.EbNovec, awgn_sim_lin2.bervec,'g+-;AWGN sim lin2;')
    semilogy(sim_in.EbNovec, awgn_sim_lin2ls.bervec,'m+-;AWGN sim lin2ls;')

    semilogy(hf_sim_in.EbNovec, hf_theory,'r+-;HF theory;')
    semilogy(hf_sim_in.EbNovec, hf_sim.bervec,'c+-;HF sim ideal MPP;')
   
    semilogy(hf_sim_in.EbNovec, hf_sim_mean12.bervec,'bx-;HF sim mean12 MPP;')
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2.bervec,'gx-;HF sim lin2 MPP;')
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2ls.bervec,'mx-;HF sim lin2ls MPP;')

    semilogy(hf_sim_in.EbNovec, hf_sim_mean12_mpd.bervec,'bo-;HF sim mean12 MPD;')
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2_mpd.bervec,'go-;HF sim lin2 MPD;')
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2ls_mpd.bervec,'mo-;HF sim lin2ls MPD;')

    hold off;
    xlabel('Eb/No (dB)')
    ylabel('BER')
    grid("minor")
    axis([min(hf_sim_in.EbNovec) max(hf_sim_in.EbNovec) 1E-3 1])

    if epslatex
        fn = "equaliser.tex";
        print(fn,"-depslatex","-S350,350");
        printf("printing... %s\n", fn);
        restore_fonts(textfontsize,linewidth);
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

% BER and PER curves to explore the effect of diversity and time interleaving
% usage:
%   octave> equaliser; run_curves_diversity()

function run_curves_diversity(runtime_scale=0.1,epslatex=0)
    max_nbits = 1E4;
    sim_in.verbose = 1;
    sim_in.EbNovec = 0:10;
    sim_in.ch = "awgn";
    sim_in.hf_en   = 0;
    sim_in.resampler = "lin2";
    sim_in.ls_pilots = 1;
    sim_in.ideal_phase = 0;
    sim_in.ls_pilots = 1;

    % AWGN -----------------------------

    awgn_theory = 0.5*erfc(sqrt(10.^(sim_in.EbNovec/10)));
    sim_in.nbits = max_nbits;

    awgn_sim_lin2ls = ber_test(sim_in);
    
    % HF -----------------------------

    hf_sim_in = sim_in; 
    hf_sim_in.EbNovec = 0:10;
    hf_sim_in.ch = "mpp";
 
    EbNoLin = 10.^(hf_sim_in.EbNovec/10);
    hf_theory = 0.5.*(1-sqrt(EbNoLin./(EbNoLin+1)));

    % run long enough to get sensible results, using rec from ITUT F.1487
    Fd_min = 1; run_time_s = 3000/Fd_min; Rb = 1400;
    % sometimes useful to run quickly to test code
    % default 0.1*run_time_s actually gives results close to theory
    run_time_s *= runtime_scale;
    hf_sim_in.nbits = floor(run_time_s*Rb);
    printf("runtime (s): %f\n", run_time_s);
    
    hf_sim_in.ideal_phase = 0; 
    sim_in.ls_pilots = 0;
    hf_sim_in.pilot_freq_weights = [1 1 1];
    hf_sim_in.resampler = "mean4";
    hf_sim_mean12 = ber_test(hf_sim_in);

    hf_sim.ls_pilots = 1;
    hf_sim_in.resampler = "lin2";
    hf_sim_lin2ls = ber_test(hf_sim_in);
 
    hf_sim_in.Nd = 2;
    hf_sim_in.combining = "mrc"; 
    hf_sim_div2_mrc = ber_test(hf_sim_in);
    hf_sim_in.nbitsperpacket = 224*10; 
    hf_sim_div2_mrc_1800 = ber_test(hf_sim_in);
    hf_sim_in.nbitsperpacket = 224*20; 
    hf_sim_div2_mrc_3600 = ber_test(hf_sim_in);
    
    if epslatex
        [textfontsize linewidth] = set_fonts();
    end

    % Plot results --------------------

    figure (1); clf;
    semilogy(sim_in.EbNovec, awgn_theory,'r+-;AWGN theory;')
    hold on;
    semilogy(sim_in.EbNovec, awgn_sim_lin2ls.bervec,'b+-;AWGN lin2ls;')

    semilogy(hf_sim_in.EbNovec, hf_theory,'r+-;MPP theory;')
    semilogy(hf_sim_in.EbNovec, hf_sim_mean12.bervec,'g-;MPP mean12 MPP;')
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2ls.bervec,'b+-;MPP lin2ls;')
    semilogy(hf_sim_in.EbNovec, hf_sim_div2_mrc.bervec,'co-;MPP div2 MRC;')
    semilogy(hf_sim_in.EbNovec, hf_sim_div2_mrc_1800.bervec,'kx-;MPP sim div2 MRC 1.6s;')
    semilogy(hf_sim_in.EbNovec, hf_sim_div2_mrc_3600.bervec,'ox-;MPP sim div2 MRC 3.2s;')
    draw_ellipse(5, 0.1 , 5, 0.02, 0, 'r--;link closes;'); 

    hold off; xlabel('Eb/No (dB)'); ylabel('BER'); grid("minor");
    legend('boxoff'); legend('location','southwest');
    axis([min(hf_sim_in.EbNovec) max(hf_sim_in.EbNovec) 1E-3 0.5])

    if epslatex
        fn = "equaliser_div_ber.tex";
        print(fn,"-depslatex","-S350,350");
        printf("printing... %s\n", fn);
    end

    figure (2); clf;
    hold on;
    semilogy(sim_in.EbNovec, awgn_sim_lin2ls.pervec,'b+-;AWGN lin2ls;')
    semilogy(hf_sim_in.EbNovec, hf_sim_mean12.pervec,'g+-;MPP mean12;')
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2ls.pervec,'b+-;MPP lin2ls;')
    semilogy(hf_sim_in.EbNovec, hf_sim_div2_mrc.pervec,'co-;MPP div2 MRC;')
    semilogy(hf_sim_in.EbNovec, hf_sim_div2_mrc_1800.pervec,'kx-;MPP div2 MRC 1.6s;')
    semilogy(hf_sim_in.EbNovec, hf_sim_div2_mrc_3600.pervec,'ox-;MPP div2 MRC 3.2s;')
    draw_ellipse(5, 0.1, 5, 0.02, 0, 'r--;link closes;'); 

    hold off; xlabel('Eb/No (dB)'); ylabel('PER'); grid("minor"); legend('boxoff');
    axis([min(hf_sim_in.EbNovec) max(hf_sim_in.EbNovec) 1E-2 1])

    if epslatex
        fn = "equaliser_div_per.tex";
        print(fn,"-depslatex","-S350,350");
        printf("printing... %s\n", fn);
        restore_fonts(textfontsize,linewidth);
    end

    % save results for plotting later
    save run_curves_diversity.txt sim_in awgn_sim_lin2ls hf_sim_in hf_sim_mean12 ...
         hf_sim_lin2ls hf_sim_div2_mrc hf_sim_div2_mrc_1800 hf_sim_div2_mrc_3600
endfunction

% load up externally generated simulation data and plot on SNR axis
% usage:
%   octave> equaliser; plot_curves_snr()
%
% Note before running plot_curves_snr(), run these at least once to generate data:
%   octave> equaliser; run_curves_diversity()
%   $ ./ofdm_c.sh

function plot_curves_snr(epslatex=0)
    Ns = 8; Ts = 0.02; Tcp = 0.002;
    Lp = 10*log10(Ns/(Ns-1));        # 700D pilot symbol loss
    Lcp = -10*log10(1-Tcp/Ts);       # 700D cyclic prefix loss
    Lil = 0.5 + 1;                   # 0.5dB IL plus 1dB for compression (to help PAPR)
    Rb = 1400;                       # Uncoded (raw) bitrate for rate 0.5 code
    B = 3000;                        # we measure SNR in a 3000 Hz bandwidth
    rate = 0.5;                      # code rate
    EbNo_to_SNR = 10*log10(Rb/B) + Lp + Lcp + Lil;
    
    # data from run_curves_diversity() above
    load("run_curves_diversity.txt");

    # data from ./ofdm_c.sh
    x700d_awgn = load("700d_awgn.txt");
    x700d_mpp = load("700d_mpp.txt");
    x700e_awgn = load("700e_awgn.txt");
    x700e_mpp = load("700e_mpp.txt");
   
    awgn_theory = 0.5*erfc(sqrt(10.^(sim_in.EbNovec/10)));
    EbNoLin = 10.^(hf_sim_in.EbNovec/10);
    hf_theory = 0.5.*(1-sqrt(EbNoLin./(EbNoLin+1)));
   
    if epslatex
        [textfontsize linewidth] = set_fonts();
    end

    % Plot results --------------------

    snrvec = sim_in.EbNovec + EbNo_to_SNR;
    snrvec_coded = snrvec + 10*log10(rate);

    figure (1); clf;
    semilogy(snrvec, awgn_theory,'r+-;AWGN theory;')
    hold on;
    semilogy(x700d_awgn(:,1), x700d_awgn(:,2)+1E-12,'b+-;AWGN 700D;')
    semilogy(snrvec, awgn_sim_lin2ls.bervec,'g+-;AWGN lin2ls;')
  
    semilogy(snrvec, hf_theory,'r+-;MPP theory;')
    semilogy(x700d_mpp(:,1), x700d_mpp(:,2)+1E-12,'b+-;MPP 700D;')
    semilogy(snrvec, hf_sim_lin2ls.bervec,'g+-;MPP lin2ls;')
    semilogy(snrvec, hf_sim_div2_mrc.bervec,'co-;MPP div2 MRC;')
    draw_ellipse(5, 0.1 , 5, 0.02, 0, 'r--;link closes;');

    hold off; xlabel('SNR (dB)'); ylabel('UBER'); grid("minor");
    legend('boxoff'); legend('location','southwest');
    axis([min(snrvec) max(snrvec) 1E-3 0.5])

    if epslatex
        fn = "snr_ber.tex";
        print(fn,"-depslatex","-S350,350");
        printf("printing... %s\n", fn);
    end

    figure (2); clf;
    hold on;
    semilogy(snrvec_coded, awgn_sim_lin2ls.pervec,'b+-;AWGN lin2ls;')
    semilogy(x700d_awgn(:,1), x700d_awgn(:,4)+1E-12,'g+-;AWGN 700D;')

    semilogy(x700d_mpp(:,1), x700d_mpp(:,4)+1E-12,'g+-;MPP 700D;')
    semilogy(snrvec_coded, hf_sim_lin2ls.pervec,'b+-;MPP lin2ls;')
    semilogy(snrvec_coded, hf_sim_div2_mrc.pervec,'co-;MPP div2 MRC;')
    semilogy(snrvec_coded, hf_sim_div2_mrc_1800.pervec,'kx-;MPP div2 MRC 1.6s;')
    semilogy(snrvec_coded, hf_sim_div2_mrc_3600.pervec,'ox-;MPP div2 MRC 3.2s;')

    xlabel('SNR (dB)'); ylabel('PER'); grid("minor"); legend('boxoff');
    axis([min(snrvec_coded) max(snrvec) 1E-2 1])
    a = axis; c = (a(2)+a(1))/2; r = (a(2)-a(1))/2;
    draw_ellipse(c,0.1,r,0.02,0,'r--;link closes;'); 
    hold off;

    if epslatex
        fn = "snr_per.tex";
        print(fn,"-depslatex","-S350,350");
        printf("printing... %s\n", fn);
        restore_fonts(textfontsize,linewidth);
    end

endfunction

