% equaliser.m
% David Rowe Aug 2023
%
% Prototyping equaliser improvements.  The equaliser uses pilot symbols to estimate the channel
% and apply corrections to (equalise) received data symbols.  

1;
pkg load signal;
qpsk;

% simple time domain example of resampling
function test_resampler
    randn('seed',1);

    % Fd is Doppler bandwidth (Hz).  Doppler signal is complex so
    % this is a total bandwidth of Fd, sperad Fd/2 either side of 0 Hz
    Fd = 2;          
    Fs = 8000; T=1/Fs; Trun=5;
    N  = Fs*Trun;
    [d1 states] = doppler_spread(Fd, Fs, N);

    % Decimate down to pilot symbol rate of 1/(Ts*Np)sample rate 6.25Hz (Np=8 for 700D)
    Ts=0.02; Np=8; Fp=1/(Ts*Np); Tp=1/Fp;
    M = Fs/Fp;
    assert (M == floor(M));
    n2 = 1:M:length(d1);
    d2 = d1(n2);
    
    % attempt resampling at mid point (doubling sample rate of d2), using a 6 point resampler
    % uses 4 past pilots and 2 future
    B=pi; n=(-3.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h);
    d3 = filter(h,1,d2);

    figure(1); clf; 
    subplot(211); hold on;
    plot(T*(0:N-1),real(d1));
    stem(Tp*(0:length(d2)-1),real(d2));
    stem(Tp*(0:length(d2)-1) - 3.5*Tp,real(d3));
    hold off; axis([0 Trun -1 1]); xlabel('time (s)'); ylabel('real');
    subplot(212); hold on;
    plot(T*(0:N-1),imag(d1));
    stem(Tp*(0:length(d2)-1),imag(d2));
    stem(Tp*(0:length(d2)-1) - 3.5*Tp,imag(d3));
    hold off; axis([0 Trun -1 1]); xlabel('time (s)'); ylabel('imag');
end

% plot frequency response of different resamplers
function plot_resampler_freq_response
    figure(4); clf; hold on;
    h = [1 1 1 1]/4; [H w] = freqz(h); plot(w,20*log10(H),'b;[1 1 1 1];');
    B=pi/2; n=(-1.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h); [H w] = freqz(h); plot(w,20*log10(H),'g;sinc pi/2;');
    B=pi; n=(-1.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h); [H w] = freqz(h); plot(w,20*log10(H),'r;sinc pi;');
    B=2*pi; n=(-3.5:0.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h); [H w] = freqz(h); plot(w,20*log10(H),'c;6 tap pi;');
    axis([0 pi -20 3]); grid; title('4 sample resamplers freq resp')
end

% Rate Rs modem simulation model -------------------------------------------------------

#{
    TODO
    [X] BER counting with random QPSK symbols
    [X] with inserted pilots at Np
    [X] additive noise
    [X] sample Doppler function at symbol rate
    [X] estimate channel from pilots using chosen filter function
    [X] equalise using channel estimate
    [ ] plot curves from different resamplers
    [ ] plot curves from different Doppler rates
    [ ] extend to use pilots from carriers either side, so pilots with slightly different channel
    [ ] extend to high bandwidth, high SNR fading
    [ ] diversity
    [ ] LPDC (combined with diversity)
    [ ] output curves and write up
#}

function sim_out = ber_test(sim_in)
    rand('seed',1);
    randn('seed',1);

    bps     = 2;     % two bits/symbol for QPSK
    Rs      = 50;    % symbol rate (needed for HF model)
    Ts      = 1/Rs;
    Np      = 8;     % one pilot every Np symbols
    Npad    = Np*3;  % extra symbols at end to deal with interpolator edges
    Nc      = 3;     % number of carriers
    Fs      = 8000;  % only used for HF channel model calculations

    verbose = sim_in.verbose;
    EbNovec = sim_in.EbNovec;
    ch   = sim_in.ch;
    nbitsvec = sim_in.nbits;
    nsymb_max = max(nbitsvec)/bps+Npad;
    ch_phase = 0;
    if isfield(sim_in,"ch_phase")
      ch_phase = sim_in.ch_phase;
    end
    pilot_freq_weights = [0 1 0];
    if isfield(sim_in,"pilot_freq_weights")
      pilot_freq_weights = sim_in.pilot_freq_weights;
    end

    % init HF model

    hf_en = 0;
    if strcmp(ch,"mpp") || strcmp(ch,"mpd")
      hf_en = 1;
      % some typical values

      if strcmp(ch,"mpp")
        dopplerSpreadHz = 1.0; path_delay_s = 2E-3;
      else
        dopplerSpreadHz = 2.0; path_delay_s = 4E-3;
      end

      spread1 = doppler_spread(dopplerSpreadHz, Rs, nsymb_max);
      spread2 = doppler_spread(dopplerSpreadHz, Rs, nsymb_max);

      % normalise power through HF channel
      hf_gain = 1.0/sqrt(var(spread1)+var(spread2));
    end

    for ne = 1:length(EbNovec)

        % work out noise power -------------
        EbNodB = EbNovec(ne);
        EsNodB = EbNodB + 10*log10(bps);
        EsNo = 10^(EsNodB/10);
        variance = 1/EsNo;

        % integer number of "modem" frames
        nbits = nbitsvec(ne);
        nsymb = nbits/bps;
        nsymb = floor(nsymb/Np)*Np + Npad;
        nbits = nsymb*bps;

        % modulator ------------------------

        tx_bits = rand(1,nbits) > 0.5;    
        tx_symb = zeros(Nc,nsymb);
        prev_tx_symb = 1;
        for s=1:nsymb
            % insert pilot every Np symbols
            if mod(s-1,8) == 0
              tx_bits(2*s-1:2*s) = [0 0];
            end
            atx_symb = qpsk_mod(tx_bits(2*s-1:2*s));
            tx_symb(1,s) = atx_symb;
        end
        % set up parallel carriers, only middle one carries data
        for c=2:Nc
          tx_symb(c,:) = tx_symb(1,:);
        end

        % channel ---------------------------

        rx_symb = tx_symb;
        ch_model = ones(Nc,nsymb).*exp(j*ch_phase);

        if hf_en

          % Simplified rate Rs simulation model that doesn't include
          % ISI, just per carrier freq-domain filtering by a single
          % complex coefficient.

          hf_model = zeros(Nc, nsymb);
          for s=1:nsymb 
            for c=1:Nc
              % middle carrier at 0 Hz, note Fs cancels
              w = 2*pi*(c-2)*Rs/Fs;
              a = path_delay_s*Fs;
              hf_model(c,s) = hf_gain*(spread1(s) + exp(-j*a*w)*spread2(s));
            end
           end
          if sim_in.ideal_phase == 1
            ch_model = ch_model .* abs(hf_model);
          else
            ch_model = ch_model .* hf_model;
          end
        end
        rx_symb = rx_symb.*ch_model;

        % variance is noise power, which is divided equally between real and
        % imag components of noise
        noise = sqrt(variance*0.5)*(randn(Nc,nsymb) + j*randn(Nc,nsymb));
        rx_symb += noise;

        % equaliser ------------------------------------------

        rx_symb_t = Ts*(0:nsymb-1);          % symbol times
        % extract pilots, filtering across freq
        if sim_in.ls_pilots
          rx_pilots = zeros(1,nsymb/Np); p = 1; local_path_delay_s = 0.004;
          A = [1 exp(j*2*pi*Rs*local_path_delay_s); 1 1; 1 exp(-j*2*pi*Rs*local_path_delay_s)];
          P = inv(A'*A)*A';
          for s=1:Np:nsymb
            h = zeros(Nc,1);
            for c=1:Nc
              h(c) = rx_symb(c,s);
            end
            g = P*h;
            rx_pilots(p) = g(1) + g(2); p++;
          end  
        else
          % weighted average across frequency
          rx_pilots = zeros(1,nsymb/Np);
          for c=1:Nc
            rx_pilots += pilot_freq_weights(c)*rx_symb(c,1:Np:nsymb);   
          end
          rx_pilots ./= sum(pilot_freq_weights);
        end
        rx_pilots_t = Ts*(0:Np:nsymb-1);     % pilot times
        rx_ch = zeros(1,nsymb);              % channel estimate of centre carrier

        % bunch of options for resampler, takes pilots and estimates channel

        if strcmp(sim_in.resampler,"nearest")
          % use nearest pilot
          for s=0:nsymb-1
            ind = round(s/Np)+1;
            ind = min(length(rx_pilots),ind); ind = max(1,ind);
            rx_ch(s+1) = rx_pilots(ind);
          end
        end
        if strcmp(sim_in.resampler,"lin2")
          % Linear interpolation between two nearest pilots, used on 700E and datac1 modes
          % 1.5dB Il AWGN, similar performance to mean4 on HF (1.5dB), trade off I guess
          rx_ch = interp1(rx_pilots_t,rx_pilots,rx_symb_t);
        end
        if strcmp(sim_in.resampler,"mean4")
          % Mean of 4 adjacent pilots, similar time duration to 700D.  Lack of HF response in resampler
          % is quite obvious, about 0.5dB IL AWGN
          filter_delay = 1.5*Np*Ts;
          h = [1 1 1 1]; h=h./sum(h);
          rx_pilots_filtered = filter(h,1,rx_pilots);
          rx_ch = interp1(rx_pilots_t-filter_delay,rx_pilots_filtered,rx_symb_t,"extrap");
        end
        if strcmp(sim_in.resampler,"sinc")
          % use sinc filter to double the sample rate of the pilots, then linear interpolation
          % note delay and B needed hand tweaking using plot_freq_response above, and run_single to adjust delay
          filter_delay = 3.5*Np*Ts;
          B=2*pi; n=(-3.5:0.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h);
          rx_pilots2 = zeros(1,2*length(rx_pilots));
          rx_pilots2(1:2:end) = rx_pilots;
          rx_pilots2_filtered = 2*filter(h,1,rx_pilots2);
          rx_pilots2_t = Ts*(0:Np/2:nsymb-1);
          rx_ch = interp1(rx_pilots2_t-filter_delay,rx_pilots2_filtered,rx_symb_t,"extrap");
       end

        % actual equalisation (just of phase)
        if sim_in.ideal_phase == 0
           for s=1:nsymb
              rx_symb(2,s) *= exp(-j*angle(rx_ch(s)));
           end
        end

        if verbose == 2
            figure(1); clf;
            subplot(211); hold on;
            plot(rx_symb_t,real(ch_model));
            plot(rx_pilots_t, real(rx_pilots),'ro');
            plot(rx_symb_t,real(rx_ch));
            hold off; axis([0 Ts*(nsymb-1) -2 2]); xlabel('time (s)'); ylabel('real');
            subplot(212); hold on;
            plot(rx_symb_t,imag(ch_model));
            plot(rx_pilots_t, imag(rx_pilots),'ro');
            plot(rx_symb_t,imag(rx_ch));
            hold off; axis([0 Ts*(nsymb-1) -2 2]); xlabel('time (symbol)'); ylabel('imag');
        end

        % extract centre carrier, remove padding at end as equaliser invalid
        rx_symb = rx_symb(2,1:end-Npad);
        nsymb -= Npad;
 
        % demodulate rx symbols to bits
        rx_bits = [];
        prev_rx_symb = 1;
        for s=1:nsymb
          arx_symb = rx_symb(s);
          two_bits = qpsk_demod(arx_symb);
          rx_bits = [rx_bits two_bits];
        end
        
        % count errors -----------------------------------------

        error_pattern = xor(tx_bits(1:end-Npad*bps), rx_bits);
        nerrors = sum(error_pattern);
        bervec(ne) = nerrors/nbits + 1E-12;
        if verbose
          printf("EbNodB: % 5.1f nbits: %7d nerrors: %5d ber: %4.3f\n", EbNodB, nbits, nerrors, bervec(ne));
          if verbose == 2
            figure(2); clf;
            plot(rx_symb*exp(j*pi/4),'+','markersize', 10);
            mx = max(abs(rx_symb));
            axis([-mx mx -mx mx]);
           end
        end
    end

    sim_out.bervec = bervec;
endfunction

function run_single(nbits = 1000,ch='awgn',EbNodB=100,resampler="lin2",ls_pilots=0)
    sim_in.pilot_freq_weights = [0 1 0];
    sim_in.verbose     = 2;
    sim_in.nbits       = nbits;
    sim_in.EbNovec     = EbNodB;
    sim_in.ch          = ch;
    sim_in.resampler   = resampler;
    sim_in.ch_phase    = 0;
    sim_in.ideal_phase = 0;
    sim_in.ls_pilots   = ls_pilots;

    sim_qpsk = ber_test(sim_in);
endfunction

function run_curves
    max_nbits = 1E5;
    sim_in.verbose = 1;
    sim_in.EbNovec = 0:10;
    sim_in.ch = "awgn";
    sim_in.hf_en   = 0;
    sim_in.resampler = "";
    sim_in.pilot_freq_weights = [0 1 0];
    sim_in.ideal_phase = 0;
    sim_in.ls_pilots = 0;

    % AWGN -----------------------------

    awgn_theory = 0.5*erfc(sqrt(10.^(sim_in.EbNovec/10)));
    sim_in.nbits  = min(max_nbits, floor(500 ./ awgn_theory));

    sim_in.resampler = "lin2"; awgn_sim_lin2 = ber_test(sim_in);
    sim_in.resampler = "mean4"; awgn_sim_mean4 = ber_test(sim_in);
    sim_in.pilot_freq_weights = [1 1 1];
    sim_in.resampler = "mean4"; awgn_sim_mean4w = ber_test(sim_in);
    sim_in.resampler = "lin2"; awgn_sim_lin2w = ber_test(sim_in);
    sim_in.ls_pilots = 1; awgn_sim_lin2ls = ber_test(sim_in);
     
    % HF -----------------------------

    hf_sim_in = sim_in; 
    hf_sim_in.EbNovec = 0:16;
    hf_sim_in.ch = "mpp";
    hf_sim.ideal_phase = 1;
    hf_sim_in.pilot_freq_weights = [0 1 0];
    hf_sim_in.ls_pilots = 0;

    EbNoLin = 10.^(hf_sim_in.EbNovec/10);
    hf_theory = 0.5.*(1-sqrt(EbNoLin./(EbNoLin+1)));

    % run long enough to get sensible results
    hf_sim_in.nbits = min(max_nbits, floor(500 ./ hf_theory));
    hf_sim_in.nbits = max(2000, floor(500 ./ hf_theory));

    hf_sim = ber_test(hf_sim_in);
    hf_sim.ideal_phase = 0;
    hf_sim_in.resampler = "lin2"; hf_sim_lin2 = ber_test(hf_sim_in);
    hf_sim_in.resampler = "mean4"; hf_sim_mean4 = ber_test(hf_sim_in);
    hf_sim_in.pilot_freq_weights = [1 1 1];
    hf_sim_in.resampler = "lin2"; hf_sim_lin2w = ber_test(hf_sim_in);
    hf_sim_in.resampler = "mean4"; hf_sim_mean4w = ber_test(hf_sim_in);

    hf_sim_in.resampler = "lin2"; hf_sim_in.ls_pilots = 1;
    hf_sim_lin2ls = ber_test(hf_sim_in);

    hf_sim_in.pilot_freq_weights = [0 1 0]; hf_sim_in.ls_pilots = 0;
    hf_sim_in.ch = "mpd";
    hf_sim_in.resampler = "lin2"; hf_sim_lin2_mpd = ber_test(hf_sim_in);
    hf_sim_in.resampler = "mean4"; hf_sim_mean4_mpd = ber_test(hf_sim_in);

    hf_sim_in.resampler = "lin2"; hf_sim_in.ls_pilots = 1;
    hf_sim_lin2ls_mpd = ber_test(hf_sim_in);

    % Plot results --------------------

    figure (3); clf;
    semilogy(sim_in.EbNovec, awgn_theory,'r+-;AWGN theory;','markersize', 10, 'linewidth', 2)
    hold on;
    semilogy(sim_in.EbNovec, awgn_sim_lin2.bervec,'bo-;AWGN sim lin2;','markersize', 10, 'linewidth', 2)
    %semilogy(sim_in.EbNovec, awgn_sim_mean4.bervec,'bx-;AWGN sim mean4;','markersize', 10, 'linewidth', 2)
    %semilogy(sim_in.EbNovec, awgn_sim_lin2w.bervec,'co-;AWGN sim lin2w;','markersize', 10, 'linewidth', 2)
    semilogy(sim_in.EbNovec, awgn_sim_mean4w.bervec,'cx-;AWGN sim mean4w;','markersize', 10, 'linewidth', 2)
    semilogy(sim_in.EbNovec, awgn_sim_lin2ls.bervec,'b*-;AWGN sim lin2ls;','markersize', 10, 'linewidth', 2)

    semilogy(hf_sim_in.EbNovec, hf_theory,'r+-;HF theory;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, hf_sim.bervec,'g+-;HF sim ideal MPP;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2.bervec,'bo-;HF sim MPP lin2;','markersize', 10, 'linewidth', 2)
    %semilogy(hf_sim_in.EbNovec, hf_sim_mean4.bervec,'bx-;HF sim MPP mean4;','markersize', 10, 'linewidth', 2)
    %semilogy(hf_sim_in.EbNovec, hf_sim_lin2w.bervec,'co-;HF sim MPP lin2w;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, hf_sim_mean4w.bervec,'cx-;HF sim MPP mean4w;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2ls.bervec,'b*-;HF sim MPP lin2ls;','markersize', 10, 'linewidth', 2)

    semilogy(hf_sim_in.EbNovec, hf_sim_lin2_mpd.bervec,'mo-;HF sim lin2 MPD;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, hf_sim_mean4_mpd.bervec,'mx-;HF sim mean4 MPD;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, hf_sim_lin2ls_mpd.bervec,'b*-;HF sim lin2ls MPD;','markersize', 10, 'linewidth', 2)

    hold off;
    xlabel('Eb/No (dB)')
    ylabel('BER')
    grid("minor")
    axis([min(hf_sim_in.EbNovec) max(hf_sim_in.EbNovec) 1E-3 1])

endfunction

