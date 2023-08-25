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
    figure(2); clf; hold on;
    h = [1 1 1 1]/4; [H w] = freqz(h); plot(w,20*log10(H),'b;[1 1 1 1];');
    B=pi/2; n=(-1.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h); [H w] = freqz(h); plot(w,20*log10(H),'g;sinc pi/2;');
    B=pi; n=(-1.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h); [H w] = freqz(h); plot(w,20*log10(H),'r;sinc pi;');
    B=pi; n=(-3.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h); [H w] = freqz(h); plot(w,20*log10(H),'c;6 tap pi;');
    axis([0 pi -20 3]); grid; title('4 sample resamplers freq resp')
end

% Rate Rs modem simulation model -------------------------------------------------------

#{
    TODO
    [X] BER counting with random QPSK symbols
    [ ] with inserted pilots at Np
    [X] additive noise
    [ ] sample Doppler function at symbol rate
    [ ] estimate channel from pilots using chosen filter function
    [ ] equalise using channel estimate
    [ ] plot curves from different resamplers
    [ ] plot curves from different Doppler rates
    [ ] extend to use pilots from carriers either side, so pilots with slightly different channel
    [ ] extend to high bandwidth, high SNR fading
    [ ] output curves and write up
#}

function sim_out = ber_test(sim_in)
    rand('seed',1);
    randn('seed',1);

    bps     = 2;     % two bits/symbol for QPSK
    Rs      = 50;    % symbol rate (needed for HF model)
    Np      = 8;     % one pilot every Np symbols

    verbose = sim_in.verbose;
    EbNovec = sim_in.EbNovec;
    hf_en   = sim_in.hf_en;
    nbitsvec = sim_in.nbits;
    nsymb = max(nbitsvec)/bps;
    ch_phase = 0;
    if isfield(sim_in,"ch_phase")
      ch_phase = sim_in.ch_phase;
    end

    % init HF model

    if hf_en
      % some typical values

      dopplerSpreadHz = 1.0; path_delay = 1E-3*Rs;

      [spread1 st] = doppler_spread(dopplerSpreadHz, Rs, nsymb);
      spread2 = doppler_spread(dopplerSpreadHz, Rs, nsymb);

      % normalise power through HF channel
      hf_gain = 1.0/sqrt(var(spread1)+var(spread2));
    end

    for ne = 1:length(EbNovec)

        % work out noise power -------------

        EbNodB = EbNovec(ne);
        EsNodB = EbNodB + 10*log10(bps);
        EsNo = 10^(EsNodB/10);
        variance = 1/EsNo;
        nbits = nbitsvec(ne);
        Nsymb = nbits/bps;

        % modulator ------------------------

        tx_bits = rand(1,nbits) > 0.5;    
        tx_symb = [];
        prev_tx_symb = 1;
        for s=1:nsymb
            % insert pilot every Np symbols
            if mod(s-1,8) == 0
              tx_bits(2*s-1:2*s) = [0 0];
            end
            atx_symb = qpsk_mod(tx_bits(2*s-1:2*s));
            tx_symb = [tx_symb atx_symb];
        end

        % channel ---------------------------

        rx_symb = tx_symb;
        ch_model = ones(1,nsymb).*exp(j*ch_phase);

        if hf_en

          % Simplified rate Rs simulation model that doesn't include
          % ISI, just per carrier freq-domain filtering by a single
          % complex coefficient.

          hf_model = zeros(1, nsymb);
          for s=1:nsymb 
            hf_model(s) = hf_gain*(spread1(s) + exp(-j*path_delay)*spread2(s));
            rx_symb(s) = rx_symb(s).*hf_model(s);
          end
          ch_model = ch_model .* hf_model;
        end
        rx_symb = rx_symb.*ch_model;

        % variance is noise power, which is divided equally between real and
        % imag components of noise
        noise = sqrt(variance*0.5)*(randn(1,nsymb) + j*randn(1,nsymb));
        rx_symb += noise;

        % demodulator ------------------------------------------

        % equalise using pilots
        rx_pilots = rx_symb(1:Np:nsymb);

        % TODO: params for each resampler, make resampler a sim_in choice
        if strcmp(sim_in.resampler,"sinc")
          filter_delay = -1.5;
          B=pi; n=(-1.5:1.5); h=(B/(2*pi))*sinc(n*B/(2*pi)); h=h./sum(h);
          rx_pilots_filtered = filter(h,1,rx_pilots);
        end
        if strcmp(sim_in.resampler,"nearest")
          filter_delay = 0;
          rx_pilots_filtered = rx_pilots;
        end

        for s=1:nsymb
          ind = floor(s/Np)+1;
          ind = min(length(rx_pilots_filtered),ind); ind = max(1,ind);
          printf("s: %d ind: %d\n",s,ind)
          rx_symb(s) *= exp(-j*angle(rx_pilots_filtered(ind)));
          %rx_symb(s) *= exp(-j*angle(hf_model(s)));
       end

        if verbose == 2
            figure(1); clf;
            subplot(211); hold on;
            plot(real(ch_model));
            stem((1:Np:nsymb),real(rx_pilots));
            stem((1:Np:nsymb) + filter_delay*Np,real(rx_pilots_filtered));
            hold off; axis([0 nsymb -2 2]); xlabel('time (symbol)'); ylabel('real');
            subplot(212); hold on;
            plot(imag(ch_model));
            stem((1:Np:nsymb),imag(rx_pilots));
            hold off; axis([0 nsymb -2 2]); xlabel('time (symbol)'); ylabel('imag');
        end

        % demodulate rx symbols to bits
        rx_bits = [];
        prev_rx_symb = 1;
        for s=1:nsymb
          arx_symb = rx_symb(s);
          two_bits = qpsk_demod(arx_symb);
          rx_bits = [rx_bits two_bits];
        end
        
        % count errors -----------------------------------------

        error_pattern = xor(tx_bits, rx_bits);
        nerrors = sum(error_pattern);
        bervec(ne) = nerrors/nbits;
        if verbose
          printf("EbNodB: % 4.1f nbits: %5d nerrors: %5d ber: %4.3f\n", EbNodB, nbits, nerrors, bervec(ne));
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

function run_single(nbits = 1000,EbNodB=100)
    sim_in.verbose   = 2;
    sim_in.nbits     = nbits;
    sim_in.EbNovec   = EbNodB;
    sim_in.hf_en     = 0;
    sim_in.resampler = "nearest";
    sim_in.ch_phase = pi/4;
 
    sim_qpsk = ber_test(sim_in);
endfunction

function run_curves
    max_nbits = 1E5;
    sim_in.verbose = 1;
    sim_in.EbNovec = 0:10;
    sim_in.hf_en   = 0;
 
    % AWGN -----------------------------

    ber_awgn_theory = 0.5*erfc(sqrt(10.^(sim_in.EbNovec/10)));
    sim_in.nbits    = min(max_nbits, floor(500 ./ ber_awgn_theory));

    sim_qpsk = ber_test(sim_in);
      
    % HF -----------------------------

    hf_sim_in = sim_in; hf_sim_in.dqpsk = 0; hf_sim_in.hf_en = 1;
    hf_sim_in.EbNovec = 0:16;

    EbNoLin = 10.^(hf_sim_in.EbNovec/10);
    ber_hf_theory = 0.5.*(1-sqrt(EbNoLin./(EbNoLin+1)));

    hf_sim_in.nbits = min(max_nbits, floor(500 ./ ber_hf_theory));
    sim_qpsk_hf = ber_test(hf_sim_in);

    % Plot results --------------------

    figure (3, 'position', [400, 10, 600, 400]); clf;
    semilogy(sim_in.EbNovec, ber_awgn_theory,'r+-;QPSK AWGN theory;','markersize', 10, 'linewidth', 2)
    hold on;
    semilogy(sim_in.EbNovec, sim_qpsk.bervec,'g+-;QPSK AWGN simulated;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, ber_hf_theory,'r+-;QPSK HF theory;','markersize', 10, 'linewidth', 2)
    semilogy(hf_sim_in.EbNovec, sim_qpsk_hf.bervec,'g+-;QPSK HF simulated;','markersize', 10, 'linewidth', 2)
    hold off;
    xlabel('Eb/No (dB)')
    ylabel('BER')
    grid("minor")
    axis([min(hf_sim_in.EbNovec) max(hf_sim_in.EbNovec) 1E-3 1])

endfunction

