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
    
    % init HF model

    if hf_en
      % some typical values

      dopplerSpreadHz = 1.0; path_delay = 1E-3*Rs;

      nsymb = max(nbitsvec)/bps;
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
            atx_symb = qpsk_mod(tx_bits(2*s-1:2*s));
            tx_symb = [tx_symb atx_symb];
        end
        % every Np symbols is a pilot
        tx_symb(1:Np:end) = 1;

        % channel ---------------------------

        rx_symb = tx_symb;

        if hf_en

          % simplified rate Rs simulation model that doesn't include
          % ISI, just freq filtering.  We assume perfect phase estimation
          % so it's just amplitude distortion.

          hf_model = zeros(1, nsymb);
          for s=1:nsymb 
            hf_model(s) = hf_gain*(spread1(s) + exp(-j*path_delay)*spread2(s));
            %TODO: this is amplitude only, include phase
            %hf_model     = abs(hf_model1(s));
            rx_symb(s) = rx_symb(s).*hf_model(s);
          end
        end

        % variance is noise power, which is divided equally between real and
        % imag components of noise
        noise = sqrt(variance*0.5)*(randn(1,nsymb) + j*randn(1,nsymb));
        rx_symb += noise;

        % demodulator ------------------------------------------

        % equalise using pilots
        rx_pilots = rx_symb(1:Np:nsymb);

        if verbose == 2
            figure(1); clf;
            subplot(211); hold on;
            plot(real(hf_model));
            stem((1:Np:nsymb),real(rx_symb(1:Np:nsymb)));
            hold off; axis([0 nsymb -4 4]); xlabel('time (s)'); ylabel('real');
            subplot(212); hold on;
            plot(imag(hf_model));
            stem((1:Np:nsymb),imag(rx_symb(1:Np:nsymb)));
            hold off; axis([0 nsymb -4 4]); xlabel('time (s)'); ylabel('imag');
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
            if sim_in.diversity && sim_in.hf_en 
              figure(3);
              plot(1:nsymb, abs(hf_model1), 1:nsymb, abs(hf_model2), 'linewidth', 2);
            end
          end
        end
    end

    sim_out.bervec = bervec;
endfunction

function run_single(nbits = 1000,EbNodB=100)
    sim_in.verbose   = 2;
    sim_in.nbits     = nbits;
    sim_in.EbNovec   = EbNodB;
    sim_in.hf_en     = 1;
    sim_in.diversity = 0;

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

