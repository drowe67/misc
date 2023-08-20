% phase_resampler.m
% David Rowe Aug 2023
%
% Phase resampler tests

function phase_resampler
    randn('seed',1);

    pkg load signal
    % Doppler bandwidth (Hz).  Doppler signal is complex so
    % this is a total bandwidth of Fd, sperad Fd/2 either side of 0 Hz
    Fd = 1;          
    Fs = 8000;
    N  = Fs*10;

    [d1 states] = doppler_spread(Fd, Fs, N);

    % Decimate down to a sample rate of 2Fd
    Fs2 = 6.25*Fd;
    M = Fs/Fs2
    assert (M == floor(M));
    n2 = 1:M:length(d1);
    d2 = d1(n2);

    % Try resampling
    B=pi/2; n=(-2:2); h=(B/(2*pi))*sinc(n*B/(2*pi));
    h
    zero_padded = zeros(1,2*length(d2));
    zero_padded(1:2:end) = d2;
    d3 = 2*filter(h,1,[zero_padded 0 0]);
    d3 = d3(3:end);
    length(d3)
    n3 = ((1:length(d3))-1)*M/2;

    % Some plots

    d4 = states.spread_lowFs;
    M = states.M;
    figure(1); clf;
    subplot(211); 
    plot(real(d1)); hold on; 
    stem((1:M:length(d4)*M),real(d4));
    hold off;
    ylabel('real');
    title('Doppler Function lowFs');
    subplot(212); 
    plot(imag(d1)); hold on; 
    stem((1:M:length(d4)*M),imag(d4));
    hold off;
    ylabel('imag');
    xlabel('Time (samples)')

    figure(2); clf;
    subplot(211); 
    plot(real(d1)); ylabel('real');
    hold on; 
    stem(n2,real(d2)); 
    stem(n3,real(d3)); 
    hold off;
    subplot(212); 
    plot(imag(d1)); xlabel('imag');
    hold on; stem(n2,imag(d2)); hold off;
    title('Doppler Function');
    xlabel('Time (samples)')

endfunction