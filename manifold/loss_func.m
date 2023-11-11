% loss_func.m
% Exploring proposed loss function for ML model

function loss_func(y_fn,gamma=0.5,frame=165,epslatex=0)
    rand('seed',1);
    newamp_700c; util;
    Fs = 8000; Lhigh = 80; Nb=20;
    F0high = (Fs/2)/Lhigh;
    rate_Lhigh_sample_freqs_kHz = (F0high:F0high:(Lhigh-1)*F0high)/1000;

    % precompute filters at rate Lhigh. Note range of harmonics is 1:Lhigh-1, as
    % we don't use Lhigh-th harmonic as it's on Fs/2

    h = zeros(Lhigh, Lhigh);
    F0high = (Fs/2)/Lhigh;
    for m=1:Lhigh-1
        h(m,:) = generate_filter(m,F0high,Lhigh,Nb);
        plot((1:Lhigh-1)*F0high,h(m,1:Lhigh-1));
    end

    y = load_f32(y_fn,Lhigh-1);
    ydB = y(frame,:);
    e = sum(10.^(ydB/10));
    ydB -= 10*log10(e);

    % filter to create smoothed ydB_hat
    ydB_hat = zeros(1,Lhigh-1);
    for m=1:Lhigh-1
      Am_rate_Lhigh = 10.^(ydB/20);
      Y = sum(Am_rate_Lhigh.^2 .* h(m,1:Lhigh-1));
      ydB_hat(m) = 10*log10(Y);
    end
    %ydB_hat += 10;

    %ydB_hat = ydB + 6*(rand(1,Lhigh-1)-0.5) + 0;
    %ydB_hat = ydB + 6*(rand(1,Lhigh-1)-0.5) + 0;

    ylin = 10.^(ydB/10); ylin_hat = 10.^(ydB_hat/10);
    wdB = -(1-gamma)*ydB; wlin = 10.^(wdB/10);
    figure(1); clf; subplot(211); hold on;
    plot(rate_Lhigh_sample_freqs_kHz,ydB,'b;y;')
    plot(rate_Lhigh_sample_freqs_kHz,ydB_hat,'g;y hat;')
    axis([0 4 -60 0])
    hold off;
    subplot(212); 
    plot(rate_Lhigh_sample_freqs_kHz,(ydB-ydB_hat).^2,'b;squared error dB*dB;')
    
    figure(2); clf; subplot(311); hold on;
    plot(rate_Lhigh_sample_freqs_kHz,ylin,'b;ylin;')
    plot(rate_Lhigh_sample_freqs_kHz,ylin_hat,'g;ylin hat;')
    hold off;
    subplot(312); plot(rate_Lhigh_sample_freqs_kHz,(ylin-ylin_hat).^2,'b;squared error;')
    subplot(313); plot(rate_Lhigh_sample_freqs_kHz,((ylin-ylin_hat).*wlin).^2,'g;weighted squared error;')

    figure(3); clf; subplot(211);
    plot(rate_Lhigh_sample_freqs_kHz,wdB)
    subplot(212);
    plot(rate_Lhigh_sample_freqs_kHz,wlin)
   
 
endfunction
