% blanker.m
% Plot spectrum for noise blanker output

function blanker(epslatex=0)
    if epslatex
        textfontsize = get(0,"defaulttextfontsize");
        linewidth = get(0,"defaultlinelinewidth");
        set(0, "defaulttextfontsize", 10);
        set(0, "defaultaxesfontsize", 10);
        set(0, "defaultlinelinewidth", 0.5);
    end

    bandwidth_hz = 20E3;  f_max_hz = bandwidth_hz/2;
    figure(1); clf; subplot(211); hold on;
    pulse_period = 20E-3; pulse_length = 100E-6;
    plot_blanker(0, 1, f_max_hz, 1/pulse_period, pulse_length, 'b')
    grid; ylabel('dB')
    axis([-f_max_hz f_max_hz -100 0]);
    hold off;

    subplot(212);
    hold on;
    pulse_period = 1/5000; pulse_length = 100E-6;
    plot_blanker(0, 1, f_max_hz, 1/pulse_period, pulse_length, 'b')
    grid; ylabel('dB')
    axis([-f_max_hz f_max_hz -100 0]);
    hold off;

    if epslatex
        fn = "blanker.tex";
        print(fn,"-depslatex","-S300,300");
        printf("printing... %s\n", fn);
        set(0, "defaulttextfontsize", textfontsize);
        set(0, "defaultaxesfontsize", textfontsize);
        set(0, "defaultlinelinewidth", linewidth);
    end

    bandwidth_hz = 1E6;  f_max_hz = bandwidth_hz/2;
    figure(2); clf;
    hold on;
    pulse_period = 1/100E3; pulse_length = 5E-6;
    plot_blanker(0, 1, f_max_hz, 1/pulse_period, pulse_length, 'b')
    grid; ylabel('dB')
    axis([-f_max_hz f_max_hz -100 0]);
    hold off;
   

endfunction

function plot_blanker(f_sig_hz, amp, f_max_hz, pulse_rate_hz, pulse_length_s, c='b')
    pulse_duty_cycle = pulse_length_s*pulse_rate_hz;
    centre_amp = (1-pulse_duty_cycle)*amp;
    N = round(f_max_hz/pulse_rate_hz);
    sideband_amp = amp*pulse_duty_cycle*sinc(pi*(1:N)*pulse_duty_cycle);
    amps = 20*log10([fliplr(sideband_amp) centre_amp sideband_amp]);
    freqs = f_sig_hz + [-fliplr(pulse_rate_hz*(1:N)) 0 pulse_rate_hz*(1:N)];
    stem(freqs,amps,c,'basevalue',-100,'markersize',3');

    if pulse_rate_hz > 1000
      leg = sprintf("R=%dkHz T=%3.0fuS",pulse_rate_hz/1000, pulse_length_s/1E-6);
      legend('location','east')
    else
      leg = sprintf("R=%dHz T=%3.0fuS",pulse_rate_hz, pulse_length_s/1E-6);
      legend('location','northeast')
    end
    legend(leg); legend('boxoff'); legend('left');

    plot([-3000 -3000],[0 -100],'g'); plot([3000 3000],[0 -100],'g')

endfunction

