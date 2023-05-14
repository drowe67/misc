% blanker.m
% Plot spectrum for noise blanker output

1;

function plot_blanker(f_sig_hz, amp, f_max_hz, pulse_rate_hz, pulse_duty_cycle, c, leg='')
  centre_amp = (1-pulse_duty_cycle)*amp;
  N = round(f_max_hz/pulse_rate_hz);
  sideband_amp = amp*pulse_duty_cycle*sinc(pi*(1:N)*pulse_duty_cycle);
  amps = 20*log10([fliplr(sideband_amp) centre_amp sideband_amp]);
  freqs = f_sig_hz + [-fliplr(pulse_rate_hz*(1:N)) 0 pulse_rate_hz*(1:N)];
  stem(freqs,amps,c,'basevalue',-100)
  if length(leg)
    legend(leg);
  end
endfunction

bandwidth_hz = 20E3;  f_max_hz = bandwidth_hz/2;
figure(1); clf; hold on;
pulse_period = 20E-3; pulse_length = 100E-6;
plot_blanker(1000, 1, f_max_hz, 1/pulse_period, pulse_length/pulse_period, 'b')
grid;
axis([-f_max_hz f_max_hz -100 0]);
hold off;

figure(2); clf; hold on;
pulse_period = 1/5000; pulse_length = 100E-6;
plot_blanker(1000, 1, f_max_hz, 1/pulse_period, pulse_length/pulse_period, 'b')
grid;
axis([-f_max_hz f_max_hz -100 0]);
hold off;