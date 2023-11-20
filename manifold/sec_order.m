% sec_order.m
% Check a few expressions 2nd order expressions in manifold.tex

% complex valued function of form:
%   x(n) = exp(-alpha)*exp(j*omega_f*n), n>0

function sec_order(epslatex=0)

    util;

    % alpha is a time constant, small alpha means slow decay, narrow BW
    alpha = 0.2; omega_f = pi/4;
    omega = 0:0.01*pi:pi;
    H = 1./(1 - exp(-alpha)*exp(j*(omega_f-omega)));

    % To compare, lets construct a second order system using pole at omega_f, 
    % distance beta from the origin
    beta = exp(-alpha);
    H1 = freqz(1,[1 -beta*exp(j*omega_f)],omega);

    figure(1); clf; 
    subplot(211); plot(omega,20*log10(abs(H))); hold on; plot(omega,20*log10(abs(H1))); 
    grid; hold off;
    subplot(212); plot(omega,angle(H)); hold on; plot(omega,angle(H1)); grid; hold off;

    % Check half power expressions
    H_peak = 1/(1-exp(-alpha));
    subplot(211); hold on; 
    plot([omega_f-alpha omega_f omega_f+alpha],20*log10(H_peak*[1/sqrt(2) 1 1/sqrt(2)]),
        'gx','markersize', 10);
    hold off;
    subplot(212); hold on; 
    phase_high = -atan(exp(-alpha)*sin(alpha)/(1-exp(-alpha)*cos(alpha)));
    plot([omega_f-alpha omega_f omega_f+alpha],[-phase_high 0 phase_high],
        'gx','markersize', 10);
    hold off;

    if epslatex, [textfontsize linewidth] = set_fonts(font_size=12); end

    figure(2); clf;

    n=0:99; alphalist = [0.1 0.05 0.01]; col_map = {'b','g','r'};
    for i=1:length(alphalist)
        alpha = alphalist(i);
        leg = sprintf('%c-;alpha %3.2f;',col_map{i},alpha);
        x = exp(-alpha*n).*exp(j*omega_f*n);
        H = 1./(1 - exp(-alpha)*exp(j*(omega_f-omega)));
        subplot(311); plot(n,x,leg); hold on; grid; legend('boxoff');
        subplot(312);  plot(omega,20*log10(abs(H)),leg); hold on; grid; legend('boxoff'); ylabel('$|H(\omega)|$ dB')
        subplot(313);  plot(omega,angle(H),leg); hold on; grid; legend('boxoff'); ylabel('$arg\{H(\omega)\}$');
    end

    if epslatex, print_eps_restore("sec_order.tex","-S300,300",textfontsize,linewidth); end
endfunction
