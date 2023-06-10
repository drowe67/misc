% measurements.m
% Plot May 2023 EMI measurements

function measurement()
    Gfeed = -1; Gsplitter = -3.5;
    RBW = 3000; RBW_dB = 10*log10(RBW);

    a = load("forty.txt"); b = load("twenty.txt");
    a = a - (Gfeed + Gsplitter);
    b = b - (Gfeed + Gsplitter);
    
    % summarise clusters
    nominal = mean(a);
    pkg load statistics;
    [ind,cent] = kmeans(b,2);
    low = min(cent); high = max(cent);
    nominal_No = nominal - RBW_dB; low_No = low - RBW_dB; high_No = high - RBW_dB; 
    
    h=figure(1); clf; subplot(211); hold on;
    plot(7.1*ones(1,length(a)), a,'b+'); plot(14.2*ones(1,length(b)), b,'b+');
    plot([7.1 14.2 14.2],[nominal low high],'ro', 'markersize',10);
    grid; xlabel('Freq (MHz)'); ylabel('dBm (3kHz)');
    axis([6 15 -120 -70]); legend('off')
    hold off;

    subplot(212);
    hold on;
    plot(7.1*ones(1,length(a)), a-RBW_dB,'b+'); plot(14.2*ones(1,length(b)), b-RBW_dB,'b+');
    plot([7.1 14.2 14.2],[nominal_No low_No high_No],'ro', 'markersize',10);
    
    c = 72.5; d = 27.7; f = 7:15;
    Fa = c - d *log10(f);
    plot(f,Fa-174,'g--;P.372 Residential;');

    grid; xlabel('Freq (MHz)'); ylabel('dBm/Hz');
    axis([6 15 -150 -110]); legend('boxoff')
    hold off;

    fn = "measurement.tex";
    print(fn,"-depslatex","-S300,300");
    printf("printing... %s\n", fn);

   % Fa from P.372-16 Sect 6.1, residential
    Fa_p372_forty = 72.5 - 27.7*log10(7.1); Fa_p372_twenty = 72.5 - 27.7*log10(14.2);
    printf("         3kHz  1Hz Sunit FaP372 Fa\n");
    printf("nominal: %4.0f %4.0f %3.1f    %2.0f     %2.0f\n", nominal, nominal_No, nominal_No/6+26.7, Fa_p372_forty, nominal_No+174);
    printf("low:     %4.0f %4.0f %3.1f    %2.0f     %2.0f\n", low, low_No, low_No/6+26.7, Fa_p372_twenty, low_No+174);
    printf("high:    %4.0f %4.0f %3.1f    %2.0f     %2.0f\n", high, high_No, high_No/6+26.7, Fa_p372_twenty, high_No+174);    
endfunction

