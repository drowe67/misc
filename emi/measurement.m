% measurements.m
% Plot May 2023 EMI measurements

function measurement()
    Gfeed = -1; Gsplitter = -3.5;
    RBW = 3000; RBW_dB = 10*log10(RBW);

    a = load("forty.txt"); b = load("twenty.txt");
    a = a - (Gfeed + Gsplitter);
    b = b - (Gfeed + Gsplitter);
    
    h=figure(1); clf; subplot(211); hold on;
    plot(a,'bo;7.1 MHz;'); plot(b,'go;14.2 MHz;');
    grid; ylabel('dBm (3kHz)');
    l = max([length(a) length(b)]);
    axis([1 l -120 -70]); legend('boxoff')
    hold off;

    subplot(212);
    hold on;
    plot(a-RBW_dB,'bo;7.1 MHz;'); plot(b-RBW_dB,'go;14.2 MHz;');
    grid; ylabel('dBm/Hz');
    axis([1 l -150 -100]); legend('boxoff')
    hold off;

    fn = "measurement.tex";
    print(fn,"-depslatex","-S300,300");
    printf("printing... %s\n", fn);

    % summarise clusters
    nominal = mean(a);
    pkg load statistics;
    [ind,cent] = kmeans(b,2);
    low = min(cent); high = max(cent);
    nominal_No = nominal - RBW_dB; low_No = low - RBW_dB; high_No = high - RBW_dB; 
    printf("         3kHz  1Hz Sunit Fa\n");
    printf("nominal: %4.0f %4.0f %2.0f    %2.0f\n", nominal, nominal_No, nominal_No/6+27, nominal_No+174);
    printf("low:     %4.0f %4.0f %2.0f    %2.0f\n", low, low - RBW_dB, low_No/6+27, low_No+174);
    printf("high:    %4.0f %4.0f %2.0f    %2.0f\n", high, high - RBW_dB, high_No/6+27, high_No+174);    
endfunction

