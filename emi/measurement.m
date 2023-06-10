% measurements.m
% Plot May 2023 EMI measurements

function measurement()
    Gfeed = -1; Gsplitter = -3.5;
    RBW = 3000; RBW_dB = 10*log10(RBW);

    forty = load("forty.txt"); twenty = load("twenty.txt");
    forty = forty - (Gfeed + Gsplitter);
    twenty = twenty - (Gfeed + Gsplitter);
    
    % summarise clusters
    A = mean(forty);
    pkg load statistics;
    [ind,cent] = kmeans(twenty,2);
    B = min(cent); C = max(cent);
    A_No = A - RBW_dB; B_No = B - RBW_dB; C_No = C - RBW_dB; 
    
    h=figure(1); clf; subplot(211); hold on;
    plot(7.1*ones(1,length(forty)), forty,'b+'); plot(14.2*ones(1,length(twenty)), twenty,'b+');
    plot([7.1 14.2 14.2],[A B C],'ro', 'markersize',10);
    text([7.1 14.2 14.2]+0.3,[A B C],{"A","B","C"})
    grid; xlabel('Freq (MHz)'); ylabel('dBm (3kHz)');
    axis([6 15 -120 -70]); legend('off')
    hold off;

    subplot(212);
    hold on;
    plot(7.1*ones(1,length(forty)), forty-RBW_dB,'b+'); plot(14.2*ones(1,length(twenty)), twenty-RBW_dB,'b+');
    plot([7.1 14.2 14.2],[A_No B_No C_No],'ro', 'markersize',10);
    text([7.1 14.2 14.2]+0.3,[A_No B_No C_No],{"A","B","C"})

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
    printf("nominal: %4.0f %4.0f %3.1f    %2.0f     %2.0f\n", A, A_No, A_No/6+26.7, Fa_p372_forty, A_No+174);
    printf("low:     %4.0f %4.0f %3.1f    %2.0f     %2.0f\n", B, B_No, B_No/6+26.7, Fa_p372_twenty, B_No+174);
    printf("high:    %4.0f %4.0f %3.1f    %2.0f     %2.0f\n", C, C_No, C_No/6+26.7, Fa_p372_twenty, C_No+174);    
endfunction

