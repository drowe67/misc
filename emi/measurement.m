% measurements.m
% Plot EMI measurements

function measurement()

    a = load("forty.txt"); b = load("twenty.txt");
    h=figure(1); clf; subplot(211); hold on;
    plot(a,'bo;7.1 MHz;'); plot(b,'go;14.2 MHz;');
    grid; ylabel('dBm (3kHz)');
    l = max([length(a) length(b)]);
    axis([1 l -120 -70]); legend('boxoff')
    hold off;

    subplot(212);
    hold on;
    plot(a-10*log10(3000),'bo;7.1 MHz;'); plot(b-10*log10(3000),'go;14.2 MHz;');
    grid; ylabel('dBm/Hz');
    axis([1 l -150 -100]); legend('boxoff')
    hold off;

    fn = "measurement.tex";
    print(fn,"-depslatex","-F:10","-S300,300");
    printf("printing... %s\n", fn);
 
endfunction

