% auto_plots.m

1;

function do_plots_A(epslatex=0, png_name="")
  util;

  if epslatex, [textfontsize linewidth] = set_fonts(font_size=14); end

  load loss_1.txt
  load loss_2.txt
  load loss_3.txt
  load loss_4.txt
  load loss_5.txt
  load loss_6.txt
  load loss_7.txt
  load loss_8.txt
  figure(1); clf; 
  semilogy(loss_1,'b;1 nn1 d=20;'); 
  hold on;
  semilogy(loss_2,'g;2 nn1 zero mean;');
  semilogy(loss_3,'r;3 nn1 lr 0.05;');
  semilogy(loss_4,'c;4 nn2;');
  semilogy(loss_5,'m;5 nn2 no zero mean;');
  semilogy(loss_6,'bk;6 nn3 small layers;');
  semilogy(loss_7,'g+-;7 nn1 Adam;');
  semilogy(loss_8,'r+-;8 nn1 lr 0.05 norm;');
  hold off;
  xlabel('Epochs'); ylabel('SD dB^2'); grid;
  axis([0 length(loss_1) 0.05 10])
  
  if length(png_name)
    print("-dpng", png_name)
  end

  if epslatex, print_eps_restore("auto.tex","-S300,300",textfontsize,linewidth); end

endfunction

function do_plots_B(epslatex=0, png_name="")
  util;

  if epslatex, [textfontsize linewidth] = set_fonts(font_size=14); end

  load loss_1.txt
  load loss_2.txt
  load loss_3.txt
  load loss_4.txt
  load loss_5.txt
  load loss_6.txt
  load loss_7.txt
  load loss_8.txt
  figure(1); clf; 
  semilogy(loss_1,'b;1 nn1 d=20;'); 
  hold on;
  semilogy(loss_2,'g;2 nn1 zero mean;');
  semilogy(loss_3,'r;3 nn1 lr 0.05;');
  semilogy(loss_4,'c;4 nn2;');
  semilogy(loss_5,'m;5 nn2 no zero mean;');
  semilogy(loss_6,'bk;6 nn3 small layers;');
  semilogy(loss_7,'g+-;7 nn1 Adam;');
  semilogy(loss_8,'r+-;8 nn1 lr 0.05 norm;');
  hold off;
  xlabel('Epochs'); ylabel('SD dB^2'); grid;
  axis([0 length(loss_1) 0.05 10])
  
  if length(png_name)
    print("-dpng", png_name)
  end

  if epslatex, print_eps_restore("auto.tex","-S300,300",textfontsize,linewidth); end

endfunction


