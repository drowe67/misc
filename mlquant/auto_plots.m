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

  load loss_B1.txt
  load loss_B2.txt
  load loss_B3.txt
  load loss_B4.txt
  load loss_B5.txt
  load loss_B6.txt
  load loss_B7.txt
  load loss_B8.txt
  load loss_B9.txt
  load loss_B10.txt
  figure(1); clf; 
  semilogy(loss_B1,'bo-;1 nn1 d=20;'); 
  hold on;
  semilogy(loss_B2,'b+-;2 nn1 d=20;');
  semilogy(loss_B3,'bx-;3 nn1 d=20;');
  semilogy(loss_B4,'go-;4 nn4 d=20;');
  semilogy(loss_B5,'ro-;5 nn1 d=15;');
  semilogy(loss_B6,'rx-;6 nn1 d=10;');
  semilogy(loss_B7,'gx-;7 nn4 d=15;');
  semilogy(loss_B8,'g+-;8 nn4 d=15;');
  semilogy(loss_B9,'g*-;9 nn4 d=15;');
  semilogy(loss_B10,'c+-;9 nn4 tanh d=15;');
  hold off;
  xlabel('Epochs'); ylabel('SD dB^2'); grid;
  axis([0 length(loss_B1) 0.05 10])
  
  if length(png_name)
    print("-dpng", png_name)
  end

  if epslatex, print_eps_restore("auto.tex","-S300,300",textfontsize,linewidth); end

endfunction

function do_plots_C(epslatex=0, png_name="")
  util;

  if epslatex, [textfontsize linewidth] = set_fonts(font_size=14); end

  load loss_C1.txt
  load loss_C2.txt
  load loss_C3.txt
  load loss_C4.txt
  load loss_C5.txt
  load loss_C6.txt
  load loss_C7.txt
  #load loss_C8.txt
  #load loss_C9.txt
  #load loss_C10.txt
  figure(1); clf; 
  semilogy(loss_C1,'b+-;1 nn1 d=15 (no noise inj);'); 
  hold on;
  semilogy(loss_C2,'g+-;2 nn5 d=15 1E-3;');
  semilogy(loss_C3,'gx-;3 nn5 d=15 1E-3;');
  semilogy(loss_C4,'r+-;4 nn5 d=10 1E-3;');
  semilogy(loss_C5,'rx-;5 nn5 d=10 1E-3;');
  semilogy(loss_C6,'c+-;6 nn5 d=20 1E-3;');
  semilogy(loss_C7,'cx-;7 nn4 d=20 3E-3;');
  #semilogy(loss_C8,'g+-;8 nn4 d=15;');
  #semilogy(loss_C9,'g*-;9 nn4 d=15;');
  #semilogy(loss_C10,'c+-;9 nn4 tanh d=15;');
  hold off;
  xlabel('Epochs'); ylabel('SD dB^2'); grid;
  axis([0 length(loss_C1) 0.05 10])
  
  if length(png_name)
    print("-dpng", png_name)
  end

  if epslatex, print_eps_restore("auto.tex","-S300,300",textfontsize,linewidth); end

endfunction

% DCT curves
function do_plots_D(epslatex=0, png_name="")
  util;

  if epslatex, [textfontsize linewidth] = set_fonts(font_size=14); end

  load loss_D1.txt
  load loss_D2.txt
  load loss_D3.txt
  load loss_D4.txt
  load loss_D5.txt
  #load loss_C6.txt
  #load loss_C7.txt
  #load loss_C8.txt
  #load loss_C9.txt
  #load loss_C10.txt
  figure(1); clf; 
  semilogy(loss_D1,'b+-;1 nn1 d=20 (no noise inj);'); 
  hold on;
  semilogy(loss_D2,'bx-;2 nn1 d=10 (no noise inj);');
  semilogy(loss_D3,'g+-;3 nn5 d=20 1E-3;');
  semilogy(loss_D4,'gx-;4 nn5 d=10 1E-3;');
  semilogy(loss_D5,'rx-;5 nn5 d=10 1E-3 norm;');
  #semilogy(loss_C6,'c+-;6 nn5 d=20 1E-3;');
  #semilogy(loss_C7,'cx-;7 nn4 d=20 3E-3;');
  #semilogy(loss_C8,'g+-;8 nn4 d=15;');
  #semilogy(loss_C9,'g*-;9 nn4 d=15;');
  #semilogy(loss_C10,'c+-;9 nn4 tanh d=15;');
  hold off;
  xlabel('Epochs'); ylabel('SD dB^2'); grid;
  axis([0 length(loss_D1) 0.05 10])
  
  if length(png_name)
    print("-dpng", png_name)
  end

  if epslatex, print_eps_restore("auto.tex","-S300,300",textfontsize,linewidth); end

endfunction

# ae3 and ae2, large epochs
function do_plots_E(epslatex=0, png_name="")
  util;

  if epslatex, [textfontsize linewidth] = set_fonts(font_size=14); end

  load loss_E1.txt
  load loss_E2.txt
  load loss_E3.txt
  load loss_E4.txt
  load loss_E5.txt
  #load loss_E6.txt
  #load loss_C7.txt
  #load loss_C8.txt
  #load loss_C9.txt
  #load loss_C10.txt
  figure(1); clf; 
  semilogy(loss_E1,'b-;1 ae3 nn1 d=10 lr 1;'); 
  hold on;
  semilogy(loss_E2,'b-;2 ae3 nn1 d=10 lr 1;');
  semilogy(loss_E3,'g-;3 ae3 nn1 d=10 lr 0.5;');
  semilogy(loss_E4,'r-;4 ae2 nn1 d=10;');
  semilogy(loss_E5,'r-;5 ae2 nn2 d=10;');
  #semilogy(loss_E6,'c+-;6 ae3 nn1 lr 1 1E-3;');
  #semilogy(loss_C7,co-;7 nn4 d=20 3E-3;');
  #semilogy(loss_C8,'g+-;8 nn4 d=15;');
  #semilogy(loss_C9,'g*-;9 nn4 d=15;');
  #semilogy(loss_C10,'c+-;9 nn4 tanh d=15;');
  hold off;
  xlabel('Epochs'); grid;
  axis([0 length(loss_E1) 0.5E-3 1E-2])
  
  if length(png_name)
    print("-dpng", png_name)
  end

  if epslatex, print_eps_restore("auto_loss.tex","-S350,250",textfontsize,linewidth); end

endfunction



