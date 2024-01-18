% mlquant_plots.m

1;

function uniform_vq(epslatex=0)
  util;

  if epslatex, [textfontsize linewidth] = set_fonts(font_size=14); end

  uniform_bits = [10 20 30 40 50];
  uniform_var  = [0.0833 0.0208 0.0052 0.0013 0.0003];
  
  load vq_train_var1.txt
  load vq_train_var2.txt
  load vq_train_var3.txt
  
  vq_bits = [log2(vq_train_var1(:,1)); 12+log2(vq_train_var1(:,1));  24+log2(vq_train_var1(:,1))];
  vq_var  = [vq_train_var1(:,2); vq_train_var2(:,2); vq_train_var3(:,2)];

  semilogy(uniform_bits, uniform_var,'b+-;Uniform;');
  hold on; semilogy(vq_bits, vq_var,'g+-;VQ;'); hold off;
  xlabel('Bits/frame'); ylabel('Quant Error $\sigma^2$')
  grid;

  if epslatex, print_eps_restore("uniform_vq.tex","-S300,300",textfontsize,linewidth); end

endfunction


