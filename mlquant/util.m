1;

function [textfontsize linewidth] = set_fonts(font_size=12)
  textfontsize = get(0,"defaulttextfontsize");
  linewidth = get(0,"defaultlinelinewidth");
  set(0, "defaulttextfontsize", font_size);
  set(0, "defaultaxesfontsize", font_size);
  set(0, "defaultlinelinewidth", 0.5);
end

function restore_fonts(textfontsize,linewidth)
  set(0, "defaulttextfontsize", textfontsize);
  set(0, "defaultaxesfontsize", textfontsize);
  set(0, "defaultlinelinewidth", linewidth);
end

function print_eps_restore(fn,sz,textfontsize,linewidth)
  print(fn,"-depslatex",sz);
  printf("printing... %s\n", fn);
  restore_fonts(textfontsize,linewidth);
end
