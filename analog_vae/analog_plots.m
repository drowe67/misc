# analog_plots.m
# Octave help script to do a few plots

1;
pkg load statistics;

function do_plots(fn='l.f32',png_fn='')
    l=load_f32(fn,1);
    s=l(1:2:end)+j*l(2:2:end);
    figure(1); clf; plot(s,'.')
    if length(png_fn)
        print("-dpng",png_fn);
    end
    figure(2); clf;
    [nn cc] = hist3([real(s) imag(s)],[25 25]);
    mesh(cc{1},cc{2},nn);

endfunction
