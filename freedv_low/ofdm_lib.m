% ofdm_lib.m
% David Rowe Oct 2023
%
% Library of function used in various simulations

1;

% Generate pilots using Barker codes which have good correlation properties
function P = barker_pilots(Nc)
  % Barker codes
  P_barker_8  = [1 1 1 -1 -1 1 -1];
  P_barker_13 = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
  % repeating length 8 Barker code works OK for Nc=8 and Nc=16
  P= zeros(1,Nc);
  for i=1:Nc
    P(i) = P_barker_8(mod(i,length(P_barker_8))+1);
  end
endfunction

function [tx_bits tx] = ofdm_modulator(Ns,Nc,Nd,M,Ncp,Winv,nbitsperframe,nframes,nsymb)
    printf("Modulate to rate Rs OFDM symbols...\n");
    P = barker_pilots(Nc); 
    tx_symb = [];
    for f=1:nframes
      tx_bits = rand(1,nframes*nbitsperframe) > 0.5; bit = 1;
      % set up Nc x Ns array of symbols with pilots
      atx_symb = zeros(Nc,Ns); 
      for c=1:Nc
        atx_symb(c,1) = P;
        for s=2:Ns
          atx_symb(c,s) = qpsk_mod(tx_bits(bit:bit+1)); bit += 2;
        end
      end
      
      % diversity copy
      tmp = [];
      for d=1:Nd;
        tmp = [tmp; atx_symb];
      end
      atx_symb = tmp;

      % add "wingman" pilots for first and last carriers
      wingman = [1 zeros(1,Ns-1)];
      atx_symb = [wingman; atx_symb; wingman];

      % normalise power when using diversity
      atx_symb *= 1/sqrt(Nd);
      
      tx_symb = [tx_symb atx_symb];
    end
    [r c] = size(tx_symb);
    assert(c == nsymb);
    
    % IDFT to convert to from freq domain rate Rs to time domain rate Fs

    printf("IDFT to time domain...\n");
    nsam = (M+Ncp)*nsymb;
    tx = zeros(1,nsam);
    for s=1:nsymb          
      st = (s-1)*(M+Ncp)+1; en = st+M+Ncp-1;
      tx(st+Ncp:en) = Winv*tx_symb(:,s)/M;
      
      % cyclic prefix
      tx(st:st+Ncp-1) = tx(st+M:en);
    end
endfunction


% init HF model
function [spread1 spread2 hf_gain hf_en path_delay_samples] = gen_hf(ch,Fs,nsam)
  printf("Generating HF model spreading samples...\n")
  hf_en = 0; spread1 = []; spread2 = []; hf_gain = 0; path_delay_samples = 0;
  if strcmp(ch,"mpp") || strcmp(ch,"mpd")
    hf_en = 1;
    % some typical values

    if strcmp(ch,"mpp")
      dopplerSpreadHz = 1.0; path_delay_s = 2E-3;
    else
      dopplerSpreadHz = 2.0; path_delay_s = 4E-3;
    end
    path_delay_samples = round(path_delay_s*Fs);
    
    spread1 = doppler_spread(dopplerSpreadHz, Fs, nsam);
    spread2 = doppler_spread(dopplerSpreadHz, Fs, nsam);

    % normalise power through HF channel
    hf_gain = 1.0/sqrt(var(spread1)+var(spread2));
  end
endfunction

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

function draw_ellipse(x1,y1,a,b,angle, leg)

  % Generate points on the ellipse
  theta = linspace(0, 2*pi, 100);  % Angle values
  x = x1 + a * cos(theta) * cos(angle) - b * sin(theta) * sin(angle);
  y = y1 + a * cos(theta) * sin(angle) + b * sin(theta) * cos(angle);

  % Plot the ellipse
  plot(x, y, leg);
end
