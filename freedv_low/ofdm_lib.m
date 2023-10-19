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

function [tx_bits tx P] = ofdm_modulator(Ns,Nc,Nd,M,Ncp,Winv,nbitsperframe,nframes,nsymb)
    printf("Modulate to rate Rs OFDM symbols...\n");
    P = barker_pilots(Nc+2); 
    tx_symb = [];
    for f=1:nframes
      tx_bits = rand(1,nframes*nbitsperframe) > 0.5; bit = 1;
      % set up Nc x Ns array of symbols with pilots
      atx_symb = zeros(Nc,Ns); 
      for c=1:Nc
        atx_symb(c,1) = P(c+1);
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
      wingman_first = [P(1) zeros(1,Ns-1)];
      wingman_last  = [P(Nc+1) zeros(1,Ns-1)];
      atx_symb = [wingman_first; atx_symb; wingman_last];

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
function [spread1 spread2 hf_gain path_delay_samples] = gen_hf(ch,Fs,nsam)
  printf("Generating HF model spreading samples...\n")
  if strcmp(ch,"mpg") || strcmp(ch,"mpp") || strcmp(ch,"mpd")  
    if strcmp(ch,"mpg")
      dopplerSpreadHz = 0.1; path_delay_s = 0.5E-3;
    end
    if strcmp(ch,"mpp")
      dopplerSpreadHz = 1.0; path_delay_s = 2E-3;
    end
    if strcmp(ch,"mpd")
      dopplerSpreadHz = 2.0; path_delay_s = 4E-3;
    end
    
    spread1 = doppler_spread(dopplerSpreadHz, Fs, nsam);
    spread2 = doppler_spread(dopplerSpreadHz, Fs, nsam);
  elseif strcmp(ch,"notch")
    % deep notch that slowly cycles between 500 and 1500 Hz
    spread1 = ones(1,nsam);
    path_delay_s = 1E-3;
    f_low = 500; f_high = 1500;
    w_low = 2*pi*f_low/Fs; w_high = 2*pi*f_high/Fs;
    w_mid = (w_low+w_high)/2;
    w_amp = (w_high - w_low)/2;
    notch_cycle_s = 60; notch_w = 2*pi/(notch_cycle_s*Fs);
    w = w_mid + w_amp*cos((1:nsam)*notch_w);
    % we use H(e^(jw)) = G1 + G2exp^(jwdFs)
    % set G1=1, |G2|=1, and find arg(G2) for H(e^(jw)) = 0
    spread2 = exp(-j*(pi + w*path_delay_s*Fs));
  else
    printf("channel model %s unknown!\n", ch);
    assert(0);
  end

  path_delay_samples = round(path_delay_s*Fs);

  % normalise power through HF channel
  hf_gain = 1.0/sqrt(var(spread1)+var(spread2));

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
