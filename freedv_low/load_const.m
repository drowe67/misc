% set up bunch of constants for each simulation, a bit like C #defines

bps     = 2;         % two bits/symbol for QPSK
Rs      = 50;        % symbol rate (needed for HF model)
Ts      = 1/Rs;
Ns      = 8;         % one pilot every Ns symbols
Npad    = 3;         % extra frames at end to deal with interpolator edges
Fs      = 8000;      % sample rate
Nd      = 1;         % number of diversity channels
div_Hz  = 1000;      % freq offset of diversity carriers
nbitsperframe = 224; % bits in a modem frame
M       = Fs/Rs;     % oversamplerate, number of samples/symbol
Tcp     = 0.004;     % length of cyclic prefix
Ncp     = Tcp*Fs;

assert(floor(M) == M);
assert(floor(Ncp) == Ncp);

verbose = sim_in.verbose;
EbNovec = sim_in.EbNovec;
ch      = sim_in.ch;

ch_phase = 0;
if isfield(sim_in,"ch_phase")
    ch_phase = sim_in.ch_phase;
end
pilot_freq_weights = [0 1 0];
if isfield(sim_in,"pilot_freq_weights")
    pilot_freq_weights = sim_in.pilot_freq_weights;
end
if isfield(sim_in,"Nd")
    Nd = sim_in.Nd;
end
combining = "ecg";
if isfield(sim_in,"combining")
    combining = sim_in.combining;
end
if isfield(sim_in,"nbitsperframe")
    nbitsperframe = sim_in.nbitsperframe;
end
nbitsperpacket = nbitsperframe;
if isfield(sim_in,"nbitsperpacket")
    nbitsperpacket = sim_in.nbitsperpacket;
end

% A few sums to set up the packet.  Using our OFDM nomenclature, we have one
% "modem frame" for the packet, so Np=1, Nbitspermodemframe == Nbitsperpacket, and
% there is just one modem frame for the FEC codeword.  This is common for 
% digital voice modes.  In this simulation we assume all payload data symbols
% are used for payload data (no text or unique word symbols).

nsymbolsperframe = nbitsperframe/bps;
nsymbolsperframepercarrier = Ns-1;
Nc = nsymbolsperframe/nsymbolsperframepercarrier;
% make sure we have an integer number of carriers
assert(floor (Nc) == Nc);
if verbose == 2
    printf("nbitsperframe: %d\n", nbitsperframe);
    printf("nsymbolsperframe: %d\n",nsymbolsperframe);
    printf("Nc: %d\n", Nc);
    printf("nbitsperpacket: %d\n", nbitsperpacket);
end

% carrier frequencies, start at 400Hz
w = 2*pi*(400 + (0:Nc*Nd+1)*Rs)/Fs;
W = zeros(Nc*Nd,M);
Wi = zeros(M,Nc*Nd);
for c=1:Nc*Nd+2
    Wfwd(c,:) = exp(-j*(0:M-1)*w(c));
    Winv(:,c) = exp(j*(0:M-1)*w(c));
end

% integer number of "modem" frames
nbits= sim_in.nbits;
assert(length(nbits) == 1);
nframes = floor(nbits/nbitsperframe) + Npad;
nsymb = nframes*Ns;
nsam = (M+Ncp)*nsymb;
if verbose == 2
    printf("nframes: %d\n", nframes);
    printf("nsymb..: %d\n", nsymb);
    printf("Seconds: %f\n", nsymb*(Ts+Tcp));
end
