% doppler_spread.m
% David Rowe Jan 2016
%
% Returns gausssian filtered doppler spreading function samples for HF channel
% modelling.  Used PathSim technical guide as a reference - thanks Moe!

function [spread_FsHz states] = doppler_spread(dopplerSpreadHz, FsHz, Nsam)

  % start with low Fs so we can work with a reasonable filter length

  sigma = dopplerSpreadHz/2;
  lowFs = ceil(10*dopplerSpreadHz);
  Ntaps = 100;
  M = FsHz/lowFs;
  assert(M == floor(M));
  Nsam_low = ceil(Nsam/M);
  
  % generate gaussian freq response and design filter

  x = 0:lowFs/100:lowFs/2;
  y = (1/(sigma*sqrt(2*pi)))*exp(-(x.^2)/(2*sigma*sigma));
  b = fir2(Ntaps-1, x/(lowFs/2), y);
  
  % generate the spreading samples, extra Ntap to fill up filter memory

  spread_lowFs = filter(b,1,randn(1,Nsam_low+Ntaps) + j*randn(1,Nsam_low+Ntaps));
  spread_lowFs = spread_lowFs(Ntaps+1:end);
  assert(length(spread_lowFs) == Nsam_low);
  
  % resample to FsHz. As FsHz >> Fs_low, we can use a linear resampler

  spread_FsHz = interp1((1:M:Nsam_low*M),spread_lowFs,1:Nsam);
  assert(length(spread_FsHz) >= Nsam);
  
  % return some states for optional unit testing
  states.x = x;
  states.y = y;
  states.b = b;
  states.spread_lowFs = spread_lowFs;
  states.M = M;
endfunction


