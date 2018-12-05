function x = istft(xfft, hop, FFTLength, win)
% ISTFT Inverse Short-Time FFT
% xfft: Short-time FFT (spectrogram) input
% hop : Hop size
% FFTLength: FFT length
% win:  Window

% Copyright 2018 The MathWorks, Inc.

numSegments = size(xfft, 2);  % Number of STFT segments
wlen        = numel(win);
xlen        = wlen + (numSegments-1)*hop;
x           = zeros(xlen,1);

% initialize the signal time segment index
indx = 0;

if mod(FFTLength, 2) == 1
    X = [xfft;conj(xfft(end:-1:2,:))];
else
    X = [xfft;conj(xfft(end-1:-1:2,:))];
end

XIFFT = real(ifft(X,[],1));
XIFFT = XIFFT(1:wlen,:) .* win;

for col = 1:numSegments
    x((indx+1):(indx+wlen)) = x((indx+1):(indx+wlen)) + XIFFT(:,col);
    indx = indx + hop;
end

% scale the signal
W0 = sum(win.^2);
x  = x.*hop/W0;

end