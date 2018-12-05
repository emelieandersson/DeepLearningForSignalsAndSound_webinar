function [targets,predictors] = HelperGenerateSpeechDenoisingFeatures(audio,noise, src)
% HelperGenerateSpeechDenoisingFeatures: Get target and predictor STFT
% signals for speech denoising.
% audio: Input audio signal
% noise: Input noise signal
% src:   Sample rate converter

% Copyright 2018 The MathWorks, Inc.

WindowLength = 256;
win          = hamming(WindowLength,'periodic');
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
Fs           = 8e3;
NumFeatures  = FFTLength/2 + 1;
NumSegments  = 8;

D            = 48/8; % Decimation factor
L            = floor( numel(audio)/D);
audio        = audio(1:D*L);

audio   = src(audio);
reset(src);

randind      = randi(numel(noise) - numel(audio) , [1 1]);
noiseSegment = noise(randind : randind + numel(audio) - 1);

noisePower   = sum(noiseSegment.^2);
cleanPower   = sum(audio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
noisyAudio   = audio + noiseSegment;

cleanSTFT = spectrogram(audio, win, Overlap, FFTLength, Fs);
cleanSTFT = abs(cleanSTFT);
noisySTFT = spectrogram(noisyAudio, win, Overlap, FFTLength,Fs);
noisySTFT = abs(noisySTFT);

noisySTFTAugmented    = [noisySTFT(:,1:NumSegments-1) noisySTFT];
 
STFTSegments = zeros( NumFeatures, NumSegments , size(noisySTFTAugmented,2) - NumSegments + 1);
for index     = 1 : size(noisySTFTAugmented,2) - NumSegments + 1
    STFTSegments(:,:,index) = noisySTFTAugmented(:,index:index+NumSegments-1);
end

targets    = cleanSTFT;
predictors = STFTSegments;