function [cleanAudio,noisyAudio,denoisedAudioFullyConnected,...
          denoisedAudioFullyConvolutional] = testDenoisingNets(ads,...
                            denoiseNetFullyConnected,...
                            denoiseNetFullyConvolutional,noisyMean,...
                            noisyStd,cleanMean,cleanStd)
% testDenoisingNets: Test fully speech denoising networks.
%
% This function is used in SpeechDenoisingExample
%
% ads: audioDatastore containing files to test
% denoiseNetFullyConnected: Fully connected pre-trained network
% denoiseNetFullyConvolutional: Fully convolutional pre-trained network
% noisyMean: Mean of the noisy STFT vectors used in training
% noisyStd: Standard deviation of the noisy STFT vectors used in training
% cleanMean: Mean of the baseline STFT vectors used in training
% cleanStd: Standard deviation of the baseline STFT vectors used in training
%
% The example used the pre-trained networks saved in denoisenet.mat

% Copyright 2018 The MathWorks, Inc.


WindowLength = 256;
win          = hamming(WindowLength,'periodic');
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
NumFeatures  = FFTLength/2 + 1;
NumSegments  = 8;

persistent src
if isempty(src)
    src = dsp.SampleRateConverter('InputSampleRate',48e3,...
        'OutputSampleRate', 8e3,...
        'Bandwidth', 7920);
end

%%
% Read the contents of a file from the datastore
[cleanAudio,info] = read(ads);
Fs = info.SampleRate;

%%
% Make sure the audio length is a multiple of the sample rate converter
% decimation factor
D            = 48/8;
L            = floor(D * numel(cleanAudio));
cleanAudio   = cleanAudio(1:L/D);

%%
% Convert the audio signal to 8 KHz:
cleanAudio   = src(cleanAudio);
reset(src)

%%
% In this testing tage, we will corrupt speech with washing machine noise
% not used in te training stage. Add noise to the speech signal such that
% SNR is 0 dB.
noise         = audioread("WashingMachine-16-8-mono-200secs.wav");

%%
% Create a random noise segment from the washing machine noise vector
randind      = randi(numel(noise) - numel(cleanAudio) , [1 1]);
noiseSegment = noise(randind : randind + numel(cleanAudio) - 1);
  
%%
% Add noise to the speech signal. The SNR is 0 dB.
noisePower   = sum(noiseSegment.^2);
cleanPower   = sum(cleanAudio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
noisyAudio   = cleanAudio + noiseSegment;

%%
% Use |spectrogram| to generate magnitude STFT vectors from the noisy audio signals:
noisySTFT  = spectrogram(noisyAudio, win, Overlap, FFTLength,Fs);
noisyPhase = angle(noisySTFT);
noisySTFT  = abs(noisySTFT);

%%
% Generate the 8-segment training predictor signals from the noisy STFT.
% The overlap between consecutive predictors is equal to 7 segments.
noisySTFT    = [noisySTFT(:,1:NumSegments-1) noisySTFT];
predictors = zeros( NumFeatures, NumSegments , size(noisySTFT,2) - NumSegments + 1);
for index     = 1 : size(noisySTFT,2) - NumSegments + 1
    predictors(:,:,index) = noisySTFT(:,index:index+NumSegments-1); 
end

%%
% Normalize the predictors by the mean and standard deviation
% computed in the training stage:
predictors(:) = (predictors(:) - noisyMean) / noisyStd;

%%
% Compute the denoised magnitude STFT by using |predict| with the two
% trained networks.
predictors = reshape(predictors,[NumFeatures, NumSegments,1,size(predictors,3)]);
STFTFullyConnected     = predict(denoiseNetFullyConnected , predictors);
STFTFullyConvolutional = predict(denoiseNetFullyConvolutional , predictors);

%%
% Scale the outputs by the mean and standard deviation used in the
% training stage
STFTFullyConnected(:)     = cleanStd * STFTFullyConnected(:)     +  cleanMean;
STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) +  cleanMean;

%%
% Compute the denoised speech signals. |istft| performs inverse STFT. Use
% the phase of the noisy STFT vectors.
denoisedAudioFullyConnected     = istft(STFTFullyConnected.' .* exp(1j*noisyPhase),  ...
                                        WindowLength-Overlap, FFTLength, win);
denoisedAudioFullyConvolutional = istft(squeeze(STFTFullyConvolutional) .* exp(1j*noisyPhase), ...
                                        WindowLength-Overlap, FFTLength, win);

%%
% Plot the clean, noisy and denoised audio signals.
figure;
subplot(411)
t = (1/Fs) * ( 0:numel(denoisedAudioFullyConnected)-1);
plot(t,cleanAudio(1:numel(denoisedAudioFullyConnected)))
title('Clean Speech')
grid on
subplot(412)
plot(t,noisyAudio(1:numel(denoisedAudioFullyConnected)))
title('Noisy Speech')
grid on
subplot(413)
plot(t,denoisedAudioFullyConnected)
title('Denoised Speech (Fully Connected Layers)')
grid on
subplot(414)
plot(t,denoisedAudioFullyConvolutional)
title('Denoised Speech (Convolutional Layers)')
grid on
xlabel('Time (s)')

%%
% Plot the clean, noisy and denoised spectrograms.
h = figure;
subplot(411)
spectrogram(cleanAudio, win, Overlap, FFTLength,Fs);
title('Clean Speech')
grid on
subplot(412)
spectrogram(noisyAudio, win, Overlap, FFTLength,Fs);
title('Noisy Speech')
grid on
subplot(413)
spectrogram(denoisedAudioFullyConnected, win, Overlap, FFTLength,Fs);
title('Denoised Speech (Fully Connected Layers)')
grid on
subplot(414)
spectrogram(denoisedAudioFullyConvolutional, win, Overlap, FFTLength,Fs);
title('Denoised Speech (Convolutional Layers)')
grid on
p = get(h,'Position');
set(h,'Position',[p(1) 65 p(3) 800]);