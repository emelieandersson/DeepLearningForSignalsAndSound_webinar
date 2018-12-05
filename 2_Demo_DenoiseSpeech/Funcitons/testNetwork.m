function [denoisedAudio, cleanAudio, noisyAudio] = testNetwork(ads, noise, src, D)

WindowLength = 256;
win          = hamming(WindowLength,"periodic");
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
inputFs      = 48e3;
Fs           = 8e3;
NumFeatures  = FFTLength/2 + 1;
NumSegments  = 8;
%%
s = load("denoisenet.mat");
denoiseNetFullyConvolutional = s.denoiseNetFullyConvolutional;
cleanMean = s.cleanMean;
cleanStd  = s.cleanStd;
noisyMean = s.noisyMean;
noisyStd  = s.noisyStd;

%% Shuffle the files in the datastore
ads = shuffle(ads);
% Read the contents of a file from the datastore
[cleanAudio,info] = read(ads);
% Make sure the audio length is a multiple of the sample rate converter decimation factor
L            = floor( numel(cleanAudio)/D);
cleanAudio   = cleanAudio(1:D*L);
% Convert the audio signal to 8 kHz:
cleanAudio   = src(cleanAudio);
reset(src)
% In this testing stage, you corrupt speech with washing machine noise not used in the training stage.

%% Create a random noise segment from the washing machine noise vector.
randind      = randi(numel(noise) - numel(cleanAudio) , [1 1]);
noiseSegment = noise(randind : randind + numel(cleanAudio) - 1);
% Add noise to the speech signal such that the SNR is 0 dB.
noisePower   = sum(noiseSegment.^2);
cleanPower   = sum(cleanAudio.^2);
noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
noisyAudio   = cleanAudio + noiseSegment;

%% Use spectrogram to generate magnitude STFT vectors from the noisy audio signals:
noisySTFT  = spectrogram(noisyAudio, win, Overlap, FFTLength,Fs);
noisyPhase = angle(noisySTFT);
noisySTFT  = abs(noisySTFT);

%% Generate the 8-segment training predictor signals from the noisy STFT. The overlap between consecutive predictors is 7 segments.
noisySTFT    = [noisySTFT(:,1:NumSegments-1) noisySTFT];
predictors = zeros( NumFeatures, NumSegments , size(noisySTFT,2) - NumSegments + 1);
for index     = 1 : size(noisySTFT,2) - NumSegments + 1
    predictors(:,:,index) = noisySTFT(:,index:index+NumSegments-1);
end

%% Normalize the predictors by the mean and standard deviation computed in the training stage:
predictors(:) = (predictors(:) - noisyMean) / noisyStd;
% Compute the denoised magnitude STFT by using predict with the two trained networks.
predictors = reshape(predictors,[NumFeatures, NumSegments,1,size(predictors,3)]);
STFTFullyConvolutional = predict(s.denoiseNetFullyConvolutional, predictors);
% Scale the outputs by the mean and standard deviation used in the training stage.
STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) +  cleanMean;

%% Compute the denoised speech signals. istft performs the inverse STFT. Use the phase of the noisy STFT vectors to reconstruct the time-domain signal.
denoisedAudio = istft(squeeze(STFTFullyConvolutional) .* exp(1j*noisyPhase), ...
    WindowLength-Overlap, FFTLength, win);
end