function speechDenoisingRealtimeApp(varargin)
%speechDenoisingRealtimeApp Graphical interface for real-time speech
%denoising example

% Copyright 2018 The MathWorks, Inc.

%% Define  system parameters
WindowLength = 256;
win          = hamming(WindowLength,'periodic');
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
Fs           = 8e3;
Outsize      = FFTLength/2 + 1;
NumSegments  = 8;
HopSize      = WindowLength - Overlap;

%% Optional noise gate runs at output of denoising for further noise suppression
ng = noiseGate('SampleRate',Fs,'Threshold',-35,...
    'HoldTime',0,'AttackTime',0.5,'ReleaseTime',.3);

%% This buffer is used for STFT computation
buff          = dsp.AsyncBuffer('Capacity',2*WindowLength);

%% This buffer is used to read most recent 8 STFT vectors
segmentBuffer = dsp.Buffer('Length',NumSegments,'OverlapLength',NumSegments-1);

%% Variable used for overlap-add in ISTFT
IFFTSegment   = zeros(WindowLength,1);

%% Define I/O objects
cleanReader  = dsp.AudioFileReader('Rainbow-16-8-mono-114secs.wav',...
    'SamplesPerFrame',HopSize,...
    'PlayCount', Inf,...
    'OutputDataType','single');
noisyReader  = dsp.AudioFileReader('RainbowNoisy-16-8-mono-114secs.wav',...
    'SamplesPerFrame', HopSize,...
    'PlayCount', Inf,...
    'OutputDataType','single');
player = audioDeviceWriter('SampleRate',Fs);
scope = dsp.TimeScope('TimeSpan',5,'SampleRate',Fs,...
    'YLimits',[-1 1],'NumInputPorts',3,...
    'LayoutDimensions',[3 1],'BufferLength',70e3);
scope.ActiveDisplay = 1;
scope.Title        =  'Noisy Audio (SNR = 0 dB)';
scope.ShowGrid = true;
scope.ActiveDisplay = 2;
scope.Title        =  'Denoised Audio';
scope.ShowGrid = true;
scope.ChannelNames = {'Denoised Audio','Noise Gate Gain'};
scope.ShowLegend = true;
scope.ActiveDisplay = 3;
scope.Title        =  'Clean Audio (Baseline)';
scope.ShowGrid = true;

%% Parameter-tuning UI
param(1).Name = 'Audio Output';
param(1).Type = 'dropdown';
param(1).InitialValue = 1;
audioSource           = 1;
param(1).Values = {'Noisy','Denoised','Clean'};
param(2).Name = 'Enable Noise Gate';
param(2).Type = 'dropdown';
param(2).InitialValue = 1;
ngOn           = true;
g              = ones(WindowLength - Overlap,1);
param(2).Values = {'Yes','No'};
param(3).Name = 'Attack Time';
param(3).InitialValue = 0.5;
param(3).Limits = [0, 5];
param(4).Name = 'Release Time';
param(4).InitialValue = 0.3;
param(4).Limits = [0, 5];
param(5).Name = 'Threshold';
param(5).InitialValue = -35;
param(5).Limits = [-70, 0];
% Create the UI and pass it the parameters
tuningUI = HelperCreateParamTuningUI(param, 'DenoisingNet');

if nargin == 1
    numSteps = varargin{1};
else
    numSteps = Inf;
end

stepCount = 1;
while ~isDone(noisyReader) && stepCount < numSteps
    
    tuneStruct = HelperUnpackUIData(tuningUI);
    if tuneStruct.Stop
        break;
    end
    if tuneStruct.Pause
        continue;
    end
    
    %% Tune parameters if sliders moved
    if tuneStruct.ValuesChanged
        vals =   tuneStruct.TuningValues;
        audioSource =  vals(1);
        ngOn           = vals(2) == 1;
        g              = ones(WindowLength - Overlap,1);
        ng.AttackTime  = vals(3);
        ng.ReleaseTime = vals(4);
        ng.Threshold   = vals(5);
    end
    
    audio = noisyReader();
    clean = cleanReader();
    
    write(buff,audio);
    
    % Read overlapping frames
    audioSegment = read(buff,WindowLength,Overlap);
    
    % Compute STFT of frame
    SFFT  =    fft(audioSegment.*win ,FFTLength);
    SFFT  =    SFFT(1:Outsize);
    
    % Read most recent 8 STFT vectors
    SFFT_Image = segmentBuffer(abs(SFFT).').';
    
    % Pass through deep learning network
    Y =   denoise(SFFT_Image);
    
    % Inverse STFFT. Use phase of noisy input audio
    YFFT      = Y.*exp(1j * angle(SFFT));
    YFFT_Full = [YFFT; conj(YFFT(end-1:-1:2))];
    
    YIFFT            = real(ifft(YFFT_Full)).*win;
    YIFFT(1:Overlap) = YIFFT(1:Overlap) +  IFFTSegment(HopSize+1:end);
    IFFTSegment      = YIFFT;
    out              = IFFTSegment(1:HopSize);
    
    % Apply optional noise gate
    if ngOn
        [out,g] = ng(out);
        g = 10.^(g/20);
    end
    
    scope(audio,[out,g],clean);
    
    % Play selected audio
    if audioSource ==1
        player(audio);
    elseif audioSource ==2
        player(out);
    else
        player(clean);
    end
    
    stepCount = stepCount + 1;
end

if ishghandle(tuningUI)
    close(tuningUI);
end

end

function y = denoise(x)
% denoise Denoise input x by passing it through pre-trained denoising deep
% network.
%
% This file is used in SpeechDenoisingExample

% Copyright 2018 The MathWorks, Inc.

persistent W1 B1 W2 B2 W3 B3 scale1 offset1 scale2 offset2 cleanMean cleanStd noisyMean noisyStd averageImage

if isempty(W1)
    
    load('denoisenet.mat','denoiseNetFullyConnected','cleanMean','cleanStd','noisyMean','noisyStd');
    averageImage = denoiseNetFullyConnected.Layers(1).AverageImage;
    
    W1    = denoiseNetFullyConnected.Layers(2).Weights;
    B1    = denoiseNetFullyConnected.Layers(2).Bias;
    
    epsilon   = 1e-5;
    
    beta1      = squeeze(denoiseNetFullyConnected.Layers(3).Offset);
    gamma1     = squeeze(denoiseNetFullyConnected.Layers(3).Scale);
    inputMean1 = squeeze(denoiseNetFullyConnected.Layers(3).TrainedMean);
    inputVar1  = squeeze(denoiseNetFullyConnected.Layers(3).TrainedVariance);
    scale1     = gamma1./sqrt(inputVar1 + epsilon);
    offset1    = beta1 - gamma1.*inputMean1./sqrt(inputVar1 + epsilon);
    
    W2    = denoiseNetFullyConnected.Layers(5).Weights;
    B2    = denoiseNetFullyConnected.Layers(5).Bias;
    
    beta2      = squeeze(denoiseNetFullyConnected.Layers(6).Offset);
    gamma2     = squeeze(denoiseNetFullyConnected.Layers(6).Scale);
    inputMean2 = squeeze(denoiseNetFullyConnected.Layers(6).TrainedMean);
    inputVar2  = squeeze(denoiseNetFullyConnected.Layers(6).TrainedVariance);
    scale2     = gamma2./sqrt(inputVar2 + epsilon);
    offset2    = beta2 - gamma2.*inputMean2./sqrt(inputVar2 + epsilon);
    
    W3    = denoiseNetFullyConnected.Layers(8).Weights;
    B3    = denoiseNetFullyConnected.Layers(8).Bias;
    
end

% Simulate each layer of the network

x  = (x-noisyMean)./noisyStd;

x = reshape(x,129*8,1);
% Subtract average image
x = x - averageImage(:);
% Fully connected layer
x = W1 * x + B1;
% Batch Normalization
x = scale1.*x + offset1;
% relu
x = max(x,0);
% Fully connected layer
x = W2 * x + B2;
% Batch Normalization
x = scale2.*x + offset2;
% relu
x = max(x,0);
% Fully connected layer
x = W3 * x + B3;

y = x*cleanStd + cleanMean;

end