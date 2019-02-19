%% Processing phonemes of TIMIT dataset 

%% Parameters
minimum_frequency=50;
maximum_frequency=7000;
%Number of mel Filters
mel_filts=40;

% number of dct 
n_dct=13;

%FFT length
nfft = 8192;

%sampling rate
fs = 16000;

%Window size
wlen = 256;
hop = 128;

% Create mel filterbank it will give matrix of triangular filters
filterbank = create_mel_filterbank(minimum_frequency,maximum_frequency,fs,nfft,mel_filts);

%% Retrieve and separate phoneme audio data
dataset = 'test'; % train or test

%Number of phonemes in train set
train_size = 177080;
%Number of phonemes in test set
test_size = 64145;

%check which folder we are processing train or test from TIMIT folder
if strcmp('train',dataset)
    folder = fullfile('TIMIT','train');
    size = train_size;
else
    folder = fullfile('TIMIT','test');
    size = test_size;
end


first = dir(folder);
%check the iterations 
cnt = 1; 

%check the dialects
dialects= 1; 

%pre-allocate to store the features(72 features)
data = zeros(6*(n_dct-1),size);

%Dialect region(DR)
for i = 3:length(first) 
    second = dir(fullfile(folder,first(i).name));
    %print to check folders in it
    disp(second)
    for j = 3:length(second) 
        third = dir(fullfile(folder,first(i).name,second(j).name,'*.wav'));
        %print to check .wav files in speaker folder
        disp(third);
        for k = 1:length(third) % Sentence text
            rfile = fullfile(folder,first(i).name,second(j).name,third(k).name);
            disp(rfile)
            
            %readsph
            [xt,fs,phn] = readsph(rfile,'t');
            
            for l=1:length(phn(:,1))
                %get phoneme label
                [ws label{cnt}] = phn{l,:};
                num_window_samps = round(ws.*fs);
                phns = xt(num_window_samps(1)+1:num_window_samps(2));
                %create buffer
                xwin = buffer(phns,wlen,wlen-hop);
                %applying hamming window
                xwin = xwin.* hamming(wlen);
                [mfccs,d_mfccs,dd_mfccs] = compute_mfccs(xwin,filterbank,nfft,n_dct);
                
                %take the mean and std of mfcc, delta and deltaDelta
                %returned
                data(:,cnt) = [mean(mfccs,2); std(mfccs,0,2); ...
                              mean(d_mfccs,2); std(d_mfccs,0,2); ...
                              mean(dd_mfccs,2); std(dd_mfccs,0,2)];
                cnt=cnt+1;
            end
        end
      
        
    end
    dialects = [dialects; cnt];
end

%% save data features and rows

data = (data-mean(data,2))./std(data,0,2);

if strcmp('train',dataset)
    save('dlsp_train.mat','label','data')
else
    save('dlsp_test.mat','label','data')
end

%% Creating Mel filterbanks from scratch
% I have referred this function definition from one of audio processing
% modules of matlab to understand the working of mel filterbank creation
% especially parameters and conversions required, instead of using in-built
% function

function [filterbank] = create_mel_filterbank(min_freq,max_freq,fs,nfft,num_mel_filts)


nfft2 = nfft/2+1; 

% Generating the Mel filterbank

%hz to mel-a perceptual scale of pitches 
nmin_mel = hz2mel(min_freq);
nmax_mel = hz2mel(max_freq);


mel_values = linspace(nmin_mel, nmax_mel, num_mel_filts + 2);

%convert mel to hz
num_freq_values = mel2hz(mel_values);
num_freq_bins = floor((nfft+1) * num_freq_values / fs);

filterbank = zeros(nfft2, num_mel_filts);

for i = 1:length(filterbank(1, :))
    for j = 1:length(filterbank(:, 1))
        if (num_freq_bins(i) <= j && j <= num_freq_bins(i + 1))
            filterbank(j, i) = (j - num_freq_bins(i)) / (num_freq_bins(i + 1) - num_freq_bins(i));
        elseif (num_freq_bins(i+ 1) <= j && j <= num_freq_bins(i + 2))
            filterbank(j, i) = (num_freq_bins(i + 2) - j) / (num_freq_bins(i + 2) - num_freq_bins(i + 1));
        end
    end
end

% Normalizing the filters
for i = 1:length(filterbank(1, :))
    filterbank(:, i) = filterbank(:, i) .* (1 / sum(filterbank(:, i)));
end
%transpose
filterbank = filterbank';

end

%% MFCC Computation
% I have referred this function definition from one of audio processing
% modules of matlab to understand the working of mel filterbank creation
% especially parameters and conversions required, instead of using in-built
% function

function [mfccs,d_mfccs,dd_mfccs] = compute_mfccs(ip_phn,filterbank,nfft,n_dct)

% Calculating the power of the spectrum

    Xabs = abs(fft(ip_phn,nfft));
    p = (abs(Xabs).^2)/nfft;

    % Removing the redundant portion of the spectrum(first half only)
    p = p(1: (end/2) + 1, :);
    
    % generate Mel power spectrogram
    lenergy = log(filterbank*p);

    % get DCT (Dicrete Cosinne transform)
    %cepstral coefficients
    ccs = dct(lenergy);
    mfccs = ccs(1:n_dct, :);
    
    
    %remove the DCT component 
    mfccs(1,:) = []; 
    
    n = size(mfccs,2);
    if n == 1
        d_mfccs = zeros(n_dct-1,1);
        dd_mfccs = zeros(n_dct-1,1);
    elseif n == 2
        %create sparse identity matrix
        I = speye(n);
        D = I(2:n, :) - I(1:n-1, :);
        d_mfccs = mfccs*D';
        dd_mfccs = zeros(n_dct-1,1);
    else
        %create sparse identity matrix
        I1 = speye(n);
        I2 = speye(n-1);
        
        D1 = I1(2:n, :) - I1(1:n-1, :);
        D2 = I2(2:n-1,:) - I2(1:n-2,:);
        %delta mfcc
        d_mfccs = mfccs*D1';
        %deltaDelat mfcc
        dd_mfccs = d_mfccs*D2';
    end
end

%% hz2mel
% function I referred from matlab defined functions for audio processing
% for conversion of hz to mel vals for mel filterbank
function melval = hz2mel(val)

melval = 1127.01028*log(1+val./700);

end

%% mel2hz
%function I referred from matlab defined functions for audio processing
% for conversion of mel to hz vals 
function hzval = mel2hz(val)

hzval = 700.*(exp(val./1127.01028)-1);

end
