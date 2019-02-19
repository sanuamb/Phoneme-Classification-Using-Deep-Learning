
%% Load the test sets and train set
train = load('dlsp_train.mat','label','data');
test = load('dlsp_test.mat','label','data');
train_size = 177080;
test_size = 64145;

%% target phonemes we want as classes struct
c{1} = {'aa', 'ae', 'ah', 'ao', 'aw', 'ax-h', 'ax',...
            'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl', 'dh', ...
            'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', ...
            'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih', 'ix', 'iy',...
            'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy', ...
            'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', ...
            'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh'};
        
c{2} = {'VS','NF','SF','WF','ST','CL'};
c{3} = {'SON', 'OBS', 'SIL'};
c{4} = {'Vowels', 'Stops', 'Fricatives', 'Nasals', 'Silences'};

%% get the target labels according to the approaches each struct of train and test
%all 61
get_train_label{1} = gen_labels(train.label,'n');

% Halberstadt's first approach
get_train_label{2} = gen_labels(train.label,'h');

%Halberstadt's second approach
get_train_label{3} = gen_labels(train.label,'H');

%scalon's approach
get_train_label{4} = gen_labels(train.label,'s');

%all 61
get_test_label{1} = gen_labels(test.label,'n');

% Halberstadt's first approach
get_test_label{2} = gen_labels(test.label,'h');

%Halberstadt's second approach
get_test_label{3} = gen_labels(test.label,'H');

%scalon's approach
get_test_label{4} = gen_labels(test.label,'s');

%% one hot encoded by converting str to int
for j = 1:4
    train_label_int = zeros(1,train_size);
    test_label_int = zeros(1,test_size);
    for i = 1:length(c{j})
        train_label_int = train_label_int + i.*strcmp(c{j}{i},get_train_label{j});
        test_label_int = test_label_int + i.*strcmp(c{j}{i},get_test_label{j});
    end
    %final train and test labels
    train_label{j} = train_label_int;
    test_label{j} = test_label_int;
end

%% Save to mat file for DNN and LSTM
save('dlsp_train_label.mat','train_label')
save('dlsp_test_label.mat','test_label')


%% generate labels
function final_labels = gen_labels(labels, selection)
%   selection = 'h'  generate 6 classes of Halberstadt's approach
%   selection = 'H' generate 3 classes of Halberstadt's approach
%   selection = 's' generate 6 classes from scalons approach

nlabels = length(labels);

final_labels = cell(1, nlabels);

if strcmp(selection, 'h')
    h1 = {'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axh', 'axr', 'ay', 'eh', ...
        'er', 'ey', 'ih', 'ix', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux', ...
        'el', 'l', 'r', 'w', 'y'};
    h2 = {'em', 'en', 'eng', 'm', 'n', 'ng', 'nx', 'dx'};
    h3 = {'s', 'z', 'sh', 'zh', 'ch', 'jh'};
    h4 = {'v', 'f', 'dh', 'th', 'hh', 'hv'};
    h5 = {'b', 'd', 'g', 'p', 't', 'k'};
    for i = 1:nlabels
        phn = labels{1, i};
        
        %if h1 then vowels
        if any(strcmp(h1,phn))
            final_labels{1, i} = 'VS';
            
        %if h2 then nasals
        elseif any(strcmp(h2,phn))
            final_labels{1, i} = 'NF';
            
         %if h3 strong fricatives
        elseif any(strcmp(h3,phn))
            final_labels{1, i} = 'SF';
            
        %if h4 weak fricatives
        elseif any(strcmp(h4,phn))
            final_labels{1, i} = 'WF';
            
        %if h5 strong fricatives
        elseif any(strcmp(h5,phn))
            final_labels{1, i} = 'ST';
            
        %else closures
        else
            final_labels{1, i} = 'CL';
        end
    end
    
elseif strcmp(selection, 'H')
    H1 = {'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axh', 'axr', 'ay', 'eh', ...
        'er', 'ey', 'ih', 'ix', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux', 'el', ...
        'l', 'r', 'w', 'y', 'em', 'en', 'eng', 'm', 'n', 'ng', 'nx', 'dx'};
    H2 = {'s', 'z', 'sh', 'zh', 'ch', 'jh', 'v', 'f', 'dh', 'th', 'hh', ...
        'hv', 'b', 'd', 'g', 'p', 't', 'k'};
    for i = 1:nlabels
        phn = labels{1, i};
        %sonorant
        if any(strcmp(H1,phn))
            final_labels{1, i} = 'SON';
        %obstruent
        elseif any(strcmp(H2,phn))
            final_labels{1, i} = 'OBS';
        %silence
        else
            final_labels{1, i} = 'SIL';
        end
    end
    
%scalons approach
elseif strcmp(selection, 's')
    s1 = {'aa', 'ae', 'ah', 'ao', 'ax', 'ax-h', 'axr', 'ay', 'aw', 'eh', ...
        'el', 'er', 'ey', 'ih', 'ix', 'iy', 'l', 'ow', 'oy', 'r', 'uh', ...
        'uw', 'ux', 'w', 'y'};
    s2 = {'p', 't', 'k', 'b', 'd', 'g', 'jh', 'ch'};
    s3 = {'s', 'sh', 'z', 'zh', 'f', 'th', 'v', 'dh', 'hh', 'hv'};
    s4 = {'m', 'em', 'n', 'nx', 'ng', 'eng', 'en'};
    for i = 1:nlabels
        phn = labels{1, i};
        if any(strcmp(s1,phn))
            final_labels{1, i} = 'Vowels';
        elseif any(strcmp(s2,phn))
            final_labels{1, i} = 'Stops';
        elseif any(strcmp(s3,phn))
            final_labels{1, i} = 'Fricatives';
        elseif any(strcmp(s4,phn))
            final_labels{1, i} = 'Nasals';
        else
            final_labels{1, i} = 'Silences';
        end
    end
else
    %if no choice all 61 phonemes
    final_labels = labels;
end

end