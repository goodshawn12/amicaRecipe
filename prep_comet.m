%% script for generating .fdt and input.param for submitting job on Comet

% add eeglab to path
cd 'C:\Users\shawn\Desktop\Emotion';

% select multiple files for processing
[filename, filepath] = uigetfile('*.set','Select One or More Files','MultiSelect', 'off');

%% 
filepath = '\\sccn.ucsd.edu\projects\Shawn\2019_Emogery\';

% define parameters
numprocs = 1;       % # of nodes
max_threads = 24;   % # of threads
num_models = 20;     % # of models of mixture ICA
max_iter = 2000;    % max number of learning steps
do_opt_block = 0;   % disabled for high density EEG
num_mix_comps = 1;  % reduce runtime / comparable results? (default 3)
do_reject = 1;
numrej = 5;
rejstart = 2;
rejint = 5;
writestep = 20;

for Subj = 1:35
    filename = sprintf('EEG_Subj_%d_64ch.set',Subj);
    
    if exist([filepath filename], 'file')
        split_name = strsplit(filename,'.');
        name_tag = '_64ch';
        
        % load dataset
        EEG = pop_loadset(filename, filepath);
        
        % generate dataset and parameter file only
        % dirname = sprintf('Subj%d_M%d',Subj,num_models);
        data_name = split_name{1};
        param_name = sprintf('emotion_S%d_M%d%s',Subj,num_models,name_tag);
        slurm_name = sprintf('amica.slurm_emotion_S%d_M%d%s',Subj,num_models,name_tag);
        data_folder = [pwd filesep 'upload_comet' filesep 'data' filesep];
        param_folder = [pwd filesep 'upload_comet' filesep 'param' filesep];
        slurm_folder = [pwd filesep 'upload_comet' filesep 'slurm' filesep];
        
        runamica15_prep(EEG.data,data_name,data_folder,param_name,param_folder, ...
            'num_models',num_models,'numprocs', numprocs, 'max_threads', max_threads, 'max_iter',max_iter, 'writestep', writestep, ...
            'do_opt_block', do_opt_block, 'num_mix_comps', num_mix_comps, ...
            'do_reject', do_reject, 'numrej', numrej, 'rejstart', rejstart, 'rejint', rejint);
        
        prep_slurm_script(slurm_folder,slurm_name,param_name,numprocs)
        
    end
end
