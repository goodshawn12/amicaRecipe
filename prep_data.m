
% eeglab
cd '/home/goodshawn12/MATLAB/2018_Emogery';
addpath('/data/projects/Shawn/2019_Emogery');
eeglab


%% cleaning pipeline from preprocessed data
filepath = '/data/projects/Shawn/2019_Emogery';
nchans = 64;
asr_stdcutoff = 20;

parfor subj_id = 1:35
    try
        filename = sprintf('EEG_Subj_%d',subj_id);
        EEG = pop_loadset( [filename, '.set']);
                
        % remove EOG channels (EX1-6)
        eog_chan_rm = {'EXG1' 'EXG2' 'EXG3' 'EXG4' 'EXG5' 'EXG6'};
        EEG = pop_select(EEG,'nochannel',eog_chan_rm);
        
        % remove (mostly) EMG channels (whose Z pos < Z(LPA,RPA))
        LPA_index = find(strcmp({EEG.chaninfo.nodatchans.labels},'LPA'));
        RPA_index = find(strcmp({EEG.chaninfo.nodatchans.labels},'RPA'));
        Z_PA = (EEG.chaninfo.nodatchans(LPA_index).Z + EEG.chaninfo.nodatchans(RPA_index).Z) / 2;
        emg_chan_rm = [];
        for it = 1:length(EEG.chanlocs)
            if EEG.chanlocs(it).Z < Z_PA
                emg_chan_rm = [emg_chan_rm, it];
            end
        end
        EEG = pop_select(EEG,'nochannel',emg_chan_rm);
        EEG.etc.emg_chan_rm = emg_chan_rm;
        chanlocs_raw = EEG.chanlocs;
        EEG.etc.chanlocs_raw = chanlocs_raw;
        
        % remove time before the first event and after the last event
        time_window = [EEG.event(1).latency, EEG.event(end).latency];
        EEG = pop_select(EEG,'point',time_window);
        
        % average re-reference
        EEG = pop_reref(EEG, []);
        
        % ASR cleaning
        EEG = clean_asr(EEG,asr_stdcutoff);
        
        % subselect channels
        EEG.etc.chanlocs_before_subselect = EEG.chanlocs;
        subsets = loc_subsets(EEG.chanlocs, nchans, 1, 1);
        channel_id = sort(subsets{1});
        EEG = pop_select(EEG,'channel',channel_id);
        
        % check rank
        if rank(double(EEG.data(:,randperm(EEG.pnts,floor(EEG.pnts/10))))) ~= EEG.nbchan
            error('EEG data with sub-selected channels were not full rank')
        end

        % save preprocessed data
        pop_saveset(EEG,'filename',[filename, '_' num2str(nchans) 'ch.set'],'filepath',filepath);
    end
end


%% cleaning pipeline from raw EEG
%{
%% load raw EEG data
subj_id = 2;

% load raw bdf file
filename = sprintf('eeg_recording_%d',subj_id);
EEG = pop_biosig([filename '.bdf']);

% remove non-EEG channels (EX7-8: LPA/RPA, Ana1-16)
EEG = pop_select(EEG,'channel',1:254);

% load channel locations
filename_chanlocs = sprintf('channel_locations_%d.elp',subj_id);
EEG = pop_chanedit(EEG,'load',{filename_chanlocs 'filetype' 'autodetect'});

% remove EOG channels (EX1-6)
EEG = pop_select(EEG,'channel',1:248);

% remove time before the first event and after the last event
time_window = [EEG.event(1).latency, EEG.event(end).latency];
EEG = pop_select(EEG,'point',time_window);

% average re-reference
EEG = pop_reref(EEG, []);

% ASR cleaning
asr_stdcutoff = 20;
EEG = clean_asr(EEG,asr_stdcutoff);

% save preprocessed data
pop_saveset(EEG,'filename',filename);


%% preprocessing
chanlocs_raw = EEG.chanlocs;
EEG.etc.chanlocs_raw = chanlocs_raw;

% High-pass filtering at 1Hz
filter_hp_cutoff = 1;
EEG = pop_eegfiltnew(EEG,[],filter_hp_cutoff,[],1,0,0); % why set revfilt = 1 (invert filter)?

% remove bad channels:
rmchan_flatline = 5;
rmchan_mincorr = 0.7;
rmchan_linenoise = 4;
channel_crit_maxbad_time = 0.5;

EEG = clean_flatlines(EEG,rmchan_flatline);
EEG = clean_channels(EEG,rmchan_mincorr,rmchan_linenoise);

% average re-reference
EEG = pop_reref(EEG, []);
% full rank average reference?

% ASR cleaning
asr_stdcutoff = 20;
EEG = clean_asr(EEG,asr_stdcutoff);

% save preprocessed data
pop_saveset(EEG,'filename',filename);


%% subselect channels

% load preprocessed, bad-channel-removed EEG data
EEG = pop_loadset();

% interpolate / subselect channels
nchans = 64;
do_interp = 0;      % option for interpolating removed channels
do_selscalp = 1;    % option for rejecting channels with Z < 0, mostly EMG

if do_interp
    if do_selscalp
        chan_keep_interp = [];
        for it = 1:length(chanlocs_raw)
            if chanlocs_raw(it).Z >= 0
                chan_keep_interp = [chan_keep_interp, it];
            end
        end
        subsets = loc_subsets(chanlocs_raw(chan_keep_interp), nchans, 1, 1);
    else
        subsets = loc_subsets(chanlocs_raw, nchans, 1, 1);
    end
    
    % interpolate channels
    EEG = pop_interp(EEG, chanlocs_raw, 'spherical');
    
else
    if do_selscalp
        chan_keep_nointerp = [];
        for it = 1:length(EEG.chanlocs)
            if EEG.chanlocs(it).Z >= 0
                chan_keep_nointerp = [chan_keep_nointerp, it];
            end
        end
        subsets = loc_subsets(EEG.chanlocs(chan_keep_nointerp), nchans, 1, 1);
    else
        subsets = loc_subsets(EEG.chanlocs, nchans, 1, 1);
    end
end

% subselect channels
channel_id = sort(subsets{1});
EEG_sel = pop_select(EEG,'channel',channel_id);

% check rank
if rank(EEG_sel.data(:,randperm(EEG_sel.pnts,floor(EEG_sel.pnts/10)))) ~= EEG_sel.nbchan
    error('EEG data with sub-selected channels were not full rank')
end

% save dataset
pop_saveset(EEG_sel);

%}