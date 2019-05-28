
%% load EEG data
cd 'C:\Users\shawn\Desktop\Emotion'

subj_id = 2;
filename = sprintf('EEG_Subj_%d_64ch.set',subj_id);
EEG = pop_loadset('filename',filename,'filepath','C:\Users\shawn\Desktop\Emotion\');

%% figure 1: plot model probability over time 
subj_id = 2;
numMod = 20;

amicaout_dir = sprintf('amicaout\\emotion_S%d_M%d_72ch_interp',subj_id,numMod);
ourdir = ['C:\Users\shawn\Desktop\Emotion\' amicaout_dir]; 
modout = loadmodout15(ourdir);

srate = 250; 
winLen = 5;     % sec
walkLen = 1;    % sec
pnts = size(modout.v,2);

numMod = modout.num_models;
v = zeros(ceil(pnts/srate/walkLen),numMod);

for it = 1:ceil(pnts/srate/walkLen)
    dataRange = (it-1)*walkLen*srate+1 : min(pnts, (it-1)*walkLen*srate+winLen*srate);
    keepIndex = find(sum(modout.v(:,dataRange),1)~=0);
    v(it,:) = mean(10.^modout.v(:,dataRange(keepIndex)),2);
end

figure, imagesc(v'); colorbar
xlabel('Time (sec)'); ylabel('Model ID'); 
set(gca,'fontsize',12); 
set(gcf,'position',[50,150,1450,450]);


hold on,
% add event markers
for it = 1:length(EEG.event)
    if ~strcmp(EEG.event(it).type,'press') && ~strcmp(EEG.event(it).type,'press1') && ...
            ~strcmp(EEG.event(it).type,'100') && ~strcmp(EEG.event(it).type,'enter') && ...
            ~strcmp(EEG.event(it).type,'768') && ~strcmp(EEG.event(it).type,'4') && ...
            ~strcmp(EEG.event(it).type,'instruct3') && ~strcmp(EEG.event(it).type,'instruct4') && ...
            ~strcmp(EEG.event(it).type,'prebase_instruct') && ~strcmp(EEG.event(it).type,'postbase_instruct')
        x = EEG.event(it).latency/EEG.srate;
        h = plot([x x],[0.5 numMod+0.5],'w--','LineWidth',1);
        if ~strcmp(EEG.event(it).type,'exit')
            text(x,0.5,EEG.event(it).type,'Rotation',45,'fontsize',12);
        end
    end
end

figname = sprintf('modelprob_S%dM%d_72ch_interp.png',subj_id,numMod);
saveas(gcf,figname)


%%
LLt = tmp;
norm_LLt = bsxfun(@rdivide,LLt,sum(LLt));

numMod = modout.num_models;
v = zeros(ceil(EEG.xmax/walkLen),numMod);

for it = 1:ceil(EEG.xmax/walkLen)
    dataRange = (it-1)*walkLen*EEG.srate+1 : min(EEG.pnts, (it-1)*walkLen*EEG.srate+winLen*EEG.srate);
    v(it,:) = mean(norm_LLt(:,dataRange(keepIndex)),2);
end

figure, imagesc(v'); colorbar
xlabel('Time (sec)'); ylabel('Model ID'); 
set(gca,'fontsize',12); 
set(gcf,'position',[50,150,1450,450]);


%% figure 2: bar plot of mean model probability in each emotional states (single subject)
numStage = 10;

% compute mean probability in each stage
modProbXState = zeros(numMod,numStage);

for segmentID = 1:numStage
    
    % define start and end events for each stage
    dataRange = [];
    if segmentID == 1
        startEvent = 'firstBaseline'; endEvent = 'endFirstBaseline';
    elseif segmentID == 2
        startEvent = 'startPreInductionInterview'; endEvent = 'endPreInductionInterview';
    elseif segmentID == 3
        startEvent = 'startInduction'; endEvent = 'endInduction';
    elseif segmentID == 4
        startEvent = 'beginStairDescent'; endEvent = 'endStairDescent';
    elseif segmentID == 5
        startEvent = 'sitInChair'; endEvent = 'Crown';
    elseif segmentID == 6
        startEvent = 'Crown'; endEvent = 'seeingLight';
    elseif segmentID == 7
        startEvent = 'seeingLight'; endEvent = 'navigateLight';
    elseif segmentID == 8
        startEvent = 'navigateLight'; endEvent = 'beginStairAscent'; % 'removeCrown';
    elseif segmentID == 9
        startEvent = 'beginStairAscent'; endEvent = 'endStairAscent';
    else
        startEvent = 'finalBaseline'; endEvent = 'endFinalBaseline';
    end
    
    startIdx = find(strcmp({EEG.event.type},startEvent));
    endIdx = find(strcmp({EEG.event.type},endEvent));
    
    if length(startIdx) > 1
        startIdx = startIdx(end);
    end
    if segmentID == 10 && isempty(endIdx)
        dataRange = ceil(EEG.event(startIdx).latency) : EEG.pnts;
    end
    
    if isempty(dataRange)
        dataRange = ceil(EEG.event(startIdx).latency) : floor(EEG.event(endIdx).latency);
    end
    keepIndex = find(sum(modout.v(:,dataRange),1) ~= 0);
    modProbXState(:,segmentID) = mean(10.^modout.v(:,dataRange(keepIndex)),2);
end

% plot mean probability 
figure, ax = axes;
h = bar(modProbXState','BarWidth',1); 
xlabel('Meditation States'); ylabel('Model Probabilities');
set(gca,'XTickLabel',{'Baseline','PreInduction','Induction','StairDescent','SitInChair','Crown','SeeLight','NavigateLight','StairAscent','FinalBaseline'},'XTick',1:size(modProbXState,2),'XTickLabelRotation',60); 
set(gca,'fontsize',14,'fontweight','bold'); set(gcf,'Units','centimeters','Position',[5 5 30 20]);
legend('Model 1','Model 2','Model 3','Model 4','Model 5','Model 6','Model 7','Model 8');

cmap = [0         0.4470    0.7410
        0.8500    0.3250    0.0980 
        0.9290    0.6940    0.1250 
        0.4940    0.1840    0.5560 
        0.4660    0.6740    0.1880  
        0.3010    0.7450    0.9330 
        0.6350    0.0780    0.1840
        0         0         0
        ]; 
colormap(cmap);


%% figure 3: directed graph to explore structure of clusters - show similarity between runs with diff # of models 
subj_id = 2;
filename = sprintf('EEG_Subj_%d.set',subj_id);
EEG = pop_loadset('filename',filename,'filepath','C:\Users\shawn\Desktop\Emotion\');

% model probabilities for all ICAMM
numMod_list = [3:20];
modProb = cell(1,length(numMod_list));
for numMod = 3:20
    
    amicaout_dir = sprintf('Subj%d_M%d',subj_id,numMod);
    ourdir = ['C:\Users\shawn\Desktop\Emotion\' amicaout_dir];
    modout = loadmodout15(ourdir);
        
    % non-overlapping sliding window average of model probabilities
    winLen = 1;     % sec
    walkLen = 1;    % sec
    v = zeros(ceil(EEG.xmax/walkLen),numMod);
    
    for it = 1:ceil(EEG.xmax/walkLen)
        dataRange = (it-1)*walkLen*EEG.srate+1 : min(EEG.pnts, (it-1)*walkLen*EEG.srate+winLen*EEG.srate);
        keepIndex = find(sum(modout.v(:,dataRange),1)~=0);
        v(it,:) = mean(10.^modout.v(:,dataRange(keepIndex)),2);
    end
    
    modProb{numMod-2} = v;
    
end

%% model matching directed graph
one_to_one_map = 0;
corrThres = 0.30;
weights = [];
source = [];
target = [];
nodeCount = 0;
nodeLabel = [];
for it = 1:length(modProb)-1
    
    % handle NaN in model probability
    nan_index_1 = find(isnan(modProb{it}(:,1)));
    nan_index_2 = find(isnan(modProb{it+1}(:,1)));
    keepIndex = setdiff(1:size(modProb{it},1), unique([nan_index_1; nan_index_2]));
    
    % compute cross correlation 
    [corr,indx,indy,corrs] = matcorr(modProb{it}(keepIndex,:)',modProb{it+1}(keepIndex,:)');
    
    % force 1-to-1 mapping (only takes max-corr match)
    if one_to_one_map
        [weightCorr,indexCorr] = max(corrs);
        weights = [weights, 1-weightCorr];
        source = [source, indexCorr + nodeCount];
        target = [target, (1:it+3) + nodeCount + it+2];
    else % report all edges with corr > corrThres
        edgeIndex = find(corrs > corrThres);
        edgeWeight = corrs(edgeIndex);
        sourceNode = mod(edgeIndex-1,it+2)+1;
        targetNode = floor((edgeIndex-1)/(it+2))+1;
        weights = [weights; 1-edgeWeight];
        source = [source; sourceNode + nodeCount];
        target = [target; targetNode + nodeCount + it+2];
    end
    
    nodeCount = nodeCount + it+2;
    nodeLabel = [nodeLabel, 1:(it+2)];

end
nodeLabel = [nodeLabel,1:(it+3)];

% G = digraph(source,target,weights);
G = graph(source,target,weights);
figure, h = plot(G,'EdgeLabel',fix((1-G.Edges.Weight)*10^3)/10^3, ...
                   'LineWidth',20*((1-G.Edges.Weight)-min((1-G.Edges.Weight)))+0.5, ...
                   'NodeLabel',[]); %nodeLabel);
set(gca,'XTickLabel',{},'YTick',1:11,'YTickLabel',num2cell(13:-1:3),'Fontsize',14)
set(gcf,'Position',[0,0,1000, 800])
for i=1:length(h.XData)
   text(h.XData(i)+0.1,h.YData(i),num2str(nodeLabel(i)),'fontsize',14);
end

% compute distances of the shortest path between all model pairs
d = distances(G);
max_numMode = numMod_list(end);
d_final = d((end-max_numMode+1):end, (end-max_numMode+1):end);
figure, imagesc(d);
figure, imagesc(d_final);
% target((end-numMod_list(end)+1):end)





%% Examine IC scalp maps
numMod = 20;
subj_id = 2;

amicaout_dir = sprintf('Subj%d_M%d',subj_id,numMod);
ourdir = ['C:\Users\shawn\Desktop\Emotion\' amicaout_dir];
modout = loadmodout15(ourdir);

model_id = 3;

EEG.icaweights = modout.W(:,:,model_id);
EEG.icasphere = modout.S;
EEG.icawinv = modout.A(:,:,model_id);
EEG = eeg_checkset(EEG);
pop_topoplot(EEG,0,[1:20]);


%% Figure 5. Parameter-based model clustering
subj_id = 2;
numMod = 20;

amicaout_dir = sprintf('Subj%d_M%d',subj_id,numMod);
ourdir = ['C:\Users\shawn\Desktop\Emotion\' amicaout_dir];
modout = loadmodout15(ourdir);

corrTh = 0.8;
sorted_corr = cell(numMod);
numIC_highcorr = nan(numMod);

for model_1 = 1:numMod-1
    for model_2 = model_1+1:numMod
        A1 = modout.A(:,:,model_1);
        A2 = modout.A(:,:,model_2);
        
        [corr_tmp,indx,indy,corrs] = matcorr(A1',A2',0,2);
        sorted_corr{model_1,model_2} = abs(corr_tmp);
        numIC_highcorr(model_1,model_2) = find(abs(corr_tmp)<corrTh,1)-1;
    end
end

figure, plot(sorted_corr{1,2},'linewidth',2); xlabel('Component Numbering'); ylabel('Coorelation of matched ICs'); set(gca,'fontsize',12);
hold on, plot([0 250],[corrTh corrTh],'--r');

figure, imagesc(numIC_highcorr);

% nomarlize distance measure
distMat = (numIC_highcorr - max(numIC_highcorr(:))) / (min(numIC_highcorr(:)) - max(numIC_highcorr(:)));
distVec = [];
linkMethod = 'average';
for it1 = 1:size(distMat,1)-1
    distVec = [distVec, distMat(it1, it1+1:end)];
end

% hierarchical clustering
clusterLink = linkage(distVec,linkMethod);

figure, 
[H,T,outperm] = dendrogram(clusterLink, 0);
ylabel('Norm num of 0.8 correlated ICs')
set(gca,'fontsize',12);


%% Examine LL(t) for different numbers of AMICA models
numMod = 3:20;
subj_id = 2;
LL_all = cell(1,length(numMod));
BIC = zeros(1,length(numMod));
AIC = zeros(1,length(numMod));
CAIC = zeros(1,length(numMod));
LL_sum = zeros(1,length(numMod));
for it = 1:length(numMod)
    
    amicaout_dir = sprintf('Subj%d_M%d',subj_id,numMod(it));
    ourdir = ['C:\Users\shawn\Desktop\Emotion\' amicaout_dir];
    modout = loadmodout15(ourdir);
    
    % log likelihood
    LL_all{it} = modout.LL;
    
    % number of parameters
    numPara = length(modout.W(:)) + length(modout.c(:)) + length(modout.alpha(:)) + ...
        length(modout.mu(:)) + length(modout.sbeta(:)) + length(modout.rho(:));
    
    % Bayesian Information Criteria (BIC)
    num_rej_data = sum(modout.Lt == 0);
    BIC(it) = log(length(modout.Lt)-num_rej_data) * numPara - 2*sum(modout.Lt);
    
    % Akaike Information Criteria (AIC)
    AIC(it) = 2*numPara - 2*sum(modout.Lt);
    
    % Corrected AIC
    CAIC(it) = 2*numPara - 2*sum(modout.Lt) + 2*numPara*(numPara+1) / (length(modout.Lt)-num_rej_data-numPara-1);

    LL_sum(it) = sum(modout.Lt);
end


figure, hold on,
for it = 1:length(numMod)
    plot(LL_all{it});
    mark_pos = 1200 - it*50;
    text(mark_pos,LL_all{it}(mark_pos),num2str(numMod(it)),'fontsize',12);
end
ylim([-2.353 -2.328])
set(gca,'fontsize',12);
xlabel('Learning Steps'); ylabel('Log Likelihood');


figure,
subplot(3,1,1); plot(numMod,BIC,'linewidth',2); ylabel('BIC');
subplot(3,1,2); plot(numMod,AIC,'linewidth',2);ylabel('AIC');
subplot(3,1,3); plot(numMod,CAIC,'linewidth',2);ylabel('CAIC');
xlabel('Number of AMICA models')

figure,
plot(numMod,LL_sum)


%% microstate analysis
% EEG = pop_loadset('filename','EEG_Subj_2.set','filepath','C:\\Users\\shawn\\Desktop\\Emotion\\');
% EEG = eeg_checkset( EEG );
% [EEG,com] = pop_FindMSTemplates(EEG, struct('MinClasses', 4, 'MaxClasses', 10, 'GFPPeaks', 1, 'IgnorePolarity', 1, 'MaxMaps', 1000, 'Restarts', 5, 'UseAAHC', 0), 0, 0);
% EEG = eeg_checkset( EEG );
% [ALLEEG,EEG,com] = pop_ShowIndMSMaps(EEG, 6, 0, ALLEEG);
% com = pop_ShowIndMSDyn(ALLEEG, EEG, 0, struct('b',0,'lambda',3.000000e-01,'PeakFit',1,'nClasses',6,'BControl',1));
% EEG = eeg_checkset( EEG );
% 
% tmp = load('MSClass_6.mat');
% MSClass = tmp.MSClass;


