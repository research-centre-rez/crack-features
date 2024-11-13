function[Layers,Map2,Map21,Cracks,CracksMap,Map,GrainStruct,GrainMap0,GrainMap,Skel,Mask,TL]=initializeJB(ChPhases)
close all
% Testing folder
MainFolder=fullfile('D:','OneDrive','UJV',...
    'Halodova Patricie - JCAMP_NRA_LOM_SEM-EDX',...
    'SEM-EDX');
Samples=dir(MainFolder);
Samples={Samples.name};
% Turncate auxiliary folders . and ..
Samples=string(Samples(3:end)');

% Crop element labels in the bottom part of the image
CropRange=[1,2000;1,2000];

Cracks=nan;
CracksMap=nan;
Layers=nan;
Map2=nan;
Map21=nan;
Map=nan;
GrainStruct=nan;
GrainMap=nan;
GrainMap0=nan;
Skel=nan;
Mask=nan;

figure()
TL=tiledlayout(1,1,'TileSpacing','tight','Padding','none');
% axl=nexttile(TL);
% axr=nexttile(TL);
axl=axes;
axr=axl;


t0=datetime;
i0=0;
Is='3';%3%:numel(Samples)% Testing case -3-

FID=fopen(fullfile(MainFolder,'log.txt'),'a');
fprintf(FID,'\n\n%s: BatchRun, Is = %s',...
    t0,Is);
fclose(FID);
%% run through all samples found
for ii=stringArray(Is) 
    ti=datetime;
    i0=i0+1;
    % Find all EDX layered images for phase analyses
    EDX_Map_list=dir(fullfile(MainFolder,Samples(ii),'EDX layered images','*.tif*'));
    % Find all cracks images (processed by Jan Blazek)
    Mask_list=dir(fullfile(MainFolder,Samples(ii),'JBmasks','*.png'));
    % Find all skeleton images (processed by Jan Blazek)
    Skeleton_lists=dir(fullfile(MainFolder,Samples(ii),'JBskeletons','*.png'));
    Js='2'%'11'%['1:' num2str(numel(Maps))]; %testing case -11-, preview case -2-
    
    FID=fopen(fullfile(MainFolder,'log.txt'),'a');
    fprintf(FID,'\n%s: ii = %d (%d/%d) Js = %s',...
        ti,ii,i0,numel(stringArray(Is)),Js);
    fclose(FID);
    %% Run through all images 
    % (all image orderings must match, file names are not tested)
%     Js=1:numel(Maps);
    j0=0;
    for jj=stringArray(Js)
        tj=datetime;
        j0=j0+1;
        FID=fopen(fullfile(MainFolder,'log.txt'),'a');
        fprintf(FID,'\n%s - %s: process started, jj=%d (%d/%d)',...
            tj,fullfile(Samples(ii),Skeleton_lists(jj).name),jj,j0,numel(Js));
        fclose(FID);

        Cracks=nan;
        CracksMap=nan;
        Layers=nan;
        Map2=nan;
        Map=nan;
        GrainStruct=nan;
        GrainMap=nan;
        Skel=nan;
        Mask=nan;

        try
        %% Read data
        
        FileName=regexp(EDX_Map_list(jj).name,'\.','split');
        FileName=FileName{1};
        % Read the EDX map (4-chanell tiff), remove scale,
        % legend and alpha layer
        Map=imread(fullfile(EDX_Map_list(jj).folder,EDX_Map_list(jj).name));
        MapSize=size(Map);
        CropRange(:,2)=min(CropRange(:,2),MapSize(1:2)');
        Map=Map(...
            CropRange(1,1):CropRange(1,2),...
            CropRange(2,1):CropRange(2,2),...
            1:3);


        % Read crack image, reprocess in binary image (double, not logical)
        % and align with EDX map
        Mask=double(imread(fullfile(Mask_list(jj).folder,Mask_list(jj).name))>0);
        Mask=Mask(...
            CropRange(1,1):CropRange(1,2),...
            CropRange(2,1):CropRange(2,2));
        % Read skeleton image, reprocess in binary image (double, not logical)
        % and align with EDX map
        Skel=double(imread(fullfile(Skeleton_lists(jj).folder,Skeleton_lists(jj).name))>0);
        Skel=Skel(...
            CropRange(1,1):CropRange(1,2),...
            CropRange(2,1):CropRange(2,2));
        if max(bwlabel(Mask),[],'all')==0||max(bwlabel(Skel),[],'all')==0
            FID=fopen(fullfile(MainFolder,'log.txt'),'a');
            fprintf(FID,'\n%s - %s: no cracks',datetime,fullfile(Samples(ii),Skeleton_lists(jj).name));
            fclose(FID);
        end
        

        %% Process data
        
        % Process the phase image

        % Threshold the EDX compound image using proximity fiter in L*a*b*
        % colourspace
        [Map21,Layers]=phaseTHR(Map,ChPhases,'-cracks',Mask,'-gauss',3,'-thr',5,'-lab');

        % Process detected phases, remove small areas and extrapolate
        % non-detected parts for surroundings

        GrainMap0=grainID2(Map2,Layers);

        % Process grains
        [GrainStruct,Layers]=grainParam(GrainMap0,Map2,Layers);

        % Filter tiny grains and mark as Matrix
        [GrainStruct,GrainMap,~,SizeLimit]=grainFilter(GrainStruct,GrainMap0,Layers);

        Map2(Mask==1)=0;
%         Identify cracks

[Cracks,CracksMap]=crackID(Skel,Mask);  
%         Assign phases to cracks
         Cracks=cracksEval(GrainMap,Cracks,GrainStruct,3,2);
         [EN,EL,IN,IL]=cracksCount(Cracks,ChPhases);
         % Save results in .mat file
        if ~exist(fullfile(MainFolder,Samples(ii),'Stats'),'dir')
            mkdir(fullfile(MainFolder,Samples(ii),'Stats'))
        end
        save(fullfile(MainFolder,Samples(ii),'Stats',FileName),...'EN','EL','IN','IL',...
            'ChPhases',...
            'Layers','GrainStruct','Cracks',...
            'GrainMap','Map2','CracksMap');

        t=datetime;
        FID=fopen(fullfile(MainFolder,'log.txt'),'a');
        fprintf(FID,'\n%s - %s: process ended. Duration: %s',t,fullfile(Samples(ii),Skeleton_lists(jj).name),t-tj);
        fclose(FID);
        catch ME
            ME.stack
            t=datetime;
            FID=fopen(fullfile(MainFolder,'log.txt'),'a');
            fprintf(FID,'\n%s - %s: Error.',t,fullfile(Samples(ii),Skeleton_lists(jj).name));
            fclose(FID);
        
            save(fullfile(MainFolder,Samples(ii),'Stats',FileName),...'EN','EL','IN','IL',...
                'ME','ChPhases',...
                'Mask','Skel',...
                'Layers','GrainStruct','Cracks',...
                'GrainMap','Map2','CracksMap');
        end
    end
    t=datetime;
    FID=fopen(fullfile(MainFolder,'log.txt'),'a');
    fprintf(FID,'\n%s - %s: Set ended. Duration: %s',t,Samples(ii),t-ti);
    fclose(FID);
end
t=datetime;
FID=fopen(fullfile(MainFolder,'log.txt'),'a');
fprintf(FID,'\n%s: Batch ended. Duration: %s',t,t-t0);
fclose(FID);
    
end

% set(gcf,'Position',[1 1 1914 795])