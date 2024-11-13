function batchPhase(PhaseDef)
 Root=fullfile('D:','OneDrive','UJV',...
    'Halodova Patricie - JCAMP_NRA_LOM_SEM-EDX',...
    'SEM-EDX');
Samples=dir(Root);
Samples={Samples.name};
% Turncate auxiliary folders . and ..
Samples=string(Samples(3:end)');

% CropRange=[23,4630;22,4720]
CropRange=[1,2000;1,2000];

Cracks=nan;
CracksMap=nan;
Layers=nan;
Map2=nan;
Map=nan;
GrainStruct=nan;
GrainMap=nan;
Skel=nan;
Mask=nan;

t0=datetime;
i0=0;
Is='3';%3%:numel(Samples)% Testing case -3-
LogFile=fullfile(Root,'log_V3.txt');
FID=fopen(LogFile,'a');
% fprintf(FID,'\n\n%s: batchPhase, Is = %s',...
%     t0,Is);
fclose(FID);

%% run through all samples found
for ii=stringArray(Is) 
    ti=datetime;
    i0=i0+1;
    % Find all EDX layered images for phase analyses
    Maps=dir(fullfile(Root,Samples(ii),'EDX layered images','*.tif*'));
    
    Js='11'%['1:' num2str(numel(Maps))]; %testing case -11-
    
    FID=fopen(LogFile,'a');
%     fprintf(FID,'\n%s: ii = %d (%d/%d) Js = %s',...
%         ti,ii,i0,numel(stringArray(Is)),Js);
    fclose(FID);
    %% Run through all images 
    % (all image orderings must match, file names are not tested)
%     Js=1:numel(Maps);
    j0=0;
    for jj=stringArray(Js)
        j0=j0+1;
        tj=datetime;
        FID=fopen(LogFile,'a');
%         fprintf(FID,'\n%s - %s: process started, jj=%d (%d/%d)',...
%             tj,fullfile(Samples(ii),Skeletons(jj).name),jj,j0,numel(Js));
        fclose(FID);

        Folder=regexp(Maps(jj).folder,filesep,'split');
        Folder=fullfile(Folder{1:end-1});
        Name=regexp(Maps(jj).name,'\.','split');
        Name=Name{1};
        MapFile=fullfile(Maps(jj).folder,Maps(jj).name);
        MaskFile=fullfile(Folder,'JBmasks',[Name '.png'])

        if ~isfile(MaskFile)
            t=datetime;
            FID=fopen(LogFile,'a');
            fprintf(FID,'\n%s - !Missing mask file!: %s:, jj=%d (%d/%d)',...
                t-tj,MaskFile,jj,j0,numel(stringArray(Js)));
            fclose(FID);

            continue
        end
    
        Map=imread(fullfile(Maps(jj).folder,Maps(jj).name));
        MapSize=size(Map);
        CropRange(:,2)=min(CropRange(:,2),MapSize(1:2)');
        Map=Map(...
            CropRange(1,1):CropRange(1,2),...
            CropRange(2,1):CropRange(2,2),...
            1:3);
        Mask=double(imread(MaskFile)>0);
        Mask=Mask(...
            CropRange(1,1):CropRange(1,2),...
            CropRange(2,1):CropRange(2,2));

        PhaseMap=phaseTHR(Map,PhaseDef,'-cracks',Mask,'-gauss',3,'-thr',5,'-lab');
    end
end
end