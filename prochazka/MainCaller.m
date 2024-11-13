function[Layers,Map2,Map21,Cracks,CracksMap,Map,GrainStruct,GrainMap0,GrainMap,Skel,Mask,TL]=MainCaller(ChPhases)
close all
% Testing folder
MainFolder=fullfile('D:','OneDrive','UJV',...
    'Halodova Patricie - JCAMP_NRA_LOM_SEM-EDX',...
    'SEM-EDX');
Samples=dir(MainFolder);
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

% vyrobí figure rozdělenou na dva grafy
figure()
TL=tiledlayout(1,2,'TileSpacing','tight','Padding','none');
axl=nexttile(TL);
axr=nexttile(TL);

CMap0=[0,0,0;vertcat(ChPhases(vertcat(ChPhases.Detect)>0).Colors)];
Labels0=["Unassigned";vertcat(ChPhases(vertcat(ChPhases.Detect)>0).Labels)]

t0=datetime;
i0=0;
Is='3';%3%:numel(Samples)% Testing case -3-

FID=fopen(fullfile(MainFolder,'log.txt'),'a');
fprintf(FID,'\n\n%s: initializeJB, Is = %s',...
    t0,Is);
fclose(FID);
%% run through all samples found
for ii=stringArray(Is) % nedefinovaná proměnná
    ti=datetime;
    i0=i0+1;
    % Find all EDX layered images for phase analyses
    EDX_Map_list=dir(fullfile(MainFolder,Samples(ii),'EDX layered images','*.tif*'));
    % Find all cracks images (processed by Jan Blazek)
    Mask_list=dir(fullfile(MainFolder,Samples(ii),'JBmasks','*.png'));
    % Find all skeleton images (processed by Jan Blazek)
    Skeleton_lists=dir(fullfile(MainFolder,Samples(ii),'JBskeletons','*.png'));
    Js='2'%'11'%['1:' num2str(numel(Maps))]; %testing case -11-
    
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
% return
        % Read the EDX map (4-chanell tiff), remove scale,
        % legend and alpha layer
        Map=imread(fullfile(EDX_Map_list(jj).folder,EDX_Map_list(jj).name));
        MapSize=size(Map);
        CropRange(:,2)=min(CropRange(:,2),MapSize(1:2)');
        Map=Map(...
            CropRange(1,1):CropRange(1,2),...
            CropRange(2,1):CropRange(2,2),...
            1:3);
%         Plot
%         imagesc(axl,mask)
%         title(axl,'EDX')
%         axis(axl,'equal')
%         imagesc(axr,imgaussfilt(mask,3))%,hold on
%         title(axr,'EDX (gauss filter 3)')
%         axis(axr,'equal')
%         return

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
%             continue
        end
        
%         imagesc(axl,mask)
%         imagesc(axr,2-Mask-skeleton)
%         title(axl,'EDX')
%         title(axr,'Crack with skeleton')
%         colormap(axr,'bone')
%         return
        %% Process data
        
        % Process the phase image

        % Threshold the EDX compound image using proximity fiter in L*a*b*
        % colourspace
        [Map21,Layers]=phaseTHR(Map,ChPhases,'-cracks',Mask,'-gauss',3,'-thr',5,'-lab');

        % Process detected phases, remove small areas and extrapolate
        % non-detected parts for surroundings
%         imagesc(axl,Map21)
%         imagesc(axr,Map21)
%         title(axl,'Original EDX image')
%         title(axr,'Labelled map')
%         colormap(axl,CMap0)
%         colormap(axr,CMap0)
%         axis([axl,axr],'equal')
%         colorbar(axl,'ticks',(1:7)*6/7-.5,'ticklabels',Labels0)
%         colorbar(axr,'ticks',(1:7)*6/7-.5,'ticklabels',Labels0)
%         return
        % Process detected phases, remove small areas and extrapolate
        % non-detected parts for surroundings
%         imagesc(axl,Map21)
%         colormap(axl,CMap0(1:end,:))

%         [Layers,Map2]=phaseID(Layers,'-spikes',2,'-ni','-cracks',Mask);
        [Layers,Map2]=phaseID(Layers,'-spikes',2,'-ni');
        
%         imagesc(axr,Map2)
%         colormap(axr,CMap0(2:end,:))
%         colorbar(axl,'ticks',(1:7)*6/7-.5,'ticklabels',Labels0)
%         colorbar(axr,'ticks',((0:7)+.5)*7/8,'ticklabels',Labels0)

% return
        GrainMap0=grainID2(Map2,Layers);

%         imagesc(axl,GrainMap0)
%         return
        [GrainStruct,Layers]=grainParam(GrainMap0,Map2,Layers);
%         return
        [GrainStruct,GrainMap,~,SizeLimit]=grainFilter(GrainStruct,GrainMap0,Layers);
%         imagesc(axr,GrainMap)

%         return
        Map2(Mask==1)=0;
%         Identify cracks
        [Cracks,CracksMap]=crackID(Skel,Mask);  
% return
%         Assign phases to cracks
%         cracks=cracksEval(GrainMap,cracks,GrainStruct,3,2);
%         [EN,EL,IN,IL]=cracksCount(cracks,ChPhases);
        if ~exist(fullfile(MainFolder,Samples(ii),'Stats'),'dir')
            mkdir(fullfile(MainFolder,Samples(ii),'Stats'))
        end
        save(fullfile(MainFolder,Samples(ii),'Stats',FileName),...'EN','EL','IN','IL',...
            'ChPhases',...
            'Layers','GrainStruct','Cracks',...
            'GrainMap','Map2','CracksMap');
%         break
        subplot(1,2,2)
%         size(Map2)
%         size(Crack)
        imagesc(Map2,'AlphaData',ones(size(Mask)))
        colormap(vertcat([255 0 0],ChPhases.Colors))
        set(gca,'CLim',[0 numel(ChPhases)],'Color','k')
        colorbar
%         legend(ChPhases.Labels)
        title('Phases processed')
%         imagesc(bwlabel(Crack),'AlphaData',Crack)
%         size(Mask)
        
        if ~exist(fullfile(MainFolder,Samples(ii),'EDX_assign'),'dir')
            mkdir(fullfile(MainFolder,Samples(ii),'EDX_assign'))
        end
        set(gcf,'Position',[1 1 1914 795])
        print(fullfile(MainFolder,Samples(ii),'EDX_assign',FileName),'-dpng')

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