function GrainSet=grainID2(Map,Layers)
T0=datetime();
subplot(1,2,1,'replace')
imagesc(Map)

Probe1=strel('disk',10);
Probe2=strel('disk',5);
    GrainSet=zeros(size(Map));
% Index to keep last grain assigned info
LastIndex=0;
    AX1=subplot(1,2,1,'replace');
AX2=subplot(1,2,2);
    % cycle thorugh phases
for ii=unique(Map)'
    % Label individual grains in ii-th phase
    Labels=bwlabel(Map==ii);

%     imagesc(AX1,Labels)
% imagesc(Labels==ii)
    cmap=colormap;
    cmap(1,:)=[0 0 0];
    colormap(cmap)
    %cycle through the grains
    for jj=1:max(Labels,[],'all')
        % Get the jj-th grain
        Grain=Labels==jj;

        % Shrink the grain to erase bottlenecks 
        Grain2=imopen(Grain,Probe1);

        % Label residual subgrains
        SubGrains=bwlabel(Grain2);
%         imagesc(AX1,SubGrains)
        % assign matrix for new grains sets
        TheGrains=zeros(size(Map));

        % If there was one subgrain detected it is the whole grain
        if max(SubGrains,[],'all')==1
            TheGrains=bwlabel(Grain);
            imagesc(AX1,TheGrains);
            drawnow
        else
            % cycle through detected subgrains
            for kk=1:max(SubGrains,[],'all')
                % get kk-th subgrain
                SubGrain=SubGrains==kk;
                % Expand the subgrain
                SubGrain=imdilate(SubGrain,Probe2);
                imagesc(AX1,SubGrain&Grain)
                % Assign the subgrain extended to extent of original grain
                TheGrains(SubGrain&Grain)=kk;
                drawnow
            end
    
            % Label areas that weren't assigned in previous step
            Leftovers=bwlabel(xor(TheGrains,Grain));
    %         imagesc(AX1,TheGrains)
    %         return
            % cycle through missing areas
            for kk=1:max(Leftovers,[],'all')%21
                % get the kk-th unassigned area
                Chunk=Leftovers==kk;
                % Get the neighbouring pixels
                Outline=xor(Chunk,bwmorph(Chunk,'thicken',2));
                % Get the subgrains indices that surround the unassigned area
                Hits=TheGrains.*Outline;
                % List the unique labels and exclude zero
                imagesc(Hits)
                Indices=unique(Hits);
                Indices=Indices(Indices>0);
                
                % Find the most common neighbouring label
                % sort the histogram output for each label
                [~,I]=sort(histc(Hits(:),Indices),'descend'); %#ok<HISTC> 
                % if sorting is successful, assign the most common label
                % otherwise the chunk is an island and assign
                % the default value 1
                if ~isempty(I)
                TheGrains(Chunk==1)=Indices(I(1));
                else
                    TheGrains(Chunk==1)=1;
                end
            end
        end
    % mark the zero pixels as none
    TheGrains(TheGrains<1)=nan;
    % Increment all new labels so they won't collide with prevous ones
    TheGrains=TheGrains+LastIndex;
    % mark nan pixels as zeros
    TheGrains(isnan(TheGrains))=0;
    % Add new labels to the grain map
    GrainSet=GrainSet+TheGrains;
    % get the last index to protect the matrix from collisions
    LastIndex=max(GrainSet,[],'all')
        imagesc(AX2,GrainSet)
        drawnow
    end %jj
end %ii
T1=datetime;
disp(duration(T1-T0))
end