function [Cracks,NodStruct,BranchStruct]=cracksEval(PhaseIMG,Cracks,Phases,ThRadius,InRadius)
    % some visualization
    close all
    figure('position',[1921 31 1920 973]);
    subplot(1,2,1)
    imagesc(PhaseIMG);
    subplot(1,2,2)

    % Init variables, allocate memory

    % imagesc(phase_img);hold on
    NC=numel(Cracks);
    [MI,NI]=size(PhaseIMG);

    % Add not-assigned phase (0 in phase_img)
    Phases(end+1).Mask=PhaseIMG==0;
    Phases(end).Label="n/a";
    Phases(end).LABcolor=[0 0 0];
    Phases(end).GRBcolor=[0 0 0];
    Phases(end).AreaPx=sum(Phases(end).Mask);

    LayerID=PhaseIMG;
    LayerID(PhaseIMG==0)=numel(Phases);

    % Number of Cracks
    CracksList=1:NC;%581

    % Build data structures
    for ii = CracksList
        sprintf('%d / %d',ii,NC)

        % Extract crack map
        % cracks
        Meat=full(Cracks(ii).Meat);
        % Extract the crack outline
        Outline=xor(Meat,imdilate(Meat,strel('disk',ThRadius)));
        % Build simpler skeleton
        Skel=bwskel(Outline+Meat>0);
        % imagesc(segment_labels+Outline)
        Cracks(ii).SkelPxJP=sum(Skel,'all');
        % Find skeleton endpoints (coordinates)
        [l,k]=find(bwmorph(Skel,'endpoints'));

        % NOTE: This part looks like some visualization, but it changes the skeleton data
        if numel(l)==0
            [L,K]=find(Skel);
            L=L(1);
            K=K(1);
            Skel(L,K)=0;
            [l,k]=find(bwmorph(Skel,'endpoints'));
            l(2)=L;
            k(2)=K;
            Sk=zeros(size(Skel));
            Sk(l(1),k(1))=1;
            Sk(l(2),k(2))=1;
            imagesc(Outline+Meat+Skel+Sk)
        end
        EndPoints=[k,l];

        % Find skeleton nodes
        Nods=bwmorph(Skel,'branchpoints');

        % Find and label crack branches
        Branches=bwlabel(Skel - imdilate(Nods, strel('disk',2)) > 0);

        % Find nodes' coordinates
        [l,k]=find(Nods);
        NodPoints=[k,l];

        % Process EndPoints
        clear EndStruct
        EndStruct(1:size(EndPoints,1),1)=struct('xy',nan,'Branches',nan,...
            'IsEnd',false,'IsTerminal',false);
        for jj=1:size(EndPoints,1)
            K=EndPoints(jj,1);
            L=EndPoints(jj,2);
            EndStruct(jj).xy=[K,L];
            Blist=unique(...
                Branches(min(max(1,L-4:L+4),size(PhaseIMG,1)),...
                min(max(1,K-4:K+4),size(PhaseIMG,2))));
            EndStruct(jj).Branches=Blist(2:end);
            EndStruct(jj).IsEnd=true;
        end

        % Find the furthest EndPoints
        Distances=nan(size(EndPoints,1));
        for jj=1:size(EndPoints,1)
            Points=vertcat(EndStruct.xy);
            Point=EndStruct(jj).xy;
            XY=[Points(:,1)-Point(1),Points(:,2)-Point(2)];
            Distances(:,jj)=sqrt(sum(XY.^2,2));
        end
        % Kde se bere druhý index?
        [Hits(1),Hits(2)]=find(Distances==max(Distances(:)),1);
        [EndStruct(Hits).IsTerminal]=deal(true);

        % Assign nods structure to contain coordinates and branch indices
        clear NodStruct
        NodStruct(1:numel(l),1)=struct('xy',nan,'Branches',nan,...
            'IsEnd',false,'IsTerminal',false);
        for jj=1:numel(l)
            NodStruct(jj).xy=[k(jj),l(jj)];
            Blist=unique(...
                Branches(min(max(1,l(jj)-4:l(jj)+4),size(PhaseIMG,1)),...
                min(max(1,k(jj)-4:k(jj)+4),size(PhaseIMG,2))));
            NodStruct(jj).Branches=Blist(2:end);
            NodStruct(jj).IsEnd=false;
        end

        % Merge EndPoints with Nodes
        NodStruct=vertcat(NodStruct, EndStruct);
        NBranch=max(Branches(:));
        IncidentPoints=padhorzcat(NodStruct.Branches)';
        clear BranchStruct
        BranchStruct(NBranch)=struct('Map',nan,'Nodes',nan,'xy',nan,...
            'OnEdge',nan(0,3),'Through',nan(0,2));
        for jj = 1:NBranch
            [Hits,~] = find(IncidentPoints==jj);
            BranchStruct(jj).Map = Branches==jj;
            BranchStruct(jj).Nodes = Hits;
            BranchStruct(jj) = sortPoints(BranchStruct(jj), NodStruct);
            [Phs,Is] = sniffSides(BranchStruct(jj), PhaseIMG);
            Phs=Phs(min(Phs,[],2)~=0,:);
            BranchStruct(jj).Through=Phs(Phs(:,1)==Phs(:,2),:);
            BranchStruct(jj).OnEdge=Phs(Phs(:,1)~=Phs(:,2),:);
            BranchStruct(jj).OnEdge;
        end

        % Connect main crack
        Through=vertcat(BranchStruct(:).Through);
        OnEdge=vertcat(BranchStruct(:).OnEdge)

        % Find and count pixels inside one phase
        Through=Through(:,1);
        Unq=unique(Through);

        % ThrS=size(Unq)
        for jj=1:numel(Unq)
            Unq(jj,2)=Phases(Unq(jj,1)).PhaseID;
            Unq(jj,3)=sum((Through==Unq(jj,1)),'all');
        end
        Cracks(ii).Through=Unq;

        % Find and count pixels on the phase edges
        OnEdge=sort(OnEdge,2);
        Unq=unique(OnEdge,'rows');  # grain_A, grain_B
        for jj=1:size(Unq,1)
            Unq(jj,3)=Phases(Unq(jj,1)).PhaseID; # phase_A
            Unq(jj,4)=Phases(Unq(jj,2)).PhaseID; # phase_B
            Unq(jj,5)=sum(prod(Unq(jj,1:2)==OnEdge,2)); # count
        end

        % Unq
        Cracks(ii).OnEdge=Unq;
    end
    Cracks=Cracks(CracksList);
end

function [Ph,I] = sniffSides(BStruct, Map)
    % number of points on the branch
    sN = size(BStruct.xy, 1);

    % allocate memory for the outputs
    I = nan(sN, 4); % for each BStruct creates 4 nans
    Ph = nan(sN, 2); % for each BStruct creates 2 nans

    for si = 1:sN % go thru all branch points
        % i1, i2
        i1 = max(si - 1, 1);
        i2 = min(si + 1, sN);
        I1 = BStruct.xy(i2,[2,1]);
        I2 = BStruct.xy(i1,[2,1]);
        T = I2 - I1;
        N = T * [0 -1;1 0];
        N = N / norm(N);
        J0 = BStruct.xy(si,[2,1]); % bod na branchi
        IsPhase = false;
        kk = 1;
        while ~IsPhase
            J10=round(J0 - kk * N); % sousední bod ve směru normály
            J1=min([max([J10; 1 1]); size(Map)]);
            if J10 == J1
                IsPhase=Map(J1(1), J1(2)) ~= 0;
            else
                IsPhase = true;
            end
            kk = kk + 1;
        end
        IsPhase = false;
        kk = 1;
        while ~IsPhase
            J20 = round(J0 + kk * N);
            J2 = min([max([J20; 1 1]); size(Map)]);
            if J20==J2
                IsPhase = Map(J2(1), J2(2)) ~= 0;
            else
                IsPhase = true;
            end
            kk = kk + 1;
        end
        I(si,:) = [J1, J2];
        Ph(si,:) = [Map(J1(1),J1(2)), Map(J2(1), J2(2))];
    end
end
