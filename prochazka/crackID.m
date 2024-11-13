function[Cracks,Meat]=crackID(Skel,Meat)

% Enumerate the cracks
Meat=bwlabel(Meat);
% Align Skeleton enumeration with cracs enumeration
Skel=Meat.*double(Skel);

% Count individual cracks
N=max(Meat(:));

% Declare the cracks structure 
Cracks(N)=struct(...
    'AreaPx',nan,...
    'SkelPxJB',nan,...
    'SkelPxJP',nan,...
    'WidthPx',nan,...
    'Meat',sparse(size(Meat)),...
    'Skel',sparse(size(Meat)));

% return
for ii=1:N
    % disp(sprintf('%d/%d',ii,N))
    Cracks(ii).AreaPx=sum(Meat==ii,'all');
    Cracks(ii).SkelPxJB=sum(Skel==ii,'all');
    Cracks(ii).WidthPx=Cracks(ii).AreaPx/Cracks(ii).SkelPxJB;
    Cracks(ii).Meat=sparse(Meat==ii);
    Cracks(ii).Skel=sparse(Skel==ii);
%     Group=segment_labels==ii;
%     imagesc(Group);
%     Group=connectGroup(Group);
    
%     [l,k]=find(skeleton==ii);
%     line(k,l,'color','r')
end
    function OUT=connectGroup(IN)
        IN=bwmorph(IN,'remove');
        IN=bwlabel(IN);
        CGN=max(IN(:));
        while CGN>1
            Piece=IN==1;
            Rest=xor(Piece,IN);
            [P,R]=findClosest(Piece,Rest);
            OUT=Piece+2*Rest;
            CGN=1;
            line(P,R,'marker','*','color','m')
        end
%         OUT=IN;
    end
    function[A,B]=findClosest(Piece,Rest)
        [P2,P1]=find(Piece>0);
        [R2,R1]=find(Rest>0);
        Distances=nan(numel(P2),numel(R2));
        for FCi=1:numel(P2)
            for FCj=1:numel(R2)
                Distances(FCi,FCj)=sqrt((P2(FCi)-R2(FCj))^2+(P1(FCi)-R1(FCj))^2);
            end
        end
        min(Distances(:))
        [P0,R0]=find(Distances==min(Distances(:)));
        B=P2(P0);
        A=P1(P0);
    end
end