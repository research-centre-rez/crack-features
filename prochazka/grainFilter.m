function[GrainStruct,GrainMap,Layers,SizeLimit]=grainFilter(GrainStruct,GrainMap,Layers,varargin)
close all
T0=datetime();
SizeLimit=9000;

Sizes=vertcat(GrainStruct(:).Numel);
Indices=sort(find(Sizes>=SizeLimit));
Bins=logspace(0,6,25);
[Bars1,Bins]=weightcounts(Sizes,Bins,'normalization','probability');
kk=0;
for ii=1:numel(Sizes)
    if ismember(ii,Indices)
        kk=kk+1;
        GrainStruct(ii).ID=kk;
        GrainMap(GrainMap==ii)=kk;
    else
        GrainMap(GrainMap==ii)=0;
    end
end
GrainStruct=GrainStruct(Indices);

GrainStruct(end+1).ID=numel(GrainStruct)+1;
GrainStruct(end).PhaseID=7
GrainStruct(end).Label="Matrix"
% GrainStruct(end).Numel=sum(GrainMap<1,'all');
Area=sum(GrainMap<1,'all');
GrainStruct(end).Numel=Area;
Edge=sum(xor(GrainMap<1,bwmorph(GrainMap<1,'shrink',1)),'all');
GrainStruct(end).Edge=Edge
GrainStruct(end).E2A=Edge/Area

GrainMap(GrainMap==0)=numel(GrainStruct);


for ii=1:numel(Layers)
    Layers(ii).Children=unique(GrainMap(Layers(ii).Map>0));
end



[Bars2]=weightcounts(vertcat(GrainStruct(:).Numel),Bins,'Normalization','probability');
X=Bins(1:end-1)+diff(Bins);
plot(X',[Bars1;Bars2]','marker','.')
set(gca,'xscale','log','yscale','log')
hold on
plot(SizeLimit*[1 1],get(gca,'YLim'))
T1=datetime();
fprintf('grainFilter: %s\n',T1-T0);
end