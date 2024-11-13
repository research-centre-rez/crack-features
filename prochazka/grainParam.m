function[GrainStruct,Layers]=grainParam(GrainMap,PhMap,Layers)
T0=datetime();
NG=max(GrainMap,[],'all');
GrainStruct(NG)=struct('ID',nan,'PhaseID',nan,'Label',"",...
    'Numel',nan,'Edge',nan,'E2A',nan);
for ii=unique(PhMap)'
    TempMap=(PhMap==ii).*GrainMap;
    TempIDs=unique(TempMap);
    TempIDs=TempIDs(TempIDs~=0);
    Label=Layers(ii).Label;
    Layers(ii).Children=TempIDs;
    parfor jj=TempIDs'
        Grain=GrainMap==jj;
        GrainStruct(jj).PhaseID=ii;
        GrainStruct(jj).Label=Label;
        Area=sum(Grain,'all');
        GrainStruct(jj).Numel=Area;
        Edge=sum(xor(Grain,bwmorph(Grain,'shrink',1)),'all');
        GrainStruct(jj).Edge=Edge
        GrainStruct(jj).E2A=Edge/Area;
    end
T1=datetime;
fprintf('grainParam %d/%d: %s\n',ii,numel(unique(PhMap)),T1-T0);
end

T1=datetime;
fprintf('grainParam: %s\n',T1-T0);
parfor ii=1:numel(GrainStruct)
    GrainStruct(ii).ID=ii;
end
T1=datetime;
fprintf('grainParam: %s\n',T1-T0);
end