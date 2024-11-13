function[Counts,Lengths,N,L]=cracksCount(Cracks,Phases)

Edges=vertcat(Cracks(:).OnEdge)
NP=max(Edges(:,3:4),[],'all')
Counts=nan(NP);
Lengths=Counts;

for ii=1:NP
    Temp=Edges(Edges(:,3)==ii,:);
    E=(ii:NP+1)-.5;
    [L,E,~,N]=weightcounts(Temp(:,4),E);
    B=E(1:end-1)+diff(E)/2;
%     [B;N]
    Counts(B,ii)=N;
    Lengths(B,ii)=L;
end
Counts
% return
Through=vertcat(Cracks(:).Through);

[L,E,~,N]=weightcounts(Through(:,2),'BinMethod','integers');
B=E(1:end-1)+diff(E)/2
% bar(N)
% set(gca,'xticklabel',[phases(:).Labels])
close all
figure('Position',[1 32 958 964])%1922
TC=tiledlayout(3,3,'TileSpacing','compact','Padding','compact');
figure('Position',[960 32 958 964])%2882
TL=tiledlayout(3,3,'TileSpacing','compact','Padding','compact');

for ii=1:NP
    Counts(ii,:)=Counts(:,ii);
    Lengths(ii,:)=Lengths(:,ii);

%     Counts(ii,ii)=N(ii);
%     Lengths(ii,ii)=L(ii);

    nexttile(TC);
    bar(1:length(Counts),Counts(:,ii))
    title(sprintf('Crack on edge of %s with:',Phases(ii).Labels))
    set(gca,'xticklabel',[Phases(:).Labels],'YLim',[0 180])
end
nexttile(TC)
bar(1:length(Counts),N)
title(sprintf('Crack in the phase'))
set(gca,'xticklabel',[Phases(:).Labels],'YLim',[0 720])

nexttile(TL,[3,3]);
bar(Counts)
set(gca,'xticklabel',[Phases(:).Labels])
legend([Phases(:).Labels])
Counts
Lengths