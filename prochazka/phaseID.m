function[Layers,TheMap,Verbose]=phaseID(Layers,varargin)
%% Data preallocation
% Estimate recipy length
Nv=max(1,numel(varargin));
% Preallocate struct for storing steps
Verbose(Nv)=struct('Layers',nan,'Map',nan,'Command',{''});
% preallocate filter for actual number of steps
Used=false(1,Nv);
FillCracks=true;
%% Preprocess phases
ii=0;
while(ii<numel(varargin))
    ii=ii+1;
    switch lower(varargin{ii})
        case {'-cracks' '-crack' '-cr'}
            % cracks are not to be extrapolated to phases
            % Last step
            FillCracks=false;
            Cracks=varargin{ii+1};
            ii=ii+1;
        case {'-spikes' '-cleanspikes'}
            % Mark small clusters and thin lines as not-assigned
            % Parameter is diameter of disk structure element
            Layers=cleanSpikes(Layers,varargin{ii+1});
            TheMap=Layers2allMaps(Layers);
            Used(ii)=true;
            Verbose(ii).Layers=Layers;
            Verbose(ii).Map=TheMap;
            Verbose(ii).Command=varargin(ii:ii+1);
            ii=ii+1;
        case {'-extrapolateni','-ni','-extrapolate'}
            % Extrapolate non-assigned pixels according to their
            % neighbourhood
            [Layers,TheMap]=extrapolateNI(Layers,TheMap);
            Used(ii)=true;
            Verbose(ii).Layers=Layers;
            Verbose(ii).Map=TheMap;
            Verbose(ii).Command=varargin(ii);
    end
end
Verbose=Verbose(Used);

% Un-assign the crack pixels
if FillCracks
    Layers=allMaps2Layers(Layers,TheMap);
%     imagesc(Layers(1).mask)
else
    TheMap(Cracks>0)=0;
    Verbose(end+1).Layers=Layers;
    Verbose(end).Map=TheMap;
    Verbose(end).Command={'-cracks',Cracks};
end
% horzcat(Verbose.Command)
% figure,imagesc(TheMap)
% cmap=circshift(vertcat(Layers(:).RGB),1,1);
% cmap(2,:)=[21,105,255];
% cmap(4,:)=[100,190,70];
% colormap(cmap);colorbar
% figure, imagesc(Layers(end).mask)
% imagesc(AllMaps)
%% Nested functions
    function Maps=Layers2allMaps(AllLayers)
        % Build indexed map from layer mask stack

        % number of layers. Last one is not-assigned
        NL=numel(AllLayers);
        Maps=zeros(size(AllLayers(1).Map));
        % cycle through all assigned layers
        for LMi=1:NL-1
            Maps(AllLayers(LMi).Map)=LMi;
        end
    end
    function MapLayers=allMaps2Layers(MapLayers,Map)
        % Update layers stack from indexed map

        % Get index list
        Indices=unique(Map)';

        % Exclude index 0 (not assigned)
        Indices=Indices(Indices~=0);
        % cycle thorugh all indices
        for aLi=Indices
            MapLayers(aLi).Map=Map==aLi;
        end
        % append last layer as not-assigned
        MapLayers(end).Map=~Map;
    end
    function [ENLayers,ENMap]=extrapolateNI(ENLayers,ENMap)
        % Extrapolate non-indexed pixels from their surroundings

        % N/A extinction counter
        NAcount=sum(ENMap<1,'all');
        % Stop when no N/A pixel is assigned to a layer
        while NAcount>0
            % Count N/A pixels
            NAcount1=sum(ENMap<1,'all');
            % Find coordinates of N/A pixels
            [K,L]=find(ENMap<1);
            % Pad mask with zeros (avoid index underflow/overflow)
            ENMap=padarray(ENMap,[1,1],0,'both');
            % Compensate for padding
            K=K+1;
            L=L+1;
            % cycle through all zero pixels
            for eNi=1:numel(K)
                % Find values surrounding eNi-th pixel, arrange them in
                % vector and find unique values and indices where they are.
%                 K(eNi);
%                 L(eNi);
%                 ENMap(K(eNi)-1:K(eNi)+1,L(eNi)-1:L(eNi)+1)
                [LLbl,~,IC]=unique(...
                    reshape(...
                    ENMap(K(eNi)-1:K(eNi)+1,L(eNi)-1:L(eNi)+1),[],1));
                if sum(LLbl,'all')>0
                % Count occurence of each unique value
                Counts=accumarray(IC,1);
                % Bind layer index to index count
                CountArr=[LLbl,Counts];
                % Remove the zero index (N/A)
                CountArr=CountArr(2:end,:);
                % find all indices with highest counts
                IC=find(CountArr(:,2)==max(CountArr(:,2)));
                % if there is only one nonzero index with highest count,
                % assign its value to the map
                if numel(IC)>1
                    IC=IC(round(rand(1)*(numel(IC)-1))+1);
                end
                    ENMap(K(eNi),L(eNi))=CountArr(IC,1);
                end
            end
            % Strip the map off the padding
            ENMap=ENMap(2:end-1,2:end-1);
            % Count N/A pixels
            NAcount2=sum(ENMap<1,'all');
            % Update the extinction counter
            NAcount=NAcount1-NAcount2;
            
        end
        % Update Layers' maps according to new mask
        ENLayers=allMaps2Layers(ENLayers,ENMap);
    end
    function CSLayers=cleanSpikes(CSLayers,Filter)
        % un-assign small cluster phases and thin blades. Features
        % smaller/thinner than [Filter] will be un-assigned.

        % decide Filter definition
        switch class(Filter)
            case 'strel'
                Probe=Filter;
            case {'double','int8','int16'}
                Probe=strel('disk',Filter);
            case 'cell'
                Probe=strel(Filter{:});
            case 'struct'
                Probe=strel(Filter.Type,Filter.Size);
        end

        % count layers
        CSN=numel(CSLayers);
        % cycle over all layers except last one (not-assigned)
        for cSi=1:CSN-1
            Map=CSLayers(cSi).Map;
            % remove small pieces and spikes
            Map2=imopen(Map,Probe);
            % assign new map to current layer
            CSLayers(cSi).Map=Map2;
            % assign the difference to not-assigned layer
            CSLayers(end).Map(xor(Map2,Map))=1;
        end
    end
%     function OUT=bwErode(IN,Extent)
%         % Label individual islands
%         IN=bwlabel(IN);
%         % Reduce features sizes by the Extent parameter. Minimum size is 1
%         % isolated pixel.
%         ShMask=bwmorph(IN,'shrink',Extent);
%         % Eliminate one-pixel features 
%         ShMask=bwmorph(ShMask,'clean');
%         
%         % Transfer labels to islands to be kept
%         ShMask=IN.*ShMask;
%         % find all labels to be kept (zeros plus labels)
%         OUT=zeros(size(IN));
% 
%         % Go though labels to be kept plus zero
%         for bEi=unique(ShMask)'
%             % Find island to be kept in original array and write there
%             % label (this solves 0 case)
%             OUT(IN==bEi)=bEi;
%         end
%         % Turn the output back in logical array
%         OUT=OUT>0;
%     end
end