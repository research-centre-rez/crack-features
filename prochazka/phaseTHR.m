function[Map,Layers,Verbose]=phaseTHR(IMG,Phases,varargin)
%% Data preallocation
% Estimate recipy length
Nv=max(1,numel(varargin));
% Preallocate struct for storing steps
Verbose(Nv)=struct('Img',nan,'command',{''});
% preallocate filter for actual number of steps
Used=false(1,Nv);

% colour matching method. true=La*b* distance evaluation, otherwise a*b*
% distance is used
LABdistance=false;
% Lightness thresholding limits
L_out=[1 99];

% Extract phases to be detected
Phases=Phases([Phases.Detect]);


Cracks=zeros(size(IMG(:,:,1)));
% Preallocate output structure
Np=numel(Phases);
Layers(Np+1,1)=struct('Map',nan,'Label','n/a','RGB',zeros(1,3),...
    'Partition',nan,'Numel',nan,'Count',nan);
%% Preprocess image
ii=0;
while(ii<numel(varargin))
    ii=ii+1;
    switch lower(varargin{ii})
        case {'-ab' '-abdistance' '-record'}
            Used(ii)=true;
            Verbose(ii).Command=varargin(ii);
            Verbose(ii).Img=IMG;
            LABdistance=false;
        case {'-lab' '-labdistance'}
            Used(ii)=true;
            Verbose(ii).Command=varargin(ii);
            Verbose(ii).Img=IMG;
            LABdistance=true;
        case {'-gauss'}
            Used(ii)=true;
            Verbose(ii).Command=varargin(ii:ii+1);
            IMG=imgaussfilt(IMG,varargin{ii+1});
            Verbose(ii).Img=IMG;
            ii=ii+1;
        case {'-imadjust','-imadj'}
                Used(ii)=true;
            if ii+1<=Nv&&isnumeric(varargin{ii+1})
                IMG=imadjust(IMG,varargin{ii+1});
                Verbose(ii).Command=varargin(ii:ii+1);
                ii=ii+1;
            else
                IMG=imadjust(IMG);
                Verbose(ii).Command=varargin(ii);
            end
                Verbose(ii).Img=IMG;
        case {'-thr' '-threshold'}
            if ii+1<=Nv||isnumeric(varargin{ii+1})
                switch numel(varargin{ii+1})
                    case 0
                        warning('incorect threshold values')
                    case 1
                        L_out(1)=varargin{ii+1};
                    case 2
                        L_out=varargin{ii+1};
                    otherwise
                        warning('incorrect threshold values')
                        L_out=varargin{ii+1}(1:2);
                end
                Verbose(ii).Command=varargin(ii:ii+1);
                Verbose(ii).Img=IMG;
                ii=ii+1;
            else
                warning('no threshold defined, default used istead.')

            end
        case {'-cracks'}
            Cracks=varargin{ii+1};
            Verbose(ii).Command=varargin(ii,ii+1);
            Verbose(ii).Img=Cracks;
            ii=ii+1;
    end
end

%% RGB to L*a*b* conversions
% Convert image and phase samples

Cracks=double(Cracks);


LAB=rgb2lab(IMG);
Markers=rgb2lab(vertcat(Phases.Colors));
% Extract L a* b* chanells
IL=LAB(:,:,1).*~Cracks;
IA=LAB(:,:,2).*~Cracks;
IB=LAB(:,:,3).*~Cracks;
class(IL)
ML=Markers(:,1);
MA=Markers(:,2);
MB=Markers(:,3);

%% Calculate distances in the (L)a*b* colourspace
% Preallocate distancing matrix
Distances=nan([size(IL),Np]);
% calculate distances
if LABdistance
    for ii=1:Np
        Distances(:,:,ii)=((IA-MA(ii)).^2+...
                           (IB-MB(ii)).^2+...
                           (IL-ML(ii)).^2).^.5;
    end
else
    for ii=1:Np
        Distances(:,:,ii)=((IA-MA(ii)).^2+...
                           (IB-MB(ii)).^2).^.5;
    end
end
% Find the lowest distance
[~,Map]=min(Distances,[],3);
% Assign white and black pixels as N+1th layer - no signal
% Find black pixels (LAB:0,0,0) - no data
Map(IL<L_out(1))=0;
% Find white pixels (LAB:100,0,0) - false data
Map(IL>L_out(2))=0;

%% Create output structure
for ii=1:Np
    MapII=Map==ii;
    Layers(ii).Map=MapII;
    Layers(ii).Label=Phases(ii).Labels;
    Layers(ii).RGB=Phases(ii).Colors;
    Layers(ii).Numel=sum(MapII,'all');
    Layers(ii).Partition=Layers(ii).Numel/numel(MapII);
    Layers(ii).Count=max(bwlabel(MapII),[],'all');
end
MapII=~Map;
Layers(end).Map=MapII;
Layers(end).Numel=sum(MapII,'all');
Layers(end).Partition=Layers(end).Numel/numel(MapII);
Layers(end).Count=max(bwlabel(MapII),[],'all');
end