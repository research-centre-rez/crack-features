function[W,Edges,Bins,C]=weightcounts(varargin)
% [Weights,Edges,Bins,Counts]=weightcounts(X)
% [Weights,Edges,Bins,Counts]=weightcounts(X,params)
%   where params are exact match of histcount function parameters
if nargin<1
    error('Not enough input parameters!');
end
Normalization='count';
X=varargin{1};
N=sum(X,'all');
for ii=1:nargin
    if ischar(varargin{ii})&&strcmpi(varargin{ii},'Normalization')
        Normalization=varargin{ii+1};
        varargin{ii+1}='count';
    end
end

[C,Edges,Bins]=histcounts(varargin{:});
BW=diff(Edges);

W=nan(size(C));
for ii=1:numel(C)
    W(ii)=sum(X(Bins==ii),'all');
end

switch lower(Normalization)
    case 'count'
    case 'countdensity'
        W=W./BW;
        C=C./BW;
    case 'cumcount'
        W=cumsum(W);
        C=cumsum(C);
    case 'probability'
        W=W/N;
        C=C/numel(X);
    case 'pdf'
        W=W./(BW*N);
        C=C./(BW*numel(X));
    case 'cdf'
        W=cumsum(W)/N;
        C=cumsum(C)/numel(X);
end

end