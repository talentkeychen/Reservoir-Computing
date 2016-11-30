function matrix=DOEcos2D_trans(neurons,phim)
matrix = zeros(neurons);
nside = sqrt(neurons);
if mod(nside,1)~=0
    error('Number of neurons must be square.')
end
[ccol, crow] = meshgrid(0:nside-1,0:nside-1);
cfun = @(n)besselj(n,phim);
for r=0:neurons-1
    matrix(r+1,:) = arrayfun(cfun,crow(:)-mod(r,nside)).*arrayfun(cfun,ccol(:)-floor(r/nside));
end
