function matrix=DOEcos2D_refl(neurons,phim)
matrix = zeros(neurons);
nside = sqrt(neurons);
if mod(nside,1)~=0
    error('Number of neurons must be square.')
end
if mod(nside,2)==0
    error('Side length (=sqrt(neurons)) must be odd.')
end
[ccol, crow] = meshgrid(0:nside-1,0:nside-1);
cfun = @(n)besselj(n,phim);

mirror = zeros(nside);
mirror(:) = 0:neurons-1;
mirror = rot90(rot90(mirror));

for r=0:neurons-1
    mr = mirror(r+1); %mr is mirror neuron of m
    matrix(r+1,:) = arrayfun(cfun,crow(:)-mod(mr,nside)).*arrayfun(cfun,ccol(:)-floor(mr/nside));
end
