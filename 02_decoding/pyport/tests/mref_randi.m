function mref_randi(out_file)
%MREF_RANDI  MATLAB ground truth for the randi() draws of the control conditions.
%
% randi(imax) after rng(seed, 'twister'), a single draw and two sequential draws, for
% fixed seeds and maxima; saved to out_file (-v7).
seeds = [20260905 + 1009*(1:60), 20260820 + 1009*(1:60), 1, 7, 123456];
imaxs = [1 2 3 7 10 59 1000 1871 2406 99991];
R = zeros(numel(seeds), numel(imaxs), 2);
for i = 1:numel(seeds)
  for k = 1:numel(imaxs)
    rng(seeds(i), 'twister'); a = randi(imaxs(k)); b = randi(imaxs(k));
    R(i, k, 1) = a; R(i, k, 2) = b;
  end
end
save(out_file, 'R', 'seeds', 'imaxs', '-v7');
end
