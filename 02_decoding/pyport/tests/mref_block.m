function mref_block(out_file)
%MREF_BLOCK  MATLAB ground truth for the block cross-validation folds of the readout.
%
% The block rule of run_decoder, K = min(round(outer_k), n);
% edges = round(linspace(0, n, K + 1)); fold(edges(k)+1:edges(k+1)) = k, for
% n = 1..150 and outer_k in {10, 5, 3}. Saves R.fold_<n>_<K> to out_file (-v7).
R = struct();
for outer_k = [10 5 3]
    for n = 1:150
        K = min(round(double(outer_k)), n);
        edges = round(linspace(0, n, K + 1));
        fold = zeros(n, 1);
        for k = 1:K
            idx = (edges(k) + 1):edges(k + 1);
            fold(idx) = k;
        end
        R.(sprintf('fold_%d_%d', n, outer_k)) = fold;
    end
end
save(out_file, '-struct', 'R', '-v7');
fprintf('wrote %s\n', out_file);
end
