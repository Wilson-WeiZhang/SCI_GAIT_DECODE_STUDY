function Y_pred_cell = predict_wiener(model, X_cell)
%PREDICT_WIENER Apply trained Wiener filter to per-trial delay-embedded EEG.
%   Y_pred_cell{k} = (X_cell{k}'+intercept) * W   → [T_k x J]
N = numel(X_cell);
Y_pred_cell = cell(1, N);
for k = 1:N
    Tk = size(X_cell{k}, 2);
    Xa = [X_cell{k}; ones(1, Tk)];           % [P+1 x T_k]
    Y_pred_cell{k} = Xa' * model.W;          % [T_k x J]
end
end
