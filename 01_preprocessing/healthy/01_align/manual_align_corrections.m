function [pairs, is_known] = manual_align_corrections(label, n_eeg, n_goni)
%MANUAL_ALIGN_CORRECTIONS  Hand-verified EEG-marker <-> goniometer-edge pairings.
%   [pairs, is_known] = manual_align_corrections(label, n_eeg, n_goni)
%     pairs    [M x 2] 0-based [eeg_ttl_idx, goni_edge_idx], filtered to the valid range
%     is_known true if `label` has a hand-verified pairing
%
%   Used for the few recording segments whose alignment figure showed a pairing
%   error the automatic matcher could not resolve. align_goni_to_eeg applies the
%   map and asserts that it was applied. Only the verified pairs are listed; every
%   other EEG marker or goniometer edge in that segment is left unmatched.

    label = char(label);
    is_known = true;
    switch label
        case 'SUB_22_sess01b'                       % EEG#10:152 <-> Goni#11:153 (first 9 EEG / 10 goni spurious)
            e = 9:151;   g = e + 1;
        case {'SUB_27_sess01', 'SUB_31_sess01b'}    % identity EEG#i <-> Goni#i (matcher false-neg at offset drift/step)
            m = min(n_eeg, n_goni); e = 0:(m-1); g = e;
        case 'SUB_40_sess02b'                       % EEG#i <-> Goni#(i-1) (EEG#1 extra S11)
            e = 1:100;   g = e - 1;
        case 'SUB_40_sess02e'                       % identity #1-16; drop #17-19 both sides
            e = 0:15;    g = e;
        case 'SUB_42_sess01'                        % EEG#2:92<->Goni#2:92; Goni#93 fake; EEG#93:297<->Goni#94:298
            e = [1:91,  92,  93:296];
            g = [1:91,  93,  (93:296) + 1];
        otherwise
            is_known = false; pairs = zeros(0, 2); return;
    end
    keep  = (e >= 0) & (e < n_eeg) & (g >= 0) & (g < n_goni);
    pairs = [e(keep).', g(keep).'];
end
