function ev = trial_filter(events, condition)
%TRIAL_FILTER Select events by condition + qc='good'.
%   ev = trial_filter(events, 'MI')   selects code='gaitimagery'
%   ev = trial_filter(events, 'EXEC') selects code='gaitexecution'
%   In both cases qc='good' is required.

switch condition
    case 'MI'
        target = 'gaitimagery';
    case 'EXEC'
        target = 'gaitexecution';
    otherwise
        error('condition must be ''MI'' or ''EXEC''');
end

mask = strcmp({events.code}, target) & strcmp({events.qc}, 'good');
ev = events(mask);
end
