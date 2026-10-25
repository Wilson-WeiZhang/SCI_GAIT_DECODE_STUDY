function E = cpc_env()
%CPC_ENV  Paths of the causal preprocessing, from the environment.
%   CPC_ROOT    project root holding data/ and the preprocessing code (default ${GAIT_PROJECT_ROOT})
%   CPC_EEGLAB  EEGLAB folder
%   E.host, E.root, E.eeglab, E.probe: written into every stage info file.
E.root = envdef('CPC_ROOT', getenv('GAIT_PROJECT_ROOT'));
E.eeglab = envdef('CPC_EEGLAB', fileparts(which('eeglab')));
E.probe = fullfile(E.root, '01_preprocessing', 'causal');
[~, h] = system('hostname');
E.host = strtrim(h);
if ~isempty(getenv('CPC_HOST')), E.host = sprintf('%s (%s)', getenv('CPC_HOST'), E.host); end
end

function v = envdef(name, default)
v = getenv(name);
if isempty(v), v = default; end
end
