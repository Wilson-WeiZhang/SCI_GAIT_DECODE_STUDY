function prep_patient_paths()
%PREP_PATIENT_PATHS  Add the patient and shared preprocessing folders to the MATLAB path.
%
%   Adds 01_preprocessing/patient/{shared,01_align}, 01_preprocessing/shared/shared and
%   01_preprocessing/shared/01_align. Call once at the top of any patient entry script.
%
%   01_preprocessing/shared/01_align is added with '-end': align_goni_to_eeg calls
%   manual_align_corrections from that folder.

    here = fileparts(mfilename('fullpath'));            % .../01_preprocessing/patient/shared
    prep_patient_code   = fileparts(here);                       % .../01_preprocessing/patient
    prep_patient_root   = fileparts(prep_patient_code);                   % .../01_preprocessing
    gait_root  = fileparts(prep_patient_root);                   % repository root
    prep_shared_shared = fullfile(gait_root, '01_preprocessing', 'shared', 'shared');
    prep_shared_align  = fullfile(gait_root, '01_preprocessing', 'shared', '01_align');

    addpath(here);
    addpath(fullfile(prep_patient_code, '01_align'));
    if exist(prep_shared_shared, 'dir')
        addpath(prep_shared_shared);
    else
        warning('prep_patient_paths:prep_sharedNotFound', ...
            'Expected prep_shared shared dir not found: %s', prep_shared_shared);
    end
    if exist(prep_shared_align, 'dir')
        addpath(prep_shared_align, '-end');      % appended, so a same-named function in the
    else
        warning('prep_patient_paths:prep_sharedAlignNotFound', ...
            ['Expected prep_shared 01_align dir not found: %s. Standard-protocol stage-01 aligns will ' ...
             'fail on manual_align_corrections.'], prep_shared_align);
    end
end
