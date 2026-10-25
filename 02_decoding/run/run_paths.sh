#!/usr/bin/env bash
# Roots of the decode chain, sourced by every shell driver in run/ (run_paths.py is the Python twin;
# python run/run_paths.py --check-sh compares the two). Every value can be overridden from the
# environment (MAIN_*). The result tree is relative to 02_decoding/, the working directory of every
# MATLAB and Python call of the chain.
MAIN_TAG=main
MAIN_GAIT=${MAIN_GAIT:-${GAIT_PROJECT_ROOT}}
MAIN_DMI=$MAIN_GAIT/02_decoding                                     # cwd of every MATLAB / pyport call on the compute server
MAIN_CODE=$MAIN_DMI/run
MAIN_H_PREP=${MAIN_H_PREP:-$MAIN_GAIT/causal_prep_output/healthy}        # causal healthy preprocessing output
MAIN_P_PREP=${MAIN_P_PREP:-$MAIN_GAIT/causal_prep_output/patient}    # causal patient preprocessing output
MAIN_H_TRIPLE=${MAIN_H_TRIPLE:-$MAIN_H_PREP/causal_triple}                 # <label>/<label>_{eeg,event,goni}.mat
MAIN_P_TRIPLE=${MAIN_P_TRIPLE:-$MAIN_P_PREP/causal_triple}
# result tree (relative to MAIN_DMI)
MAIN_R=${MAIN_R:-result/$MAIN_TAG}
MAIN_P_REST_TRIPLE=${MAIN_P_REST_TRIPLE:-$MAIN_DMI/$MAIN_R/triples/pat_restgood}   # derived rest-qc-good root (patient_rest_root.py)
MAIN_BASIS_IW=$MAIN_R/bases/iw_lso                                         # <label>_shared_delta_pca_lso.mat (80)
MAIN_BASIS_EXEC=$MAIN_R/bases/exec_lso                                     # <label>_shared_delta_pca_lso.mat (80)
MAIN_BASIS_HG=$MAIN_R/bases/hglobal                                        # shared_pca_basis.mat
MAIN_EXPORTS=$MAIN_R/exports                                               # <export_id>/<label>_{export,rest}.mat + markers
MAIN_Q=$MAIN_R/_queue                                                      # claim / done / failed markers of the decode bundles
MAIN_JOBS=$MAIN_R/_jobs                                                    # <bundle>/<decode_id>/ pushed by the hosts
MAIN_R14=result/zero_phase_prep
MAIN_PY=${MAIN_PY:-${PYTHON:-python}}            # python (numpy / pandas / h5py / scipy)
MAIN_MATLAB=${MAIN_MATLAB:-${MATLAB:-matlab}}
MAIN_SNAPSHOT_TAR=$MAIN_DMI/pyport
MAIN_DECODER_MD5=9a17ee13c766c03f41e9fe56517c4b40
