SAMPLES=NOCOSMO_12345_CFPRIOR_NOREP

nohup python spin_m1m2case.py --reweight-sample-option reweight --min-bw3 0.1 --parameter1 m1 --parameter2 m2 --parameter3 chieff \
  --samples-dim1 ../../data/processed/GWTC3_${SAMPLES}_m1src.h5 \
  --samples-dim2 ../../data/processed/GWTC3_${SAMPLES}_m2src.h5 \
  --samples-dim3 ../../data/processed/GWTC3_${SAMPLES}_chieff.h5 \
  --pathplot m1m2chieff_250430/ --output-filename m1m2chieff_250430/12345_norep \
  --n-iterations 900 --buffer-start 200 --buffer-interval 100 \
  --samples-redshift ../../data/processed/GWTC3_${SAMPLES}_redshift.h5 \
  --samples-dl ../../data/processed/GWTC3_${SAMPLES}_dl.h5 \
  --samples-vt m1m2chieff_250430/VT_12345_CFPRIOR_NOREP.hdf5 & 
