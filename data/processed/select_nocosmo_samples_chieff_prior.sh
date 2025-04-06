JAMDATADIR=/home/jam.sadiq/PopModels/projectKDE/Analysis/Data/public_data/latest_O3a_O3b_public_data

python ../../src/data_scripts/get_random_PE_nocosmo.py \
  --rsample 100 \
  --inverse-chieff-prior-weight \
  --o2filesname ${JAMDATADIR}/GWTC1/*.h5  --o3afilesname ${JAMDATADIR}/GWTC2_1/NONcosmo/*.h5 --o3bfilesname ${JAMDATADIR}/GWTC3/NONcosmo/*.h5 \
  --tag GWTC3_NOCOSMO_TEST_CHIEFF_PRIOR --seed 314 --eventsType BBH --min-median-mass 5 
