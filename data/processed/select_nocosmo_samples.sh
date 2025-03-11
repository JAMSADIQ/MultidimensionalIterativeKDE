JAMDATADIR=/home/jam.sadiq/PopModels/projectKDE/Analysis/Data/public_data/latest_O3a_O3b_public_data

python smart_noncosmodata.py \
  --parameter1 mass_1_source --parameter2 mass_2_source --parameter3 luminosity_distance --parameter4 redshift --parameter5 chi_eff \
  --rsample 100 \
  --pathh5files / \
  --o2filesname ${JAMDATADIR}/GWTC1/*.h5  --o3afilesname ${JAMDATADIR}/GWTC2_1/NONcosmo/*.h5 --useO3b --o3bfilesname ${JAMDATADIR}/GWTC3/NONcosmo/*.h5 \
  --tag GWTC3_NOCOSMO_ORIG --eventsType BBH --min-median-mass 5 
