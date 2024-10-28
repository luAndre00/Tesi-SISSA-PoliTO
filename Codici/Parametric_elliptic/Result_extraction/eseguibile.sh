echo "Questo file esegue TUTTE le estrazioni dei dati, sia per PINN che per PIARCH"
echo "Inizio con PINN"
cd PINN_extraction
./eseguibile.sh
cd ..
echo "Finito con PINN"
echo "Inizio con PIARCH"
cd PIARCH_extraction
./eseguibile.sh
cd ..
echo "Finito con PIARCH"
