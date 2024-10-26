echo "Questo file esegue TUTTE le estrazioni dei dati, sia per PINN che per PIARCH"
echo "Inizio con PIARCH"
cd PIARCH_extraction
./eseguibile.sh
cd ..
cd PINN_extraction
./eseguibile.sh
echo "Finito con PINN"
