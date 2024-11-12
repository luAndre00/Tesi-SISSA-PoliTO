cd Results
./delete_all.sh
cd ..

python3 extraction_PIARCH.py --checkp "ffe_1"    --mu 0.5 --path_err "Stokes_05" --pipes 1 --sigma1 1 
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/1
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/1

python3 extraction_PIARCH.py --checkp "ffe_5"    --mu 0.5 --path_err "Stokes_05" --pipes 1 --sigma1 5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/5
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/5

python3 extraction_PIARCH.py --checkp "ffe_10"   --mu 0.5 --path_err "Stokes_05" --pipes 1 --sigma1 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/10

python3 extraction_PIARCH.py --checkp "ffe_1_5"  --mu 0.5 --path_err "Stokes_05" --pipes 2 --sigma1 1 --sigma2 5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/1_5
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/1_5

python3 extraction_PIARCH.py --checkp "ffe_1_10" --mu 0.5 --path_err "Stokes_05" --pipes 2 --sigma1 1 --sigma2 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/1_10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/1_10

python3 extraction_PIARCH.py --checkp "ffe_5_10" --mu 0.5 --path_err "Stokes_05" --pipes 2 --sigma1 5 --sigma2 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/5_10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_05/5_10

##################################################################################################

python3 extraction_PIARCH.py --checkp "ffe_1"    --mu 1 --path_err "Stokes_1" --pipes 1 --sigma1 1 
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/1
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/1

python3 extraction_PIARCH.py --checkp "ffe_5"    --mu 1 --path_err "Stokes_1" --pipes 1 --sigma1 5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/5
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/5

python3 extraction_PIARCH.py --checkp "ffe_10"   --mu 1 --path_err "Stokes_1" --pipes 1 --sigma1 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/10

python3 extraction_PIARCH.py --checkp "ffe_1_5"  --mu 1 --path_err "Stokes_1" --pipes 2 --sigma1 1 --sigma2 5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/1_5
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/1_5

python3 extraction_PIARCH.py --checkp "ffe_1_10" --mu 1 --path_err "Stokes_1" --pipes 2 --sigma1 1 --sigma2 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/1_10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/1_10

python3 extraction_PIARCH.py --checkp "ffe_5_10" --mu 1 --path_err "Stokes_1" --pipes 2 --sigma1 5 --sigma2 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/5_10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_1/5_10


##################################################################################################

python3 extraction_PIARCH.py --checkp "ffe_1"    --mu 1.5 --path_err "Stokes_15" --pipes 1 --sigma1 1 
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/1
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/1

python3 extraction_PIARCH.py --checkp "ffe_5"    --mu 1.5 --path_err "Stokes_15" --pipes 1 --sigma1 5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/5
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/5

python3 extraction_PIARCH.py --checkp "ffe_10"   --mu 1.5 --path_err "Stokes_15" --pipes 1 --sigma1 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/10

python3 extraction_PIARCH.py --checkp "ffe_1_5"  --mu 1.5 --path_err "Stokes_15" --pipes 2 --sigma1 1 --sigma2 5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/1_5
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/1_5

python3 extraction_PIARCH.py --checkp "ffe_1_10" --mu 1.5 --path_err "Stokes_15" --pipes 2 --sigma1 1 --sigma2 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/1_10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/1_10

python3 extraction_PIARCH.py --checkp "ffe_5_10" --mu 1.5 --path_err "Stokes_15" --pipes 2 --sigma1 5 --sigma2 10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/5_10
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/stokes/extraction_stokes/Results/mu_15/5_10







