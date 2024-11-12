#cancella tutto 
cd Results
./delete_all.sh
cd ..

#un solo pipe

python3 extraction_PIARCH.py --sigma1 1.0 --pipes 1 --checkp "1" --mu1 3 --mu2 0.01 --path_err "alpha_0,01"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/1
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/1

python3 extraction_PIARCH.py --sigma1 1.0 --pipes 1 --checkp "1" --mu1 3 --mu2 0.1 --path_err "alpha_0,1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/1
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/1

python3 extraction_PIARCH.py --sigma1 1.0 --pipes 1 --checkp "1" --mu1 3 --mu2 1 --path_err "alpha_1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/1
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/1



python3 extraction_PIARCH.py --sigma1 5.0 --pipes 1 --checkp "5" --mu1 3 --mu2 0.01 --path_err "alpha_0,01"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/5

python3 extraction_PIARCH.py --sigma1 5.0 --pipes 1 --checkp "5" --mu1 3 --mu2 0.1 --path_err "alpha_0,1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/5

python3 extraction_PIARCH.py --sigma1 5.0 --pipes 1 --checkp "5" --mu1 3 --mu2 1 --path_err "alpha_1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/5



python3 extraction_PIARCH.py --sigma1 10 --pipes 1 --checkp "10" --mu1 3 --mu2 0.01 --path_err "alpha_0,01"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/10

python3 extraction_PIARCH.py --sigma1 10 --pipes 1 --checkp "10" --mu1 3 --mu2 0.1 --path_err "alpha_0,1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/10

python3 extraction_PIARCH.py --sigma1 10 --pipes 1 --checkp "10" --mu1 3 --mu2 1 --path_err "alpha_1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/10

#########
#due pipe
#########

python3 extraction_PIARCH.py --sigma1 1.0 --sigma2 5.0 --pipes 2 --checkp "1_5" --mu1 3 --mu2 0.01 --path_err "alpha_0,01"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/1_5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/1_5

python3 extraction_PIARCH.py --sigma1 1.0 --sigma2 5.0 --pipes 2 --checkp "1_5" --mu1 3 --mu2 0.1 --path_err "alpha_0,1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/1_5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/1_5

python3 extraction_PIARCH.py --sigma1 1.0 --sigma2 5.0 --pipes 2 --checkp "1_5" --mu1 3 --mu2 1 --path_err "alpha_1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/1_5
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/1_5



python3 extraction_PIARCH.py --sigma1 1.0 --sigma2 10.0 --pipes 2 --checkp "1_10" --mu1 3 --mu2 0.01 --path_err "alpha_0,01"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/1_10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/1_10

python3 extraction_PIARCH.py --sigma1 1.0 --sigma2 10.0 --pipes 2 --checkp "1_10" --mu1 3 --mu2 0.1 --path_err "alpha_0,1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/1_10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/1_10

python3 extraction_PIARCH.py --sigma1 1.0 --sigma2 10.0 --pipes 2 --checkp "1_10" --mu1 3 --mu2 1 --path_err "alpha_1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/1_10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/1_10



python3 extraction_PIARCH.py --sigma1 5.0 --sigma2 10.0 --pipes 2 --checkp "5_10" --mu1 3 --mu2 0.01 --path_err "alpha_0,01"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/5_10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_001/5_10

python3 extraction_PIARCH.py --sigma1 5.0 --sigma2 10.0 --pipes 2 --checkp "5_10" --mu1 3 --mu2 0.1 --path_err "alpha_0,1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/5_10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_01/5_10

python3 extraction_PIARCH.py --sigma1 5.0 --sigma2 10.0 --pipes 2 --checkp "5_10" --mu1 3 --mu2 1 --path_err "alpha_1"
mv *txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/5_10
mv *png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/FourierEmbedding/poisson/extraction_poisson/Results/mu_1/5_10









