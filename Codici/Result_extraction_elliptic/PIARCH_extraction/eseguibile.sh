

#!/bin/bash
echo "cancello prima tutti i contenuti delle cartelle dentro results"
cd Results
cd mu2_0,01
rm -rf baseline/*
rm -rf extra/*
rm -rf strong/*
rm -rf grid/*
rm -rf strong_grid_latin/*
rm -rf strong_grid/*
rm -rf strong_grid_random/*
rm -rf strong_grid_chebyshev/*
cd ..
cd mu2_1
rm -rf baseline/*
rm -rf extra/*
rm -rf strong/*
rm -rf grid/*
rm -rf strong_grid/*
rm -rf strong_grid_latin/*
rm -rf strong_grid_random/*
rm -rf strong_grid_chebyshev/*
cd ..
cd mu2_0,1
rm -rf baseline/*
rm -rf extra/*
rm -rf strong/*
rm -rf grid/*
rm -rf strong_grid/*
rm -rf strong_grid_latin/*
rm -rf strong_grid_random/*
rm -rf strong_grid_chebyshev/*
cd ..
cd ..
echo "cancellato tutto! Ora parto con le estrazioni"

#Prima estraggo mu2_0.01
python3 extraction_PIARCH.py --checkp "baseline" --mu1 3 --mu2 0.01 --path_err "alpha_0,01"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/baseline
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/baseline

python3 extraction_PIARCH.py --checkp "extra" --mu1 3 --mu2 0.01 --path_err "alpha_0,01" --extra 0
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/extra
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/extra

python3 extraction_PIARCH.py --checkp "strong" --mu1 3 --mu2 0.01 --path_err "alpha_0,01" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong

python3 extraction_PIARCH.py --checkp "grid" --mu1 3 --mu2 0.01 --path_err "alpha_0,01" --physic_sampling "grid" --parametric_sampling "grid"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/grid

python3 extraction_PIARCH.py --checkp "strong_grid" --mu1 3 --mu2 0.01 --path_err "alpha_0,01" --physic_sampling "grid" --parametric_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid

python3 extraction_PIARCH.py --checkp "strong_grid_random" --mu1 3 --mu2 0.01 --path_err "alpha_0,01" --physic_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid_random
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid_random

python3 extraction_PIARCH.py --checkp "strong_grid_latin" --mu1 3 --mu2 0.01 --path_err "alpha_0,01" --physic_sampling "grid" --parametric_sampling "latin" --strong 1 
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid_latin
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid_latin

python3 extraction_PIARCH.py --checkp "strong_grid_chebyshev" --mu1 3 --mu2 0.01 --path_err "alpha_0,01" --physic_sampling "grid" --parametric_sampling "chebyshev" --strong 1 
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid_chebyshev
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,01/strong_grid_chebyshev

#Poi estraggo mu2_1
python3 extraction_PIARCH.py --checkp "baseline" --mu1 3 --mu2 1 --path_err "alpha_1"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/baseline
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/baseline

python3 extraction_PIARCH.py --checkp "extra" --mu1 3 --mu2 1 --path_err "alpha_1" --extra 0
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/extra
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/extra

python3 extraction_PIARCH.py --checkp "strong" --mu1 3 --mu2 1 --path_err "alpha_1" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong

python3 extraction_PIARCH.py --checkp "grid" --mu1 3 --mu2 1 --path_err "alpha_1" --physic_sampling "grid" --parametric_sampling "grid"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/grid

python3 extraction_PIARCH.py --checkp "strong_grid" --mu1 3 --mu2 1 --path_err "alpha_1" --physic_sampling "grid" --parametric_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid

python3 extraction_PIARCH.py --checkp "strong_grid_random" --mu1 3 --mu2 1 --path_err "alpha_1" --physic_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid_random
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid_random

python3 extraction_PIARCH.py --checkp "strong_grid_latin" --mu1 3 --mu2 1 --path_err "alpha_1" --physic_sampling "grid" --parametric_sampling "latin" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid_latin
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid_latin

python3 extraction_PIARCH.py --checkp "strong_grid_chebyshev" --mu1 3 --mu2 1 --path_err "alpha_1" --physic_sampling "grid" --parametric_sampling "chebyshev" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid_chebyshev
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_1/strong_grid_chebyshev

#Poi estraggo mu2_0.1
python3 extraction_PIARCH.py --checkp "baseline" --mu1 3 --mu2 0.1 --path_err "alpha_0,1"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/baseline
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/baseline

python3 extraction_PIARCH.py --checkp "extra" --mu1 3 --mu2 0.1 --path_err "alpha_0,1" --extra 0
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/extra
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/extra

python3 extraction_PIARCH.py --checkp "strong" --mu1 3 --mu2 0.1 --path_err "alpha_0,1" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong

python3 extraction_PIARCH.py --checkp "grid" --mu1 3 --mu2 0.1 --path_err "alpha_0,1" --physic_sampling "grid" --parametric_sampling "grid"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/grid

python3 extraction_PIARCH.py --checkp "strong_grid" --mu1 3 --mu2 0.1 --path_err "alpha_0,1" --physic_sampling "grid" --parametric_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid

python3 extraction_PIARCH.py --checkp "strong_grid_random" --mu1 3 --mu2 0.1 --path_err "alpha_0,1" --physic_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid_random
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid_random

python3 extraction_PIARCH.py --checkp "strong_grid_latin" --mu1 3 --mu2 0.1 --path_err "alpha_0,1" --physic_sampling "grid"  --parametric_sampling "latin" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid_latin
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid_latin

python3 extraction_PIARCH.py --checkp "strong_grid_chebyshev" --mu1 3 --mu2 0.1 --path_err "alpha_0,1" --physic_sampling "grid"  --parametric_sampling "chebyshev" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid_chebyshev
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_elliptic/Result_extraction/PIARCH_extraction/Results/mu2_0,1/strong_grid_chebyshev

echo "Tutte le istruzioni eseguite"
