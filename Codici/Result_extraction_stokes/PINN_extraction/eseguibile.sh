#!/bin/bash
echo "cancello prima tutti i contenuti delle cartelle dentro results"
cd Results
cd mu_0,5
rm -rf baseline/*
rm -rf strong/*
rm -rf strong_grid/*
rm -rf grid/*
rm -rf chebyshev/*
rm -rf fourier/*
rm -rf 200_neurons/*
rm -rf fourier_tanh/*
cd ..
cd mu_1
rm -rf baseline/*
rm -rf strong/*
rm -rf strong_grid/*
rm -rf grid/*
rm -rf chebyshev/*
rm -rf fourier/*
rm -rf 200_neurons/*
rm -rf fourier_tanh/*
cd ..
cd mu_1,5
rm -rf baseline/*
rm -rf strong/*
rm -rf strong_grid/*
rm -rf grid/*
rm -rf chebyshev/*
rm -rf fourier/*
rm -rf 200_neurons/*
rm -rf fourier_tanh/*
cd ..
cd ..
echo "cancellato tutto! Ora parto con le estrazioni"

#Prima estraggo mu_0.5
python3 extraction_PINN.py --checkp "baseline" --mu 0.5 --path_err "Stokes_05" --neurons 100
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/baseline
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/baseline

python3 extraction_PINN.py --checkp "strong" --mu 0.5 --path_err "Stokes_05" --neurons 100 --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/strong
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/strong

python3 extraction_PINN.py --checkp "strong_grid" --mu 0.5 --path_err "Stokes_05" --neurons 100 --physic_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/strong_grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/strong_grid

python3 extraction_PINN.py --checkp "grid" --mu 0.5 --path_err "Stokes_05" --neurons 100 --physic_sampling "grid"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/grid

python3 extraction_PINN.py --checkp "chebyshev" --mu 0.5 --path_err "Stokes_05" --neurons 100 --physic_sampling "chebyshev"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/chebyshev
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/chebyshev

python3 extraction_PINN.py --checkp "fourier" --mu 0.5 --path_err "Stokes_05" --neurons 100 --fourier 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/fourier
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/fourier

python3 extraction_PINN.py --checkp "200_neurons" --mu 0.5 --path_err "Stokes_05" --neurons 200
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/200_neurons
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/200_neurons

python3 extraction_PINN.py --checkp "fourier_tanh" --mu 0.5 --path_err "Stokes_05" --neurons 100 --func "tanh" --fourier 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/fourier_tanh
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_0,5/fourier_tanh

#Poi estraggo mu_1
python3 extraction_PINN.py --checkp "baseline" --mu 1 --path_err "Stokes_1" --neurons 100
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/baseline
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/baseline

python3 extraction_PINN.py --checkp "strong" --mu 1 --path_err "Stokes_1" --neurons 100 --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/strong
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/strong

python3 extraction_PINN.py --checkp "strong_grid" --mu 1 --path_err "Stokes_1" --neurons 100 --physic_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/strong_grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/strong_grid

python3 extraction_PINN.py --checkp "grid" --mu 1 --path_err "Stokes_1" --neurons 100 --physic_sampling "grid"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/grid

python3 extraction_PINN.py --checkp "chebyshev" --mu 1 --path_err "Stokes_1" --neurons 100 --physic_sampling "chebyshev"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/chebyshev
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/chebyshev

python3 extraction_PINN.py --checkp "fourier" --mu 1 --path_err "Stokes_1" --neurons 100 --fourier 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/fourier
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/fourier

python3 extraction_PINN.py --checkp "200_neurons" --mu 1 --path_err "Stokes_1" --neurons 200
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/200_neurons
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/200_neurons

python3 extraction_PINN.py --checkp "fourier_tanh" --mu 1 --path_err "Stokes_1" --neurons 100 --func "tanh" --fourier 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/fourier_tanh
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1/fourier_tanh

#Prima estraggo mu_1.5
python3 extraction_PINN.py --checkp "baseline" --mu 1.5 --path_err "Stokes_15" --neurons 100
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/baseline
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/baseline

python3 extraction_PINN.py --checkp "strong" --mu 1.5 --path_err "Stokes_15" --neurons 100 --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/strong
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/strong

python3 extraction_PINN.py --checkp "strong_grid" --mu 1.5 --path_err "Stokes_15" --neurons 100 --physic_sampling "grid" --strong 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/strong_grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/strong_grid

python3 extraction_PINN.py --checkp "grid" --mu 1.5 --path_err "Stokes_15" --neurons 100 --physic_sampling "grid"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/grid
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/grid

python3 extraction_PINN.py --checkp "chebyshev" --mu 1.5 --path_err "Stokes_15" --neurons 100 --physic_sampling "chebyshev"
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/chebyshev
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/chebyshev

python3 extraction_PINN.py --checkp "fourier" --mu 1.5 --path_err "Stokes_15" --neurons 100 --fourier 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/fourier
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/fourier

python3 extraction_PINN.py --checkp "200_neurons" --mu 1.5 --path_err "Stokes_15" --neurons 200
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/200_neurons
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/200_neurons

python3 extraction_PINN.py --checkp "fourier_tanh" --mu 1.5 --path_err "Stokes_15" --neurons 100 --func "tanh" --fourier 1
mv *.png /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/fourier_tanh
mv *.txt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/Result_extraction/PINN_extraction/Results/mu_1,5/fourier_tanh

echo "Tutte le istruzioni eseguite"
