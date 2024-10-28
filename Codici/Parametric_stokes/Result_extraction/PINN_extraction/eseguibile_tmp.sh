

python3 extraction_PINN.py --checkp "strong_chebyshev" --mu 0.5 --path_err "Stokes_05" --neurons 100 --physic_sampling "chebyshev" --strong 1
python3 extraction_PINN.py --checkp "strong_chebyshev" --mu 1 --path_err "Stokes_1" --neurons 100 --physic_sampling "chebyshev" --strong 1
python3 extraction_PINN.py --checkp "strong_chebyshev" --mu 1.5 --path_err "Stokes_15" --neurons 100 --physic_sampling "chebyshev" --strong 1
rm *png
