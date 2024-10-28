#Sposto prima i valori delle loss
mv baseline.npy baseline_pinn.npy
mv strong.npy strong_pinn.npy
mv strong_grid.npy strong_grid_pinn.npy
mv grid.npy grid_pinn.npy
mv chebyshev.npy chebyshev_pinn.npy
mv fourier.npy fourier_pinn.npy
mv 200_neurons.npy 200_neurons_pinn.npy
mv fourier_tanh.npy fourier_tanh_pinn.npy

cp baseline_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cp strong_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cp strong_grid_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cp grid_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cp chebyshev_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cp fourier_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cp 200_neurons_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cp fourier_tanh_pinn.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN

#Poi tutti i checkpoint
cd lightning_logs
cd version_0
cd checkpoints
mv epoch=49999-step=50000.ckpt baseline_pinn.ckpt
cp baseline_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cd ..
cd ..
cd version_1
cd checkpoints
mv epoch=49999-step=50000.ckpt strong_pinn.ckpt
cp strong_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cd ..
cd ..
cd version_2
cd checkpoints
mv epoch=49999-step=50000.ckpt strong_grid_pinn.ckpt
cp strong_grid_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cd ..
cd ..
cd version_3
cd checkpoints
mv epoch=49999-step=50000.ckpt grid_pinn.ckpt
cp grid_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cd ..
cd ..
cd version_4
cd checkpoints
mv epoch=49999-step=50000.ckpt chebyshev_pinn.ckpt
cp chebyshev_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cd ..
cd ..
cd version_5
cd checkpoints
mv epoch=49999-step=50000.ckpt fourier_pinn.ckpt
cp fourier_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cd ..
cd ..
cd version_6
cd checkpoints
mv epoch=49999-step=50000.ckpt 200_neurons_pinn.ckpt
cp 200_neurons_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN
cd ..
cd ..
cd version_7
cd checkpoints
mv epoch=49999-step=50000.ckpt fourier_tanh_pinn.ckpt
cp fourier_tanh_PINN.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PINN


