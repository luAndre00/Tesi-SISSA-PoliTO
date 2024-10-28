#Sposto prima i valori delle loss
mv baseline.npy baseline_PIARCH.npy
mv strong.npy strong_PIARCH.npy
mv strong_grid.npy strong_grid_PIARCH.npy
mv grid.npy grid_PIARCH.npy
mv chebyshev.npy chebyshev_PIARCH.npy
mv fourier.npy fourier_PIARCH.npy
mv 200_neurons.npy 200_neurons_PIARCH.npy
mv fourier_tanh.npy fourier_tanh_PIARCH.npy

cp baseline_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cp strong_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cp strong_grid_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cp grid_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cp chebyshev_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cp fourier_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cp 200_neurons_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cp fourier_tanh_PIARCH.npy /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH

#Poi tutti i checkpoint
cd lightning_logs
cd version_0
cd checkpoints
mv epoch=49999-step=50000.ckpt baseline_PIARCH.ckpt
cp baseline_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cd ..
cd ..
cd version_1
cd checkpoints
mv epoch=49999-step=50000.ckpt strong_PIARCH.ckpt
cp strong_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cd ..
cd ..
cd version_2
cd checkpoints
mv epoch=49999-step=50000.ckpt strong_grid_PIARCH.ckpt
cp strong_grid_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cd ..
cd ..
cd version_3
cd checkpoints
mv epoch=49999-step=50000.ckpt grid_PIARCH.ckpt
cp grid_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cd ..
cd ..
cd version_4
cd checkpoints
mv epoch=49999-step=50000.ckpt chebyshev_PIARCH.ckpt
cp chebyshev_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cd ..
cd ..
cd version_5
cd checkpoints
mv epoch=49999-step=50000.ckpt fourier_PIARCH.ckpt
cp fourier_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cd ..
cd ..
cd version_6
cd checkpoints
mv epoch=49999-step=50000.ckpt 200_neurons_PIARCH.ckpt
cp 200_neurons_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH
cd ..
cd ..
cd version_7
cd checkpoints
mv epoch=49999-step=50000.ckpt fourier_tanh_PIARCH.ckpt
cp fourier_tanh_PIARCH.ckpt /scratch/atataran/Tesi-SISSA-PoliTO/Codici/Parametric_stokes/check/Checkpoint/PIARCH


