

python3 spectral_bias.py --alpha 150  --sigma 0.01  --fourier 1
python3 spectral_bias.py --alpha 150  --sigma 0.05  --fourier 1
python3 spectral_bias.py --alpha 150  --sigma 0.1   --fourier 1
python3 spectral_bias.py --alpha 150  --sigma 0.5   --fourier 1
python3 spectral_bias.py --alpha 150  --sigma 1     --fourier 1 
python3 spectral_bias.py --alpha 150  --sigma 5     --fourier 1
python3 spectral_bias.py --alpha 150  --sigma 10    --fourier 1
mv *loss* loss
mv *alpha* soluzioni
