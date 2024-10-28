
python3 spectral_bias.py --alpha 1    --name baseline
python3 spectral_bias.py --alpha 50   --name baseline
python3 spectral_bias.py --alpha 100  --name baseline
python3 spectral_bias.py --alpha 150  --name baseline

python3 spectral_bias.py --alpha 1    --name fourier --fourier 1 
python3 spectral_bias.py --alpha 50   --name fourier --fourier 1 
python3 spectral_bias.py --alpha 100  --name fourier --fourier 1 
python3 spectral_bias.py --alpha 150  --name fourier --fourier 1 
