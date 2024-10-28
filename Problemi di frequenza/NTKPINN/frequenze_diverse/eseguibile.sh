
python3 spectral_bias_NTK.py --alpha 1 
python3 spectral_bias_NTK.py --alpha 2 
python3 spectral_bias_NTK.py --alpha 3 
python3 spectral_bias_NTK.py --alpha 4 
python3 spectral_bias_NTK.py --alpha 5 
python3 spectral_bias_NTK.py --alpha 6 
python3 spectral_bias_NTK.py --alpha 7 
python3 spectral_bias_NTK.py --alpha 8 
python3 spectral_bias_NTK.py --alpha 9 
python3 spectral_bias_NTK.py --alpha 10
python3 spectral_bias_NTK.py --alpha 15
python3 spectral_bias_NTK.py --alpha 20
mv *png Results/baseline

python3 spectral_bias_NTK.py --alpha 1  --ntk 1
python3 spectral_bias_NTK.py --alpha 2  --ntk 1
python3 spectral_bias_NTK.py --alpha 3  --ntk 1
python3 spectral_bias_NTK.py --alpha 4  --ntk 1
python3 spectral_bias_NTK.py --alpha 5  --ntk 1
python3 spectral_bias_NTK.py --alpha 6  --ntk 1
python3 spectral_bias_NTK.py --alpha 7  --ntk 1
python3 spectral_bias_NTK.py --alpha 8  --ntk 1
python3 spectral_bias_NTK.py --alpha 9  --ntk 1
python3 spectral_bias_NTK.py --alpha 10 --ntk 1
python3 spectral_bias_NTK.py --alpha 15 --ntk 1
python3 spectral_bias_NTK.py --alpha 20 --ntk 1
mv *png Results/ntk