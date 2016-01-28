@Echo OFF
Echo Launch dir: "%~dp0"
Echo Current dir: "%CD%"

"C:\Program Files\7-Zip\7z.exe" a -ttar t_sne_bhcuda.tar "%~dp0"__init__.py bhtsne_cuda.py "%~dp0"\bin\windows\t_sne_bhcuda.exe "%~dp0"\bin\windows\cudart32_75.dll "%~dp0"\bin\windows\cudart64_75.dll

"C:\Program Files\7-Zip\7z.exe" a -tgzip t_sne_bhcuda.tar.gz "%~dp0"t_sne_bhcuda.tar

del "%~dp0"t_sne_bhcuda.tar

Pause&Exit