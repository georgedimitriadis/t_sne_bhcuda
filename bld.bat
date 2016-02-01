xcopy %RECIPE_DIR%\bin\windows\t_sne_bhcuda.exe %PREFIX%\Scripts\t_sne_bhcuda.exe* /E
xcopy %RECIPE_DIR%\bin\windows\cudart64_75.dll %PREFIX%\Scripts\cudart64_75.dll* /E
xcopy %RECIPE_DIR%\bin\windows\cudart32_75.dll %PREFIX%\Scripts\cudart32_75.dll* /E
xcopy %RECIPE_DIR%\__init__.py %PREFIX%\Lib\site-packages\t_sne_bhcuda\__init__.py* /E
xcopy %RECIPE_DIR%\bhtsne_cuda.py %PREFIX%\Lib\site-packages\t_sne_bhcuda\bhtsne_cuda.py* /E
xcopy %RECIPE_DIR%\t_sne_spikes.py %PREFIX%\Lib\site-packages\t_sne_bhcuda\t_sne_spikes.py* /E
