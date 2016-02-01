#T-SNE with CUDA and Barnes Hut (with Python wrapper)

This is an extension of the C++ t-sne with barnes-hut algorithm written by Laurens van der Maaten. The original code can be found here: [lvdmaaten/bhtsne](https://github.com/lvdmaaten/bhtsne/). There are three main differences.


1. The cuda code targets not the actual t-sne code but the part that generates the perplexities. More specifically it implements a fast cuda algorithm for calculating euclidean distances. That way the most time consuming step in the barnes-hut algorithm (calculating the euclidean distances of sample pairs) becomes much fater.
The code for the euclidean distance calculation in cuda was taken from here:
*Chang, Darjen, Nathaniel A. Jones, Dazhuo Li, and Ming Ouyang. “Compute Pairwise Euclidean Distances of Data Points with GPUs.” In Proceedings of the IASTED International Symposium Computational Biology and Bioinformatics. Orlando, Florida, USA, 2008.*
The code for the dev_array was taken from Valerio Restocchi's blog, here: [dev_array: A Useful Array Class for CUDA](https://www.quantstart.com/articles/dev_array_A_Useful_Array_Class_for_CUDA).
Currenlty the cuda code works only with the bh part of the original code (if the theta parameter is set to larger than 0). The exact part of the code (theta = 0) is left the same (requires BLAS or a windows equivalent to calculate the euclidean distances). In the future the cuda euclidean distances calculation will be implemented in the exact part of the code to allow even faster calculation than BLAS.
The current cuda implementation checks the amount of available gpu memory and uses a user defined percentage of it (set by default to 80% of the available gpu memory). If you set the gpu_mem parameter to 0 then the code runs on the cpu (as per the original). A value larger than 0 and smaller than 1 tells the program to use that percentage of the available gpu memory. If the memory required to store all distance pairs (4\*N\*N bytes where N is the number of samples) is larger than what the gpu can offer then the algorithm iterates the saving of the distances in chunks that can be temporarily held in gpu memory. Of course if the available RAM is smaller than 4\*N\*N then the program crashes (maybe a mmap implementation in the future wouldn't add too much time in the reading of the distances from hard disk).
The cuda was written using CUDA 7.5 (January 2016) but should work with anything over 5.0 (untested claim). The code generation is set to compute_35,sm_35 but compute_20,sm_20 might still work (again untested claim).

2. The second change is that the code now is a Visual Studio (2013) project using the nvcc compiler (through the Nsight Visual Studio Edition 4.0). Maybe there will be a make file in the future for cross platform compilation but for now if you are using Linux or Mac you will have to make your own project. There shouldn't be any windows specific libraries and I have used int_least64_t instead of long long to make matters a bit easier. Also you might want to have a look at the original Maatens t-sne code on ideas on how to get started in \*nix* systems.

3. The final (and most minor change) is a small re-writing of the python wrapper (originaly developed by Pontus Stenetorp). Now you can use it to generate the data.dat file or read the results.dat file without actually running the t-sne code. The data.dat file also carries a header that passes the required parameters into the t_sne_gpu executable. It also allows the user to call the sklearn implementation of t-sne (no cuda, no C++ = quite more slow) in case something has gone wrong and the C++ executable doe not run.


##Notes for use
###Installation
 This is a conda package so you can install it using conda install -c georgedimitriadis t_sne_bhcuda. This will add the t_sne_bhcuda executable into the Scripts folder of the python environment that you installed the package in. The t_sne() function in the bhtsne_cuda script of the module will call this executable.

 **Important note for Windows users:** If you have not used cuda before, then you need to be aware that windows by default will stop and restart the nvidia driver if it thinks that the gpu is stuck. That by default will happen if the gpu does anything that takes longer than 2 seconds. The current code will not work under these conditions with sample sizes over a certain number. If the code requires more than 2 seconds to calculate the distances then windows will restart the driver and the program will fail (you will get a notification of this at the bottom of your screen). In order to get windows off your back do what he says: [Nvidia Display Device Driver Stopped Responding And Has Recovered Successfully (FIX)](https://www.youtube.com/watch?v=QQJ9T0oY-Jk). Also have a look here for MSDN info on the relative registry values [TDR Registry Keys](https://msdn.microsoft.com/en-us/library/windows/hardware/ff569918%28v=vs.85%29.aspx).

###Use
Read the example in the bhtsne_cuda script on how to call t_sne. 

If you want to use t-sne on spikes as detected by the spikedetect portion of the phy module then check out the t_sne_spikes script. The result of this operation is a 2 x Nspikes array that will be saved in the same directory that the .kwik file you provided to the function is in. The phy module will be able to detect this array (saved in .npy format) and display in its GUI the results of the t-sne operation superimposing clustering information if it exists.
