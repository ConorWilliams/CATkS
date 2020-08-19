# CATkS

Instructions for compilation:
```
mkdir build

cd external
tar -xzf nauty27r1.tar.gz
cd nauty27r1
./configure
make

cd ../../build
cmake ..
make
```


Instructions for running:
```
cd build
mkdir run
cd run
mkdir dump
../diffusion ../../data/PotentialA.fs run.out
```

Frame for each time step `dump/` will contain a corresponding `.xyz` file of all atomic coordinates (and vacancy centres).

The file `run.out` highlights just vacancies and H atoms with headings:

time, activation energy, vacancy coordinates [x, y, z]..., hydrogen coordinates [x, y, z]...
