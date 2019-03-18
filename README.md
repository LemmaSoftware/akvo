# Akvo 

Akvo provides processing of surface NMR data. It aims to be simple to use yet flexible for accommodating changes to processing flow. Akvo is written primarily in Python 3 with a small amount of R as well. The application is written around a Qt GUI with plotting provided by Matplotlib.

The bleeding-edge code may be accesed using the git client 
```
git clone https://git.lemmasoftware.org/akvo.git  
```
or, using our GitHub mirror 
```
git clone https://github.com/LemmaSoftware/akvo.git  
```

## Installation 

Installation is straightforward. The only prerequisite that is sometimes not properly handled is PyQt5 which sometimes needs to be manually installed. 
```
python3 setup.py build 
python3 setup.py install
```

Alternatively, release versions can be installed via pip
```
pip install akvo
```


## Team 

Akvo is developed by several teams including the University of Utah.

## Capabilities 

Akvo currently has preprocessing capabilities for VistaClara GMR data. 

## Benefits 

Processing steps are retained and logged in the processed file header, which is written in YAML. 
This allows data processing to be repeatible, which is a major benefit. 

## Languages

Akvo is written primarily in Python 3. The graphical user unterface is written in PyQt5.  An interface to modelling software written in C++ (Lemma and Merlin) is in development. 
