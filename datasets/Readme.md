Actual data that we used in this research is available at https://drive.google.com/file/d/1d566eJK2pmq3oCyuI7IlVBg7ZxIpDwC-/view?usp=sharing

The dataset we used contains data on wind velocity over continental United States it is a part of much larger dataset retrieved from \href{https://data.caltech.edu/records/2126}{doi:10.22002/D1.2126} . It was originally obtained from USA National Renewable Energy Laboratory's (NREL) Wind Integration National Database Toolkit (WINDT). For our work we used only small subset of data with the size almost equal to the size of initially provided data (10,605 pairs against 10,495 pairs and same temporal resolution).

Each image covers a region 200 km $\times$ 200 km. 
 The grid size for low resolution image is 8 km and 2 km for high resolution (thus, each piece low-resolution data is float array of shape $25\times 25$). Data is generated for time points within 6 months interval in 2014. For each region, 256 snapshots made every 4 hours are available. 

We preprocessed data by extracting actual wind speeds from PNG files and linearly scaling them to interval [0; 1].

Train set contains 8,448 pairs of corresponding high resolution image/small resolution image. Validation and Test sets contain 1,024 pairs each. 
