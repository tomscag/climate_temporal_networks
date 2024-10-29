Routine to import data from Copernicus

1) Run ./Launcher.sh, adjusting for parameters

2) Run python preprocessing_models.py to:
	- Extract data
	- Regrid and combine in a single nc file
	- Compute anomalies and save them in a nc file



Routine to join data for projections
1) Run unzip "*.zip" to extract all the nc files

2) Rename all the nc files (not necessary actually)

3) Regrid the files one by one (much more computationally efficient)

3) To set the record dimension to time run ./Set_record_dimension_ncfiles.sh

4) Join all the file with ncrcat *.nc All.nc


Important: 

Climate models use different calendars, for example Hadley Centre models in CMIP6 use a 360 day calendar, 
where every month has exactly 30 days. 
Most of them use a fixed 365-day calendar, and others include leap-years.
See here for a detailed description of model's calendars: https://loca.ucsd.edu/loca-calendar/

awi_cm_1_1_mr: include leap years
CESM2: fixed 365 days/year


