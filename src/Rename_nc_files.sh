
for file in $(find ./get_data/*.nc -type f);
do
	echo ${file}
	var=$(ncdump -t -v time $file | grep -o 'time = "\w\w\w\w' | grep -o '[^"]*$')
	mv $file ./src/$var.nc

done
