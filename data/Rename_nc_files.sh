
for file in $(find ./*.nc -type f);
do
	echo ${file}
	var=$(ncdump -t -v time $file | grep -o 'time = "\w\w\w\w' | grep -o '[^"]*$')
	mv $file ./$var.nc

done
