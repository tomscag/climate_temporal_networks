# Change record dimension of every nc files

for file in $(find ./*.nc -type f);
do
        echo ${file}
	ncks -O --mk_rec_dmn time $file $file
done

