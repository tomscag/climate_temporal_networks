
for item in $(seq 1970 2022); do
	fname="${item}.py"
	echo "year=$item" >>${fname}
	cat import_daily_aggregate_loop.py>>${fname}
        nohup python ${fname} &	

	done

