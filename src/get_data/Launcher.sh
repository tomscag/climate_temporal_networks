
for item in $(seq 1994 1994); do
	fname="${item}.py"
	echo "year=$item" >>${fname}
	cat import_daily_aggregate_loop.py>>${fname}
        nohup python ${fname} &	

	done

