for logfile in $(ls | grep prog.txt); do

	
	echo -e $"\n\n\n"________________________________$"\n"Logfile:$"\n""$logfile"$"\n"

	cat "$logfile" | tail -n 20
done

