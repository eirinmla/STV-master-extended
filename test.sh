for TEST in simplemod{1..8}
do
	echo
	echo $TEST
	echo init | python3 main.py verify ucl --filename $TEST
	echo
done
