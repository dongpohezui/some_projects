
NET=8.8.8
for n in $(seq 1 254); do
  ADDR=${NET}.${n}
  echo  "${ADDR}\t$(dig -x ${ADDR} +short)"
done

