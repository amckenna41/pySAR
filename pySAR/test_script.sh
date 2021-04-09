#!/bin/bash
for i in {1..20}; do
  echo -e "\nROUND $i\n"
  for j in {1..10}; do
    main.py &
  done
  wait
done 2>/dev/null
