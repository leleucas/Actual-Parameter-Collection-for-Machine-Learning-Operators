ps -eo pid,user,lstart,etime,cmd | grep "./gen_fakepara" | awk '{system("kill"" "$1)}'
