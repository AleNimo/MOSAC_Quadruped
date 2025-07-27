target extended-remote :3333
monitor reset halt
load
b main
continue