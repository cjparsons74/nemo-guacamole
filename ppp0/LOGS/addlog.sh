#!/bin/bash
#

addtolog(){
msg="$1"

echo "$(date -R) >>>>>>>>>>> $msg" >> ~/ppp/LOGS/work.log

}

addtolog "$1"
