#!/bin/bash
# Lithium battery Amp-hour timer
# April 2022
#
# ref: https://simonprickett.dev/controlling-raspberry-pi-gpio-pins-from-bash-scripts-traffic-lights/
#
# Pi header
#
#     GND   
#     |     
# 2 4 6 8 --GPIO14
# 1 3 5 7
# ^
# |
# square pin
#
# usage:
#   sudo ./ah_timer.sh

# Common path for all GPIO access
BASE_GPIO_PATH=/sys/class/gpio
DISCHARGE_CURRENT_AMPS=25.0
SECONDS_PER_HOUR=3600.0

# Utility function to export a pin if not already exported
exportPin()
{
  if [ ! -e $BASE_GPIO_PATH/gpio$1 ]; then
    echo "$1" > $BASE_GPIO_PATH/export
  fi
}

# Utility function to set a pin as an output
setInput()
{
  echo "in" > $BASE_GPIO_PATH/gpio$1/direction
}

exportPin 14
setInput 14
start_time=$(date +%s)

# stop when we get a 0 on GPIO14
discharging=`cat $BASE_GPIO_PATH/gpio14/value`
while [[ $discharging -eq 1 ]]
do
    sleep 1
    current_time=$(date +%s)
    discharging=`cat $BASE_GPIO_PATH/gpio14/value`
    AH=$(python -c "print(($current_time-$start_time)*$DISCHARGE_CURRENT_AMPS/$SECONDS_PER_HOUR)" )
    printf "%5.1f AH\n" $AH
done

