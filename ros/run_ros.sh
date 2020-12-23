mode=$1

if [[ "$mode" == "site" ]]; then
	launch_file='launch/site.launch';
else
	launch_file='launch/styx.launch';
fi

echo "Running launch file $launch_file"
catkin_make && \
source devel/setup.sh && \
roslaunch $launch_file
