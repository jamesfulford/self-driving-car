# System Integration Project
This is the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car.

## Operations

### Setup

[Install Docker](https://docs.docker.com/engine/installation/). It's good for you.

Build a docker image.
```bash
docker build . -t capstone
```

Instantiate a docker container from the image.
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Compile and Run ROS code

Run the latest code. This will compile the catkin workspace, source the correct settings, and launch the program inside of ROS.
```bash
bash run.sh
```

Run the Simulator.

### Train model
```bash
cd /capstone/data
python model.py
```

## Simulate Real World Testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.

2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```

3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```

4. Launch your project in site mode
```bash
roslaunch launch/site.launch
```

5. Confirm that traffic light detection works on real life images
