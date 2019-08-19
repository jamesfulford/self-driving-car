# System Integration Project
This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the [project introduction](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Install, Compile, and Run
[Install Docker](https://docs.docker.com/engine/installation/). It's good for you.

Build a docker image.
```bash
docker build . -t capstone
```

Instantiate a docker container from the image.
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

Run the latest code. This will compile the catkin workspace, source the correct settings, and launch the program inside of ROS.
```bash
bash run.sh
```

Run the Simulator.

### Training Environment
Build a docker image (based on Term 1 Docker environment).
```bash
cd data
docker build . -t model
```

Run model.py in a new docker container
```bash
docker run -v $PWD:/data --rm -it model
```

### Real world testing
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
