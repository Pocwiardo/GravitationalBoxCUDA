# CUDA Gravitational Box

The project simulates the movement of multiple particles (celestial bodies) in the gravitational field using multithreading in CUDA. Simulation data is saved to a csv file. Additional Python script allows visualization of the simulation.

## Functionality

Particles are defined as objects of the Planet class. Initial coordinates, velocities and masses of particles
are random (within a reasonable range). The sources of gravity, are objects of the GravitySource class and their coordinates are also random. The user has the option
to define the number of gravity sources.

In the main loop of the program, appropriate variables are copied, the main kernel is executed and
then the coordinates are saved to a csv file. Columns indicate x,y coordinates of
subsequent planets, and the rows - coordinates for subsequent time units.
The kernel calculates subsequent planetary coordinates. For each planet calculations
are executed on a separate thread. The resultant force from all sources is calculated in the loop
(we assume that there are so few sources that loop calculations are faster). The force of gravity
is calculated using the gravity equation, and acceleration - from the second law of dynamics. Location changes based on velocity, and velocity changes based on acceleration.

## Visualisation

Coordinates of the planets can be visualised in Python using matplotlib library. Simulation of hundreds of thousands planets makes an interesting visualisation. We can observe that with random velocities and positions, although most particles leave their system or fall on the gravity source, many move stable on the orbit.

![gif](./images_readme/example.gif)
